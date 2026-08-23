"""Minimal ResNet50 + ArcFace training and LFW evaluation utilities."""

from __future__ import annotations

import csv
import io
import math
import os
import struct
from array import array
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.transforms import v2


IMAGE_SIZE = 112
EMBEDDING_DIM = 512
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)


def train_transform():
    return v2.Compose(
        [
            v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )


def evaluation_transform():
    return v2.Compose(
        [
            v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )


def _read_record(stream, offset):
    """Read one DMLC RecordIO entry at an offset from an MXNet .idx file."""

    stream.seek(offset)
    chunks = []
    while True:
        header = stream.read(8)
        if len(header) != 8:
            raise ValueError(f"Invalid RecordIO header at offset {offset}")
        magic, encoded_length = struct.unpack("<II", header)
        if magic != 0xCED7230A:
            raise ValueError(f"Invalid RecordIO magic at offset {offset}")
        flag = encoded_length >> 29
        length = encoded_length & ((1 << 29) - 1)
        padded_length = (length + 3) & ~3
        chunk = stream.read(padded_length)
        if len(chunk) != padded_length:
            raise ValueError(f"Truncated RecordIO entry at offset {offset}")
        chunks.append(chunk[:length])
        if flag in (0, 3):
            return b"".join(chunks)
        if flag not in (1, 2):
            raise ValueError(f"Invalid RecordIO continuation flag {flag}")
        chunks.append(struct.pack("<I", magic))


def _unpack_mx_record(record):
    """Return an MXImageRecord label and its encoded image payload."""

    header_size = struct.calcsize("<IfQQ")
    flag, label, _, _ = struct.unpack("<IfQQ", record[:header_size])
    payload = record[header_size:]
    if flag:
        label = np.frombuffer(payload, dtype="<f4", count=flag).copy()
        payload = payload[flag * 4 :]
    return label, payload


def _parse_index_offsets(lines):
    keys = array("q")
    offsets = array("q")
    for line in lines:
        if not line.strip():
            continue
        key, offset = map(int, line.rstrip().split("\t"))
        keys.append(key)
        offsets.append(offset)
    if not keys:
        raise ValueError("train.idx is empty")

    keys = np.asarray(keys, dtype=np.int64)
    offsets = np.asarray(offsets, dtype=np.int64)
    sorted_keys = np.sort(keys)
    if np.any(sorted_keys[1:] == sorted_keys[:-1]):
        raise ValueError("train.idx contains duplicate keys")
    return keys, offsets


def _is_image_record(payload):
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
    except (OSError, ValueError):
        return False
    return True


class MXRecordIODataset(Dataset):
    """Read InsightFace train.rec/train.idx directly without an MXNet install."""

    def __init__(self, root, transform=None, max_samples=None):
        self.root = Path(root)
        self.rec_path = self.root / "train.rec"
        self.idx_path = self.root / "train.idx"
        self.transform = transform or evaluation_transform()
        for path in (self.rec_path, self.idx_path):
            if not path.exists():
                raise FileNotFoundError(path)

        with self.idx_path.open(encoding="utf-8") as stream:
            keys, offsets = _parse_index_offsets(stream)

        first_position = int(np.argmin(offsets))
        with self.rec_path.open("rb") as stream:
            first_label, _ = _unpack_mx_record(
                _read_record(stream, int(offsets[first_position]))
            )
        has_metadata = isinstance(first_label, np.ndarray) and len(first_label) >= 2
        if has_metadata:
            image_end, metadata_end = map(int, first_label[:2])
            if keys[first_position] != 0:
                raise ValueError("RecordIO metadata does not match train.idx")
            offsets = offsets[(keys > 0) & (keys < image_end)]
            metadata_class_count = (
                metadata_end - image_end if metadata_end > image_end else metadata_end
            )
        else:
            metadata_class_count = None

        # Index keys can be shuffled. Physical offsets retain the writer's
        # identity-grouped order, which is required for a useful prefix subset.
        offsets = np.sort(offsets)

        property_path = self.root / "property"
        if property_path.exists():
            values = property_path.read_text(encoding="utf-8").replace(",", " ").split()
            class_count = int(values[0])
        elif metadata_class_count:
            class_count = metadata_class_count
        else:
            raise ValueError("Cannot determine class count: property and metadata missing")

        self.labels = None
        self.skipped_records = 0
        if max_samples and max_samples < len(offsets):
            selected_offsets = []
            labels = []
            with self.rec_path.open("rb") as stream:
                for offset in offsets:
                    label, image_bytes = _unpack_mx_record(
                        _read_record(stream, int(offset))
                    )
                    if not _is_image_record(image_bytes):
                        self.skipped_records += 1
                        continue
                    selected_offsets.append(offset)
                    labels.append(int(label[0] if isinstance(label, np.ndarray) else label))
                    if len(selected_offsets) == max_samples:
                        break
            offsets = np.asarray(selected_offsets, dtype=np.int64)
            if not len(offsets):
                raise ValueError("No readable images found in train.rec")
            classes, self.labels = np.unique(labels, return_inverse=True)
            self.classes = classes.tolist()
        else:
            self.classes = range(class_count)

        self.offsets = offsets
        self._stream = None
        self._pid = None

    def __len__(self):
        return len(self.offsets)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_stream"] = None
        state["_pid"] = None
        return state

    def _get_stream(self):
        pid = os.getpid()
        if self._stream is None or self._pid != pid:
            if self._stream is not None:
                self._stream.close()
            self._stream = self.rec_path.open("rb")
            self._pid = pid
        return self._stream

    def __getitem__(self, index):
        record = _read_record(self._get_stream(), int(self.offsets[index]))
        label, image_bytes = _unpack_mx_record(record)
        if self.labels is not None:
            label = self.labels[index]
        elif isinstance(label, np.ndarray):
            label = label[0]
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = image.convert("RGB")
        return self.transform(image), int(label)


class FaceEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=EMBEDDING_DIM, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = resnet50(weights=weights)
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, images):
        return F.normalize(self.backbone(images), dim=1)


class ArcMarginProduct(nn.Module):
    """ArcFace additive angular-margin classification head."""

    def __init__(self, embedding_dim, class_count, scale=64.0, margin=0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(class_count, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.set_margin(margin)

    def set_margin(self, margin):
        self.cos_margin = math.cos(margin)
        self.sin_margin = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.margin_correction = math.sin(math.pi - margin) * margin

    def forward(self, embeddings, labels, return_cosine=False):
        # ArcFace's sqrt/angle calculation is unstable in float16 near +/-1.
        with torch.autocast(device_type=embeddings.device.type, enabled=False):
            embeddings = F.normalize(embeddings.float(), dim=1)
            weight = F.normalize(self.weight.float(), dim=1)
            cosine = F.linear(embeddings, weight).clamp(-1 + 1e-7, 1 - 1e-7)
            sine = torch.sqrt((1.0 - cosine.square()).clamp_min(1e-7))
            phi = cosine * self.cos_margin - sine * self.sin_margin
            phi = torch.where(
                cosine > self.threshold, phi, cosine - self.margin_correction
            )
            one_hot = F.one_hot(
                labels, num_classes=self.weight.shape[0]
            ).to(cosine.dtype)
            margin_logits = (
                one_hot * phi + (1.0 - one_hot) * cosine
            ) * self.scale
            if return_cosine:
                return margin_logits, cosine
            return margin_logits


def read_lfw_pairs(pairs_csv, image_root):
    """Return (path1, path2, is_same) tuples from this project's pairs.csv."""

    image_root = Path(image_root)
    pairs = []
    with Path(pairs_csv).open(encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream):
            if not row or row[0] == "name":
                continue
            if row[3]:
                name1, number1, name2, number2 = row[:4]
                is_same = False
            else:
                name1, number1, number2 = row[:3]
                name2 = name1
                is_same = True
            path1 = image_root / name1 / f"{name1}_{int(number1):04d}.jpg"
            path2 = image_root / name2 / f"{name2}_{int(number2):04d}.jpg"
            pairs.append((path1, path2, is_same))
    return pairs


class ImagePathDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = list(paths)
        self.transform = transform or evaluation_transform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
        return self.transform(image), str(self.paths[index])


@torch.inference_mode()
def extract_embeddings(model, loader, device):
    model.eval()
    embeddings = {}
    for images, paths in loader:
        images = images.to(device)
        values = F.normalize(
            model(images) + model(torch.flip(images, dims=[3])), dim=1
        ).cpu().numpy()
        embeddings.update(zip(paths, values))
    return embeddings


def evaluate_lfw(pairs, embeddings, fold_size=600):
    """Evaluate with 10-fold threshold selection when given the standard 6000 pairs."""

    similarities = np.asarray(
        [
            float(np.dot(embeddings[str(path1)], embeddings[str(path2)]))
            for path1, path2, _ in pairs
        ]
    )
    labels = np.asarray([is_same for _, _, is_same in pairs], dtype=bool)
    if len(pairs) % fold_size:
        raise ValueError("Pair count must be divisible by fold_size")

    thresholds = np.linspace(-1.0, 1.0, 2001)
    fold_accuracies = []
    fold_thresholds = []
    fold_count = len(pairs) // fold_size
    for fold in range(fold_count):
        test_mask = np.zeros(len(pairs), dtype=bool)
        test_mask[fold * fold_size : (fold + 1) * fold_size] = True
        train_predictions = similarities[~test_mask, None] >= thresholds
        train_accuracy = (train_predictions == labels[~test_mask, None]).mean(axis=0)
        threshold = float(thresholds[train_accuracy.argmax()])
        accuracy = float(
            ((similarities[test_mask] >= threshold) == labels[test_mask]).mean()
        )
        fold_thresholds.append(threshold)
        fold_accuracies.append(accuracy)

    return {
        "accuracy": float(np.mean(fold_accuracies)),
        "std": float(np.std(fold_accuracies)),
        "fold_accuracies": fold_accuracies,
        "fold_thresholds": fold_thresholds,
        "similarities": similarities,
        "labels": labels,
    }


def _self_check():
    keys, offsets = _parse_index_offsets(io.StringIO("1\t32\n0\t0\n"))
    assert keys.tolist() == [1, 0] and offsets.tolist() == [32, 0]
    assert np.sort(offsets).tolist() == [0, 32]
    payload = struct.pack("<IfQQ", 0, 7.0, 1, 0) + b"image"
    label, image = _unpack_mx_record(payload)
    assert label == 7 and image == b"image"
    assert not _is_image_record(b"not an image")
    head = ArcMarginProduct(embedding_dim=4, class_count=3)
    head.set_margin(0.25)
    logits, cosine = head(
        torch.randn(2, 4), torch.tensor([0, 2]), return_cosine=True
    )
    assert logits.shape == cosine.shape == (2, 3)
    assert torch.isfinite(logits).all() and torch.isfinite(cosine).all()


if __name__ == "__main__":
    _self_check()
    print("ArcFace self-check passed")
