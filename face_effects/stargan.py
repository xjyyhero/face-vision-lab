"""StarGAN v1 utilities for multi-attribute editing on CelebA."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import autograd, nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.transforms import v2
from tqdm import tqdm


HAIR_ATTRIBUTES = {"Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair"}


def celeba_transform(image_size=128, train=False):
    transforms = [v2.CenterCrop(178), v2.Resize((image_size, image_size))]
    if train:
        transforms.append(v2.RandomHorizontalFlip())
    transforms.extend(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.5,) * 3, (0.5,) * 3),
        ]
    )
    return v2.Compose(transforms)


class CelebAAttributes(Dataset):
    """Read selected CelebA attributes using the official data split."""

    PARTITIONS = {"train": "0", "valid": "1", "test": "2"}

    def __init__(
        self, root, attributes, partition="train", transform=None, max_samples=None
    ):
        self.root = Path(root)
        self.image_root = self.root / "img_align_celeba"
        self.attributes = list(attributes)
        self.transform = transform or celeba_transform(train=partition == "train")
        try:
            partition_id = self.PARTITIONS[partition]
        except KeyError as error:
            raise ValueError(f"partition must be one of {tuple(self.PARTITIONS)}") from error

        for path in (
            self.image_root,
            self.root / "list_attr_celeba.csv",
            self.root / "list_eval_partition.csv",
        ):
            if not path.exists():
                raise FileNotFoundError(path)

        with (self.root / "list_eval_partition.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            rows = csv.DictReader(stream)
            selected_images = {
                row["image_id"] for row in rows if row["partition"] == partition_id
            }

        self.samples = []
        with (self.root / "list_attr_celeba.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            rows = csv.DictReader(stream)
            missing = set(self.attributes) - set(rows.fieldnames or ())
            if missing:
                raise ValueError(f"Unknown CelebA attributes: {sorted(missing)}")
            for row in rows:
                if row["image_id"] not in selected_images:
                    continue
                labels = [float(int(row[name]) > 0) for name in self.attributes]
                self.samples.append((row["image_id"], labels))
                if max_samples and len(self.samples) >= max_samples:
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        filename, labels = self.samples[index]
        with Image.open(self.image_root / filename) as image:
            image = image.convert("RGB")
        return self.transform(image), torch.tensor(labels), filename


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, inputs):
        return inputs + self.layers(inputs)


class Generator(nn.Module):
    def __init__(self, attribute_count, base_channels=64, residual_blocks=6):
        super().__init__()
        layers = [
            nn.Conv2d(3 + attribute_count, base_channels, 7, 1, 3, bias=False),
            nn.InstanceNorm2d(base_channels, affine=True),
            nn.ReLU(inplace=True),
        ]
        channels = base_channels
        for _ in range(2):
            layers.extend(
                [
                    nn.Conv2d(channels, channels * 2, 4, 2, 1, bias=False),
                    nn.InstanceNorm2d(channels * 2, affine=True),
                    nn.ReLU(inplace=True),
                ]
            )
            channels *= 2
        layers.extend(ResidualBlock(channels) for _ in range(residual_blocks))
        for _ in range(2):
            layers.extend(
                [
                    nn.ConvTranspose2d(channels, channels // 2, 4, 2, 1, bias=False),
                    nn.InstanceNorm2d(channels // 2, affine=True),
                    nn.ReLU(inplace=True),
                ]
            )
            channels //= 2
        layers.extend([nn.Conv2d(channels, 3, 7, 1, 3), nn.Tanh()])
        self.layers = nn.Sequential(*layers)

    def forward(self, images, labels):
        labels = labels[:, :, None, None].expand(-1, -1, *images.shape[2:])
        return self.layers(torch.cat((images, labels), dim=1))


class Discriminator(nn.Module):
    def __init__(
        self, image_size, attribute_count, base_channels=64, repeat_num=6
    ):
        super().__init__()
        layers = [nn.Conv2d(3, base_channels, 4, 2, 1), nn.LeakyReLU(0.01)]
        channels = base_channels
        for _ in range(1, repeat_num):
            layers.extend(
                [
                    nn.Conv2d(channels, channels * 2, 4, 2, 1),
                    nn.LeakyReLU(0.01),
                ]
            )
            channels *= 2
        feature_size = image_size // (2**repeat_num)
        if feature_size < 1:
            raise ValueError("image_size is too small for discriminator repeat_num")
        self.features = nn.Sequential(*layers)
        self.source_head = nn.Conv2d(channels, 1, 3, 1, 1, bias=False)
        self.attribute_head = nn.Conv2d(
            channels, attribute_count, feature_size, bias=False
        )

    def forward(self, images):
        features = self.features(images)
        source = self.source_head(features)
        attributes = self.attribute_head(features).flatten(1)
        return source, attributes


def initialize_weights(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def sample_target_labels(labels):
    """Shuffle real label vectors so sampled targets keep valid co-occurrences."""

    if len(labels) == 1:
        return 1.0 - labels
    shift = int(torch.randint(1, len(labels), ()).item())
    return torch.roll(labels, shift, dims=0)


def make_attribute_targets(labels, attributes):
    """Return one target per attribute, with mutually exclusive hair colours."""

    targets = []
    hair_indices = [i for i, name in enumerate(attributes) if name in HAIR_ATTRIBUTES]
    for index, name in enumerate(attributes):
        target = labels.clone()
        if name in HAIR_ATTRIBUTES:
            target[:, hair_indices] = 0
            target[:, index] = 1
        else:
            target[:, index] = 1 - target[:, index]
        targets.append(target)
    return targets


def gradient_penalty(discriminator, real, fake):
    alpha = torch.rand(len(real), 1, 1, 1, device=real.device)
    mixed = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    score, _ = discriminator(mixed)
    gradients = autograd.grad(score.sum(), mixed, create_graph=True)[0]
    return ((gradients.flatten(1).norm(2, dim=1) - 1) ** 2).mean()


def train_epoch(
    generator,
    discriminator,
    loader,
    generator_optimizer,
    discriminator_optimizer,
    device,
    *,
    lambda_cls=1.0,
    lambda_rec=10.0,
    lambda_gp=10.0,
    n_critic=5,
    start_step=0,
    max_steps=None,
    deadline=None,
    log_interval=100,
    epoch=None,
    show_progress=False,
):
    generator.train()
    discriminator.train()
    totals = {"d": 0.0, "g": 0.0, "batches": 0, "g_updates": 0}

    batches = tqdm(
        loader,
        desc=f"Epoch {epoch}" if epoch is not None else "Training",
        unit="batch",
        leave=True,
        disable=not show_progress,
    )
    for batch_index, (real, real_labels, _) in enumerate(batches):
        step = start_step + batch_index
        if (max_steps is not None and step >= max_steps) or (
            deadline is not None and time.monotonic() >= deadline
        ):
            break
        real, real_labels = real.to(device), real_labels.to(device)
        target_labels = sample_target_labels(real_labels)

        with torch.no_grad():
            fake = generator(real, target_labels)
        real_score, real_prediction = discriminator(real)
        fake_score, _ = discriminator(fake)
        d_loss = (
            -real_score.mean()
            + fake_score.mean()
            + lambda_cls
            * F.binary_cross_entropy_with_logits(real_prediction, real_labels)
            + lambda_gp * gradient_penalty(discriminator, real, fake)
        )
        discriminator_optimizer.zero_grad(set_to_none=True)
        d_loss.backward()
        discriminator_optimizer.step()
        totals["d"] += d_loss.item()

        if (step + 1) % n_critic == 0:
            discriminator.requires_grad_(False)
            fake = generator(real, target_labels)
            fake_score, fake_prediction = discriminator(fake)
            reconstructed = generator(fake, real_labels)
            g_loss = (
                -fake_score.mean()
                + lambda_cls
                * F.binary_cross_entropy_with_logits(fake_prediction, target_labels)
                + lambda_rec * F.l1_loss(reconstructed, real)
            )
            generator_optimizer.zero_grad(set_to_none=True)
            g_loss.backward()
            generator_optimizer.step()
            discriminator.requires_grad_(True)
            totals["g"] += g_loss.item()
            totals["g_updates"] += 1

        totals["batches"] += 1
        if show_progress:
            batches.set_postfix(
                step=step + 1,
                D=f"{totals['d'] / totals['batches']:.4f}",
                G=f"{totals['g'] / max(totals['g_updates'], 1):.4f}",
            )
        if log_interval and (step + 1) % log_interval == 0:
            print(
                f"step {step + 1}: D={totals['d'] / totals['batches']:.4f}, "
                f"G={totals['g'] / max(totals['g_updates'], 1):.4f}"
            )

    batches.close()
    return {
        "d_loss": totals["d"] / max(totals["batches"], 1),
        "g_loss": totals["g"] / max(totals["g_updates"], 1),
        "steps": totals["batches"],
    }


class InceptionMetrics(nn.Module):
    """Return ImageNet logits and pool features for IS and FID."""

    def __init__(self):
        super().__init__()
        weights = Inception_V3_Weights.DEFAULT
        self.model = inception_v3(weights=weights).eval()
        self.preprocess = weights.transforms()
        self.features = None
        self.model.avgpool.register_forward_hook(self._capture_features)

    def _capture_features(self, _module, _inputs, output):
        self.features = output.flatten(1)

    def forward(self, images):
        logits = self.model(self.preprocess((images + 1) / 2))
        return logits, self.features


def frechet_distance(real_features, fake_features):
    """Compute FID without adding a SciPy dependency."""

    real_features = np.asarray(real_features, dtype=np.float64)
    fake_features = np.asarray(fake_features, dtype=np.float64)
    real_mean, fake_mean = real_features.mean(0), fake_features.mean(0)
    real_cov = np.cov(real_features, rowvar=False)
    fake_cov = np.cov(fake_features, rowvar=False)
    values, vectors = np.linalg.eigh(real_cov)
    real_sqrt = (vectors * np.sqrt(np.clip(values, 0, None))) @ vectors.T
    middle = real_sqrt @ fake_cov @ real_sqrt
    covariance_trace = np.sqrt(np.clip(np.linalg.eigvalsh(middle), 0, None)).sum()
    return float(
        np.square(real_mean - fake_mean).sum()
        + np.trace(real_cov)
        + np.trace(fake_cov)
        - 2 * covariance_trace
    )


def inception_score(probabilities, splits=10):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    scores = []
    for part in np.array_split(probabilities, min(splits, len(probabilities))):
        marginal = part.mean(0, keepdims=True)
        kl = part * (np.log(part + 1e-12) - np.log(marginal + 1e-12))
        scores.append(np.exp(kl.sum(1).mean()))
    return float(np.mean(scores)), float(np.std(scores))


@torch.inference_mode()
def evaluate_quality(generator, loader, device, max_images=5000):
    generator.eval()
    inception = InceptionMetrics().to(device)
    real_features, fake_features, fake_probabilities = [], [], []
    count = 0
    for real, labels, _ in loader:
        real, labels = real.to(device), labels.to(device)
        fake = generator(real, sample_target_labels(labels))
        _, real_feature = inception(real)
        fake_logits, fake_feature = inception(fake)
        remaining = max_images - count
        real_features.append(real_feature[:remaining].cpu().numpy())
        fake_features.append(fake_feature[:remaining].cpu().numpy())
        fake_probabilities.append(fake_logits[:remaining].softmax(1).cpu().numpy())
        count += min(len(real), remaining)
        if count >= max_images:
            break
    if count < 2:
        raise ValueError("At least two images are required for FID/IS")
    fid = frechet_distance(np.concatenate(real_features), np.concatenate(fake_features))
    score_mean, score_std = inception_score(np.concatenate(fake_probabilities))
    return {"images": count, "fid": fid, "is_mean": score_mean, "is_std": score_std}


def _self_check():
    attributes = ["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"]
    labels = torch.tensor([[1.0, 0, 0, 0, 1], [0, 1, 0, 1, 0]])
    targets = make_attribute_targets(labels, attributes)
    assert all(target[:, :3].sum(1).eq(1).all() for target in targets[:3])
    generator = Generator(len(attributes), base_channels=8, residual_blocks=1)
    discriminator = Discriminator(32, len(attributes), base_channels=8, repeat_num=3)
    images = torch.randn(2, 3, 32, 32)
    fake = generator(images, labels)
    source, prediction = discriminator(fake)
    assert fake.shape == images.shape and source.ndim == 4
    assert prediction.shape == labels.shape


if __name__ == "__main__":
    _self_check()
    print("StarGAN self-check passed")
