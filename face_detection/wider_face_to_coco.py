"""Convert official WIDER FACE text annotations to COCO JSON."""

import argparse
import json
from pathlib import Path


def parse_annotations(text):
    """Yield (image path, boxes) from a WIDER FACE annotation file."""
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        image_path = lines[index].strip()
        index += 1
        if not image_path:
            continue
        if index >= len(lines):
            raise ValueError(f"Missing face count for {image_path}")

        face_count = int(lines[index])
        index += 1
        boxes = []
        for _ in range(face_count):
            if index >= len(lines):
                raise ValueError(f"Missing bounding box for {image_path}")
            values = [int(value) for value in lines[index].split()]
            index += 1
            if len(values) < 4:
                raise ValueError(f"Invalid bounding box for {image_path}: {values}")

            x, y, width, height = values[:4]
            if width <= 0 or height <= 0:
                continue
            invalid = values[7] if len(values) > 7 else 0
            boxes.append((x, y, width, height, invalid))
        yield image_path, boxes


def image_size(path):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required: pip install opencv-python") from exc

    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    height, width = image.shape[:2]
    return width, height


def convert(dataset_root, split):
    filename = f"wider_face_{split}_bbx_gt.txt"
    annotation_paths = (
        dataset_root / "wider_face_annotations" / "wider_face_split" / filename,
        dataset_root / "wider_face_split" / filename,
    )
    annotation_path = next((path for path in annotation_paths if path.exists()), None)
    if annotation_path is None:
        searched = "\n".join(str(path) for path in annotation_paths)
        raise FileNotFoundError(f"Cannot find {filename}. Searched:\n{searched}")

    image_root = dataset_root / f"WIDER_{split}" / "images"
    records = parse_annotations(annotation_path.read_text(encoding="utf-8"))

    images = []
    annotations = []
    annotation_id = 1
    for image_id, (relative_path, boxes) in enumerate(records, start=1):
        width, height = image_size(image_root / relative_path)
        images.append(
            {
                "id": image_id,
                "file_name": relative_path,
                "width": width,
                "height": height,
            }
        )
        for x, y, box_width, box_height, invalid in boxes:
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x, y, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": int(bool(invalid)),
                }
            )
            annotation_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "face"}],
    }


def self_test():
    sample = """0--Parade/example.jpg
2
10 20 30 40 0 0 0 0 0 0
1 2 3 4 0 0 0 1 0 0
1--Handshaking/example.jpg
0
"""
    records = list(parse_annotations(sample))
    assert records == [
        (
            "0--Parade/example.jpg",
            [(10, 20, 30, 40, 0), (1, 2, 3, 4, 1)],
        ),
        ("1--Handshaking/example.jpg", []),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", nargs="?", type=Path)
    parser.add_argument("split", nargs="?", choices=("train", "val"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("Self-test passed")
        return
    if args.dataset_root is None or args.split is None:
        parser.error("dataset_root and split are required")

    output_path = args.dataset_root / "annotations" / f"wider_face_{args.split}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = convert(args.dataset_root, args.split)
    output_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(
        f"Saved {len(result['images'])} images and "
        f"{len(result['annotations'])} boxes to {output_path}"
    )


if __name__ == "__main__":
    main()
