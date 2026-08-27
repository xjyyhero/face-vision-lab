"""Single-image 3D face reconstruction with MediaPipe Face Landmarker."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import cv2
import numpy as np


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


def download_model(path):
    path = Path(path)
    if path.exists() and path.stat().st_size:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".download")
    print(f"Downloading Face Landmarker model to {path} ...")
    urllib.request.urlretrieve(MODEL_URL, temporary)
    temporary.replace(path)
    return path


def _triangles_from_edges(edges):
    neighbors = {}
    for start, end in edges:
        neighbors.setdefault(start, set()).add(end)
        neighbors.setdefault(end, set()).add(start)
    triangles = {
        tuple(sorted((start, end, third)))
        for start, end in edges
        for third in neighbors[start] & neighbors[end]
    }
    return np.asarray(sorted(triangles), dtype=np.int32)


def reconstruct_face(image_path, model_path):
    """Return a coloured 468-vertex mesh reconstructed from one RGB image."""

    import mediapipe as mp

    image_path = Path(image_path)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_path), delegate=mp.tasks.BaseOptions.Delegate.CPU
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        output_facial_transformation_matrixes=True,
    )
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
        result = detector.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        )
    if not result.face_landmarks:
        raise ValueError(f"No face detected in {image_path}")

    connections = mp.tasks.vision.FaceLandmarksConnections
    edges = {
        tuple(sorted((edge.start, edge.end)))
        for edge in connections.FACE_LANDMARKS_TESSELATION
    }
    triangles = _triangles_from_edges(edges)
    vertex_count = int(triangles.max()) + 1
    landmarks = result.face_landmarks[0][:vertex_count]
    height, width = image_rgb.shape[:2]
    screen = np.asarray([[point.x, point.y] for point in landmarks])
    vertices = np.asarray(
        [[point.x - 0.5, 0.5 - point.y, -point.z] for point in landmarks],
        dtype=np.float32,
    )
    vertices -= np.median(vertices, axis=0)
    vertices /= np.ptp(vertices[:, :2], axis=0).max()
    pixels_x = np.clip((screen[:, 0] * width).astype(int), 0, width - 1)
    pixels_y = np.clip((screen[:, 1] * height).astype(int), 0, height - 1)
    colors = image_rgb[pixels_y, pixels_x].astype(np.float32) / 255.0
    transform = (
        np.asarray(result.facial_transformation_matrixes[0]).tolist()
        if result.facial_transformation_matrixes
        else None
    )
    return {
        "vertices": vertices,
        "triangles": triangles,
        "edges": np.asarray(sorted(edges), dtype=np.int32),
        "colors": colors,
        "screen": screen,
        "transform": transform,
        "image_bgr": image_bgr,
    }


def save_obj(path, vertices, triangles, colors):
    """Export geometry and per-vertex RGB values using the common OBJ extension."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write("# MediaPipe single-image face reconstruction\n")
        for vertex, color in zip(vertices, colors):
            values = (*vertex, *color)
            stream.write("v " + " ".join(f"{value:.6f}" for value in values) + "\n")
        for triangle in triangles + 1:
            stream.write(f"f {triangle[0]} {triangle[1]} {triangle[2]}\n")
    return path


def save_overlay(path, image_bgr, screen, edges):
    output = image_bgr.copy()
    height, width = output.shape[:2]
    points = np.column_stack((screen[:, 0] * width, screen[:, 1] * height)).astype(int)
    for start, end in edges:
        cv2.line(output, tuple(points[start]), tuple(points[end]), (50, 230, 100), 1)
    cv2.imwrite(str(path), output)
    return Path(path)


def _rotate_y(vertices, angle):
    radians = np.deg2rad(angle)
    rotation = np.asarray(
        [
            [np.cos(radians), 0, np.sin(radians)],
            [0, 1, 0],
            [-np.sin(radians), 0, np.cos(radians)],
        ]
    )
    return vertices @ rotation.T


def render_view(vertices, triangles, colors, angle=0, size=512):
    """Render one orthographic view using a small OpenCV software rasterizer."""

    rotated = _rotate_y(vertices, angle)
    scale = 0.82 * size / np.ptp(rotated[:, :2], axis=0).max()
    points = rotated[:, :2] * scale
    points[:, 0] += size / 2
    points[:, 1] = size / 2 - points[:, 1]
    canvas = np.full((size, size, 3), 245, dtype=np.uint8)
    depth_order = np.argsort(rotated[triangles, 2].mean(axis=1))
    for index in depth_order:
        triangle = triangles[index]
        surface = rotated[triangle]
        normal = np.cross(surface[1] - surface[0], surface[2] - surface[0])
        light = 0.45 + 0.55 * abs(normal[2]) / (np.linalg.norm(normal) + 1e-8)
        color = np.clip(colors[triangle].mean(axis=0) * light * 255, 0, 255)
        cv2.fillConvexPoly(
            canvas,
            np.round(points[triangle]).astype(np.int32),
            tuple(int(value) for value in color[::-1]),
        )
    return canvas


def save_multiview(path, vertices, triangles, colors):
    views = [render_view(vertices, triangles, colors, angle) for angle in (-35, 0, 35)]
    titles = ("Left 35 deg", "Front", "Right 35 deg")
    for image, title in zip(views, titles):
        cv2.putText(
            image, title, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2
        )
    output = np.concatenate(views, axis=1)
    cv2.imwrite(str(path), output)
    return Path(path)


def save_metadata(path, mesh, source_image):
    metadata = {
        "source_image": str(source_image),
        "vertex_count": len(mesh["vertices"]),
        "triangle_count": len(mesh["triangles"]),
        "facial_transformation_matrix": mesh["transform"],
        "coordinate_note": "Relative depth under a weak-perspective camera model",
    }
    Path(path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _self_check():
    edges = {(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)}
    assert len(_triangles_from_edges(edges)) == 4
    vertices = np.asarray([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], dtype=float)
    image = render_view(vertices, np.asarray([[0, 1, 2]]), np.ones((3, 3)))
    assert image.shape == (512, 512, 3) and image.dtype == np.uint8


if __name__ == "__main__":
    _self_check()
    print("3D face geometry self-check passed")
