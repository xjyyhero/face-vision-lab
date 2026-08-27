"""Real-time face stickers, beauty effects, video recording, and FPS metrics."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from face_effects.face_3d import download_model


FACE_OVAL = np.asarray(
    [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
        379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
        234, 127, 162, 21, 54, 103, 67, 109,
    ]
)
OUTER_LIPS = np.asarray(
    [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267,
     0, 37, 39, 40, 185]
)
INNER_LIPS = np.asarray(
    [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312,
     13, 82, 81, 80, 191]
)
LEFT_EYE = np.asarray([33, 133, 159, 145])
RIGHT_EYE = np.asarray([362, 263, 386, 374])


def _landmark_points(landmarks, width, height):
    return np.asarray(
        [[point.x * width, point.y * height] for point in landmarks],
        dtype=np.int32,
    )


def _face_mask(points, shape):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(points[FACE_OVAL]), 255)
    return cv2.GaussianBlur(mask, (31, 31), 0)


def _beautify_cpu(frame, mask):
    smooth = cv2.bilateralFilter(frame, 9, 60, 60)
    beauty = cv2.addWeighted(frame, 0.35, smooth, 0.65, 10)
    alpha = (mask.astype(np.float32) / 255.0 * 0.8)[:, :, None]
    return np.clip(frame * (1 - alpha) + beauty * alpha, 0, 255).astype(np.uint8)


def _beautify_cuda(frame, mask):
    import torch
    from torch.nn import functional as F

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA effects requested, but torch.cuda.is_available() is False")
    image = torch.from_numpy(frame).cuda().permute(2, 0, 1)[None].float() / 255.0
    alpha = torch.from_numpy(mask).cuda()[None, None].float() / 255.0 * 0.8
    smooth = F.avg_pool2d(image, 7, stride=1, padding=3)
    output = (image * (1 - alpha) + smooth * alpha + 0.04 * alpha).clamp(0, 1)
    return (output[0].permute(1, 2, 0) * 255).byte().cpu().numpy()


def _apply_lipstick(frame, points):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points[OUTER_LIPS]], 255)
    cv2.fillPoly(mask, [points[INNER_LIPS]], 0)
    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    lipstick = np.empty_like(frame)
    lipstick[:] = (55, 35, 190)
    alpha = (mask.astype(np.float32) / 255.0 * 0.48)[:, :, None]
    return np.clip(frame * (1 - alpha) + lipstick * alpha, 0, 255).astype(np.uint8)


def _draw_stickers(frame, points):
    layer = frame.copy()
    left = points[LEFT_EYE].mean(axis=0).astype(int)
    right = points[RIGHT_EYE].mean(axis=0).astype(int)
    eye_distance = max(int(np.linalg.norm(right - left)), 20)
    radius_x, radius_y = int(eye_distance * 0.32), int(eye_distance * 0.20)

    for center in (left, right):
        cv2.ellipse(layer, tuple(center), (radius_x, radius_y), 0, 0, 360,
                    (210, 190, 120), -1, cv2.LINE_AA)
        cv2.ellipse(layer, tuple(center), (radius_x, radius_y), 0, 0, 360,
                    (25, 25, 25), max(2, eye_distance // 28), cv2.LINE_AA)
    cv2.line(layer, tuple(left + [radius_x, 0]), tuple(right - [radius_x, 0]),
             (25, 25, 25), max(2, eye_distance // 30), cv2.LINE_AA)
    cv2.line(layer, tuple(left - [radius_x, 0]), tuple(points[234]),
             (25, 25, 25), max(2, eye_distance // 35), cv2.LINE_AA)
    cv2.line(layer, tuple(right + [radius_x, 0]), tuple(points[454]),
             (25, 25, 25), max(2, eye_distance // 35), cv2.LINE_AA)

    face_width = max(int(np.linalg.norm(points[454] - points[234])), 40)
    top = points[10]
    brim_y = int(top[1] - face_width * 0.06)
    brim_left = (int(top[0] - face_width * 0.52), brim_y)
    brim_right = (int(top[0] + face_width * 0.52), brim_y)
    crown = np.asarray(
        [
            [top[0] - face_width * 0.34, brim_y],
            [top[0] - face_width * 0.25, brim_y - face_width * 0.36],
            [top[0] + face_width * 0.25, brim_y - face_width * 0.36],
            [top[0] + face_width * 0.34, brim_y],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(layer, crown, (45, 70, 210), cv2.LINE_AA)
    cv2.line(layer, brim_left, brim_right, (25, 25, 25),
             max(5, face_width // 20), cv2.LINE_AA)
    cv2.line(layer, tuple(crown[0]), tuple(crown[3]), (30, 40, 120),
             max(3, face_width // 35), cv2.LINE_AA)
    return cv2.addWeighted(layer, 0.82, frame, 0.18, 0)


def apply_effects(frame, landmarks, device="cpu"):
    height, width = frame.shape[:2]
    points = _landmark_points(landmarks, width, height)
    mask = _face_mask(points, frame.shape)
    output = _beautify_cuda(frame, mask) if device == "cuda" else _beautify_cpu(frame, mask)
    output = _apply_lipstick(output, points)
    return _draw_stickers(output, points)


def create_sample_video(image_path, output_path, seconds=5, fps=20, size=512):
    """Create a short moving-face input so the demo is reproducible without a webcam."""

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    scale = max(size / image.shape[1], size / image.shape[0])
    resized = cv2.resize(image, None, fx=scale, fy=scale)
    y = (resized.shape[0] - size) // 2
    x = (resized.shape[1] - size) // 2
    base = resized[y : y + size, x : x + size]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size)
    )
    frame_count = int(seconds * fps)
    for index in range(frame_count):
        phase = 2 * np.pi * index / frame_count
        matrix = cv2.getRotationMatrix2D(
            (size / 2, size / 2), 2.5 * np.sin(phase), 1 + 0.025 * np.cos(phase)
        )
        matrix[:, 2] += (8 * np.sin(phase), 5 * np.cos(phase))
        writer.write(cv2.warpAffine(base, matrix, (size, size), borderMode=cv2.BORDER_REFLECT))
    writer.release()
    return output_path


def make_browser_playable(path):
    """Replace an OpenCV MP4V file with a browser-compatible H.264 MP4."""

    try:
        from imageio_ffmpeg import get_ffmpeg_exe
    except ImportError as error:
        raise ImportError("Install the video encoder: pip install imageio-ffmpeg") from error

    path = Path(path)
    temporary = path.with_name(f"{path.stem}.h264{path.suffix}")
    subprocess.run(
        [
            get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(path),
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(temporary),
        ],
        check=True,
    )
    temporary.replace(path)
    return path


def process_video(input_source, output_path, model_path, device="cpu", max_frames=None):
    """Apply effects to a camera/video stream and save video plus FPS metrics."""

    import mediapipe as mp

    source = int(input_source) if str(input_source).isdigit() else str(input_source)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise ValueError(f"Cannot open video source: {input_source}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError(f"Invalid video dimensions: {width}x{height}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot create output video: {output_path}")
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_path), delegate=mp.tasks.BaseOptions.Delegate.CPU
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
    )
    frames = face_frames = 0
    landmark_seconds = effect_seconds = total_seconds = 0.0
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
        while max_frames is None or frames < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            started = time.perf_counter()
            landmark_started = time.perf_counter()
            result = detector.detect_for_video(
                mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                ),
                int(frames * 1000 / fps),
            )
            landmark_seconds += time.perf_counter() - landmark_started
            if result.face_landmarks:
                effect_started = time.perf_counter()
                frame = apply_effects(frame, result.face_landmarks[0], device)
                effect_seconds += time.perf_counter() - effect_started
                face_frames += 1
            frames += 1
            total_seconds += time.perf_counter() - started
            current_fps = frames / max(total_seconds, 1e-9)
            cv2.putText(frame, f"FPS {current_fps:.1f} | {device.upper()}", (14, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 240, 80), 2)
            writer.write(frame)
    capture.release()
    writer.release()
    if not frames:
        raise ValueError("Video source contained no readable frames")
    make_browser_playable(output_path)
    metrics = {
        "input": str(input_source),
        "output": str(output_path),
        "frames": frames,
        "face_frames": face_frames,
        "face_detection_rate": face_frames / frames,
        "resolution": [width, height],
        "source_fps": fps,
        "measured_fps": frames / total_seconds,
        "landmark_ms_per_frame": 1000 * landmark_seconds / frames,
        "effect_ms_per_frame": 1000 * effect_seconds / max(face_frames, 1),
        "landmark_device": "CPU",
        "effect_device": device.upper(),
        "output_codec": "H.264/yuv420p",
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def _self_check():
    frame = np.full((256, 256, 3), 150, dtype=np.uint8)
    points = np.tile([128, 128], (468, 1)).astype(np.int32)
    angles = np.linspace(0, 2 * np.pi, len(FACE_OVAL), endpoint=False)
    points[FACE_OVAL] = np.column_stack((128 + 80 * np.cos(angles), 128 + 100 * np.sin(angles)))
    points[LEFT_EYE] = [[75, 105], [105, 105], [90, 98], [90, 112]]
    points[RIGHT_EYE] = [[151, 105], [181, 105], [166, 98], [166, 112]]
    points[OUTER_LIPS] = np.column_stack((128 + 35 * np.cos(angles[:len(OUTER_LIPS)]),
                                          165 + 12 * np.sin(angles[:len(OUTER_LIPS)])))
    points[INNER_LIPS] = np.column_stack((128 + 20 * np.cos(angles[:len(INNER_LIPS)]),
                                          165 + 5 * np.sin(angles[:len(INNER_LIPS)])))
    points[10], points[234], points[454] = (128, 30), (48, 128), (208, 128)
    mask = _face_mask(points, frame.shape)
    output = _draw_stickers(_apply_lipstick(_beautify_cpu(frame, mask), points), points)
    assert output.shape == frame.shape and output.dtype == np.uint8 and mask.max() == 255


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="0", help="Video path or camera index")
    parser.add_argument("--output", default="work_dirs/realtime_effects/demo.mp4")
    parser.add_argument("--model", default="data/models/face_landmarker.task")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        print("Real-time effects self-check passed")
    else:
        download_model(args.model)
        print(process_video(args.input, args.output, args.model, args.device, args.max_frames))
