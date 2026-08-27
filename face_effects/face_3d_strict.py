"""Strict task-8 pipeline: official 3DDFA_V2 reconstruction + OpenGL rendering."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


THREEDDFA_URL = "https://github.com/cleardusk/3DDFA_V2.git"
THREEDDFA_COMMIT = "1b6c67601abffc1e9f248b291708aef0e43b55ae"


def _run(command, cwd=None):
    subprocess.run(command, cwd=cwd, check=True)


def prepare_3ddfa(repo_dir):
    """Fetch a pinned official 3DDFA_V2 revision and apply a Windows-safe NMS shim."""

    repo_dir = Path(repo_dir).resolve()
    if not (repo_dir / ".git").exists():
        if repo_dir.exists() and any(repo_dir.iterdir()):
            raise FileExistsError(f"{repo_dir} exists but is not a 3DDFA_V2 git checkout")
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        repo_dir.mkdir(exist_ok=True)
        _run(["git", "init"], cwd=repo_dir)
        _run(["git", "remote", "add", "origin", THREEDDFA_URL], cwd=repo_dir)

    try:
        current = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        _run(["git", "fetch", "--depth", "1", "origin", THREEDDFA_COMMIT], cwd=repo_dir)
        _run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=repo_dir)
        current = THREEDDFA_COMMIT
    if current != THREEDDFA_COMMIT:
        raise RuntimeError(
            f"Expected 3DDFA_V2 {THREEDDFA_COMMIT}, found {current}. "
            "Use a new third_party/3DDFA_V2 directory."
        )

    # The official ONNX detector still imports its compiled Cython NMS extension.
    # Use the pure-Python NMS that is already shipped in the same official repo.
    wrapper = repo_dir / "FaceBoxes/utils/nms_wrapper.py"
    source = wrapper.read_text(encoding="utf-8")
    source = source.replace(
        "from .nms.cpu_nms import cpu_nms, cpu_soft_nms",
        "from .nms.py_cpu_nms import py_cpu_nms",
    ).replace("return cpu_nms(dets, thresh)", "return py_cpu_nms(dets, thresh)")
    wrapper.write_text(source, encoding="utf-8")

    bfm_model = repo_dir / "bfm/bfm.py"
    source = bfm_model.read_text(encoding="utf-8")
    bfm_model.write_text(
        source.replace("astype(np.long)", "astype(np.int64)"), encoding="utf-8"
    )
    return repo_dir


def _load_official_models(repo_dir):
    import yaml

    repo_dir = Path(repo_dir).resolve()
    repo_string = str(repo_dir)
    if repo_string not in sys.path:
        sys.path.insert(0, repo_string)

    from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
    from TDDFA_ONNX import TDDFA_ONNX

    config_path = repo_dir / "configs/mb1_120x120.yml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["checkpoint_fp"] = str(repo_dir / config["checkpoint_fp"])
    config["bfm_fp"] = str(repo_dir / config["bfm_fp"])
    return FaceBoxes_ONNX(), TDDFA_ONNX(**config)


def _sample_vertex_colors(image_bgr, vertices_screen):
    height, width = image_bgr.shape[:2]
    x = np.clip(np.rint(vertices_screen[0]), 0, width - 1).astype(np.int32)
    y = np.clip(np.rint(vertices_screen[1]), 0, height - 1).astype(np.int32)
    return image_bgr[y, x, ::-1].astype(np.float32) / 255.0


def _normalise_for_opengl(vertices_screen, image_height):
    vertices = vertices_screen.T.astype(np.float32).copy()
    vertices[:, 1] = image_height - vertices[:, 1]
    vertices -= np.median(vertices, axis=0)
    scale = float(np.ptp(vertices[:, :2], axis=0).max())
    if scale <= 0:
        raise ValueError("Degenerate reconstructed mesh")
    vertices /= scale
    return vertices


def reconstruct_with_3ddfa(image_path, repo_dir, output_dir):
    """Run the official 3DDFA_V2 ONNX pipeline and export a coloured OBJ mesh."""

    image_path = Path(image_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(image_path)

    face_boxes, tddfa = _load_official_models(repo_dir)
    boxes = face_boxes(image_bgr)
    if not boxes:
        raise ValueError(f"No face detected in {image_path}")
    box = max(boxes, key=lambda item: (item[2] - item[0]) * (item[3] - item[1]))
    parameters, roi_boxes = tddfa(image_bgr, [box])
    vertices_screen = tddfa.recon_vers(
        parameters, roi_boxes, dense_flag=True
    )[0].astype(np.float32)
    triangles = np.asarray(tddfa.tri, dtype=np.int32)
    colors = _sample_vertex_colors(image_bgr, vertices_screen)

    from utils.serialization import ser_to_obj

    obj_path = output_dir / "3ddfa_reconstructed_face.obj"
    ser_to_obj(
        image_bgr,
        [vertices_screen.copy()],
        triangles,
        height=image_bgr.shape[0],
        wfp=str(obj_path),
    )
    return {
        "vertices": _normalise_for_opengl(vertices_screen, image_bgr.shape[0]),
        "vertices_screen": vertices_screen,
        "triangles": triangles[:, ::-1].copy(),
        "colors": colors,
        "image_bgr": image_bgr,
        "face_box": [float(value) for value in box],
        "obj_path": obj_path,
    }


def save_dense_overlay(path, mesh, stride=12):
    """Save a visual check that dense 3DDFA vertices align with the input face."""

    output = mesh["image_bgr"].copy()
    points = np.rint(mesh["vertices_screen"][:2, ::stride].T).astype(np.int32)
    for x, y in points:
        cv2.circle(output, (int(x), int(y)), 1, (40, 240, 90), -1, cv2.LINE_AA)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), output)
    return path


def _rotation_y(angle_degrees):
    radians = np.deg2rad(angle_degrees)
    cosine, sine = np.cos(radians), np.sin(radians)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = (
        (cosine, 0, sine),
        (0, 1, 0),
        (-sine, 0, cosine),
    )
    return transform


def render_opengl_multiview(path, mesh, angles=(-35, 0, 35), size=512):
    """Render the same reconstructed mesh from three angles with pyrender/OpenGL."""

    try:
        import pyrender
        import trimesh
    except ImportError as error:
        raise ImportError(
            "Install the strict renderer first: pip install pyrender==0.1.45 "
            "'pyglet<2' trimesh PyOpenGL"
        ) from error

    vertex_colors = np.column_stack(
        (np.clip(mesh["colors"] * 255, 0, 255).astype(np.uint8),
         np.full(len(mesh["colors"]), 255, dtype=np.uint8))
    )
    geometry = trimesh.Trimesh(
        vertices=mesh["vertices"],
        faces=mesh["triangles"],
        vertex_colors=vertex_colors,
        process=False,
    )
    camera_pose = np.eye(4, dtype=np.float32)
    camera_pose[2, 3] = 2.2
    camera = pyrender.OrthographicCamera(xmag=0.65, ymag=0.65)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.5)
    renderer = pyrender.OffscreenRenderer(size, size)
    images = []
    try:
        for angle in angles:
            scene = pyrender.Scene(
                bg_color=(245, 245, 245, 255), ambient_light=(0.35, 0.35, 0.35)
            )
            rotated = geometry.copy()
            rotated.apply_transform(_rotation_y(angle))
            scene.add(pyrender.Mesh.from_trimesh(rotated, smooth=True))
            scene.add(camera, pose=camera_pose)
            scene.add(light, pose=camera_pose)
            color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            image_bgr = cv2.cvtColor(color, cv2.COLOR_RGBA2BGR)
            label = "Front" if angle == 0 else f"{abs(angle)} deg {'left' if angle < 0 else 'right'}"
            cv2.putText(
                image_bgr, label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (35, 35, 35), 2, cv2.LINE_AA,
            )
            images.append(image_bgr)
    finally:
        renderer.delete()

    output = np.concatenate(images, axis=1)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), output)
    return path


def save_strict_metadata(path, mesh, source_image, metrics=None, dependencies=None):
    metadata = {
        "task": "8.2-8.3 strict",
        "reconstruction": "official 3DDFA_V2 ONNX (3DMM parameter regression)",
        "official_commit": THREEDDFA_COMMIT,
        "rendering": "pyrender OpenGL offscreen renderer",
        "source_image": str(Path(source_image).resolve()),
        "vertex_count": int(len(mesh["vertices"])),
        "triangle_count": int(len(mesh["triangles"])),
        "face_box_xyxy_score": mesh["face_box"],
        "obj_path": str(mesh["obj_path"]),
        "metrics_seconds": metrics or {},
        "dependencies": dependencies or {},
    }
    path = Path(path)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def save_strict_report(path, metadata):
    """Write a ready-to-submit experiment record from the measured run."""

    metrics = metadata.get("metrics_seconds", {})
    dependencies = metadata.get("dependencies", {})
    report = f"""# 任务 8：3D 人脸重建实验记录（严格版）

## 实验目的

从单张人脸图像回归 3DMM 参数，恢复稠密三维人脸网格，导出 OBJ 模型，并使用 OpenGL 生成左侧、正面和右侧三视角结果。

## 方法与环境

- 重建算法：官方 3DDFA_V2 ONNX，固定版本 `{metadata['official_commit']}`
- 人脸检测：官方 FaceBoxes ONNX
- 三维表示：3D Morphable Model（身份、表情、姿态与投影参数）
- 渲染方式：pyrender 的 OpenGL 离屏渲染
- 输入图像：`{metadata['source_image']}`
- Python：{dependencies.get('python', '见运行环境')}
- PyTorch：{dependencies.get('torch', '见运行环境')}
- OpenCV：{dependencies.get('opencv', '见运行环境')}
- ONNX Runtime：{dependencies.get('onnxruntime', '见运行环境')}

## 实验过程

1. 使用 FaceBoxes 检测输入图像中的人脸区域。
2. 使用 3DDFA_V2 回归 62 维 3DMM/姿态参数并解码稠密网格。
3. 从原图采样顶点颜色，按官方三角拓扑导出 OBJ。
4. 将网格归一化后绕 Y 轴旋转 -35°、0°、35°，使用 OpenGL 离屏渲染三视图。

## 实验结果

- 顶点数：{metadata['vertex_count']}
- 三角面数：{metadata['triangle_count']}
- 3DDFA_V2 重建耗时：{metrics.get('reconstruction', '未记录')} 秒
- OpenGL 三视图渲染耗时：{metrics.get('rendering', '未记录')} 秒
- 3D 模型：`3ddfa_reconstructed_face.obj`
- 稠密点对齐图：`3ddfa_dense_overlay.png`
- 多角度渲染图：`3ddfa_opengl_multiview.png`

## 结果分析与结论

实验成功从单张图像恢复了具有稠密顶点和固定三角拓扑的三维人脸表面。稠密点对齐图用于确认模型与眼、鼻、口及轮廓区域基本贴合；OpenGL 三视图证明输出是可旋转、可渲染的三维网格，而不是二维图像变换。任务 8.2 的 3DDFA_V2 单图重建、任务 8.3 的 OpenGL 渲染和多角度可视化均已完成。

单图重建仍存在固有限制：被遮挡区域和后脑没有直接观测，纹理来自输入视角采样，侧视图可能出现拉伸；结果适合算法演示与可视化，不等同于高精度三维扫描。
"""
    path = Path(path)
    path.write_text(report, encoding="utf-8")
    return path


def dependency_report():
    """Return version evidence for the experiment record."""

    import onnxruntime
    import pyrender
    import torch
    import trimesh

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "onnxruntime": onnxruntime.__version__,
        "pyrender": getattr(pyrender, "__version__", "0.1.45"),
        "trimesh": trimesh.__version__,
    }
