# Face Vision Lab

智能视觉 AI 实习项目，围绕人脸检测、识别、关键点定位与视觉特效展开。

## 第三周任务

| 任务 | 内容 | 对应文件 |
|---|---|---|
| 5.1–5.3 | MS-Celeb-1M 上训练 ResNet50 + ArcFace，并做 LFW 10 折验证 | [`face_recognition/07_arcface_training.ipynb`](face_recognition/07_arcface_training.ipynb)、[`face_recognition/arcface.py`](face_recognition/arcface.py)、[`face_recognition/arcface_config.json`](face_recognition/arcface_config.json) |
| 6.1–6.3 | 动态量化、性能对比与 ONNX 推理 | [`face_recognition/08_model_optimization.ipynb`](face_recognition/08_model_optimization.ipynb) |

## 第二周任务

| 任务 | 内容 | 对应文件 |
|---|---|---|
| 3.1–3.3 | WIDER FACE 检测训练、评估与展示 | [`face_detection/05_wider_face_detection_training.ipynb`](face_detection/05_wider_face_detection_training.ipynb) |
| 3.2 | MMDetection 配置与数据转换 | [`face_detection/retinanet_r50_fpn_wider_face.py`](face_detection/retinanet_r50_fpn_wider_face.py)、[`face_detection/wider_face_to_coco.py`](face_detection/wider_face_to_coco.py) |
| 3.3 | 实验结果报告 | [`face_detection/experiment_report.md`](face_detection/experiment_report.md) |
| 4.1–4.3 | 300W 关键点训练、NME 与人脸对齐 | [`face_landmarks/06_300w_hrnet_landmarks.ipynb`](face_landmarks/06_300w_hrnet_landmarks.ipynb) |

## 第一周任务

| 任务 | 内容 | 对应文件 |
|---|---|---|
| 1.1–1.4 | Python、Git、Docker、Jupyter 与 OpenCV 基础 | [`docs/environment_setup.md`](docs/environment_setup.md)、[`test_env.ipynb`](test_env.ipynb)、[`openCV_sample/image_processing.ipynb`](openCV_sample/image_processing.ipynb) |
| 2.1 | 人脸检测、对齐、识别和验证基础 | [`face_recognition/01_face_recognition_basics.ipynb`](face_recognition/01_face_recognition_basics.ipynb) |
| 2.2 | CelebA 与 LFW 数据集探索和可视化 | [`face_recognition/02_dataset_exploration.ipynb`](face_recognition/02_dataset_exploration.ipynb)、[`face_recognition/dataset_analysis.md`](face_recognition/dataset_analysis.md) |
| 2.3 | 使用 MMDetection Grounding DINO 检测人脸 | [`face_recognition/03_face_detection.ipynb`](face_recognition/03_face_detection.ipynb) |
| 2.4 | YuNet 5 点定位与 SFace LFW 人脸验证 | [`face_recognition/04_face_landmarks.ipynb`](face_recognition/04_face_landmarks.ipynb) |

## 项目结构

```text
face-vision-lab/
├── docs/                         # 环境文档与任务说明
├── face_detection/               # 第二周检测训练、评估与报告
├── face_landmarks/               # 第二周关键点训练、NME 与对齐
├── face_recognition/
│   ├── 01_face_recognition_basics.ipynb
│   ├── 02_dataset_exploration.ipynb
│   ├── 03_face_detection.ipynb
│   ├── 04_face_landmarks.ipynb
│   ├── 07_arcface_training.ipynb
│   ├── 08_model_optimization.ipynb
│   ├── arcface.py
│   ├── arcface_config.json
│   └── dataset_analysis.md
├── openCV_sample/                # OpenCV 图像处理示例
├── Dockerfile
├── requirements.txt
└── test_env.ipynb
```

`data/` 及其中自动下载的模型权重不作为源码提交。Notebook 的运行结果可以直接保存在 `.ipynb` 中展示。

## 运行环境

基础环境配置见 [`docs/environment_setup.md`](docs/environment_setup.md)。人脸检测 Notebook 已在以下 Windows GPU 环境中验证：

| 软件 | 版本 |
|---|---|
| Python | 3.11 |
| PyTorch | 2.1.0 + CUDA 11.8 |
| MMCV | 2.1.0 |
| MMEngine | 0.10.7 |
| MMDetection | 3.3.0 |
| OpenCV | 4.10.0 |
| GPU | NVIDIA GeForce RTX 3080 |

在项目目录中选择 `.venv` 的 Jupyter 内核，然后启动 Notebook：

```bash
jupyter notebook
```

前两周建议按 `01` 到 `06` 的顺序运行；第三周依次运行 `07` 和 `08`。

## 数据集目录

数据集体积较大，已通过 `.gitignore` 排除。请在本地保持以下结构：

```text
data/
├── CelebA/
│   ├── img_align_celeba/
│   ├── list_attr_celeba.csv
│   ├── identity_CelebA.txt
│   └── list_eval_partition.csv
├── MS-Celeb-1M/
│   ├── train.rec
│   ├── train.idx
│   ├── property
│   └── *.bin
├── lfw/
│   ├── lfw-deepfunneled/
│   └── pairs.csv
└── WIDER_FACE/
    ├── WIDER_train/images/
    ├── WIDER_val/images/
    ├── wider_face_annotations/wider_face_split/
    └── annotations/
```

## Notebook 说明

### 01：人脸识别基础

介绍人脸检测、关键点定位、对齐、特征提取、识别与验证的基本概念。

### 02：数据集探索

统计 CelebA 和 LFW 的数据规模、身份与属性分布，并使用 OpenCV、Matplotlib 展示样本和关键点。

### 03：人脸检测

使用 MMDetection 的 Grounding DINO 模型和 `face .` 文本提示检测多张图片，输出人脸框、置信度、推理时间和阈值分析结果。

### 04：关键点定位与 LFW 验证

使用 OpenCV YuNet 检测人脸并定位双眼、鼻尖和左右嘴角，再使用 SFace 对齐人脸、提取特征并完成 6000 个 LFW 配对验证。输出包括：

- 有效配对覆盖率与 Accuracy；
- Precision、Recall、F1 和 Specificity；
- 混淆矩阵、ROC-AUC、ROC 曲线和相似度分布图。

YuNet 和 SFace 权重会在首次运行时自动下载到 `data/models/`。

### 07：ResNet50 + ArcFace 训练

只使用 MS-Celeb-1M 或从中选取的真实子集训练 512 维人脸特征模型，不使用其他数据集替代。代码直接读取 InsightFace 的 `train.rec` 和 `train.idx`。默认配置针对单张 RTX 3080 的两小时预算，按 RecordIO 物理写入顺序保留 100 万张真实 MS1M 图像并重新映射其中实际出现的身份，训练 10 epochs，并在 1.7 小时达到硬上限时保存当前模型。LFW 使用原图与水平翻转图的融合特征进行独立 10 折验证；训练检查点、Loss/Accuracy 曲线和验证结果写入 `work_dirs/arcface_resnet50/`。

### 08：模型优化与部署准备

对任务 5 模型执行 Linear 动态量化，比较量化前后的模型大小、CPU 延迟和 LFW 准确率，并导出、检查和测试 ONNX 模型。最终模型写入 `output/models/`。

## Docker

```bash
docker build -t face-vision-lab:week1 .
docker run --rm face-vision-lab:week1
```

Docker 用于验证项目依赖与基础运行环境；GPU 推理实验在 Windows `.venv` 中运行。
