# 项目环境配置说明

## 1. 开发环境

- 操作系统：macOS
- 处理器：Apple Silicon
- 环境管理：Anaconda
- Conda 环境名称：`face-lab`
- Python：3.14.6
- Docker：29.5.3
- 开发工具：Visual Studio Code、Jupyter Notebook

## 2. Python 依赖版本

| 软件 | 版本 |
|---|---:|
| PyTorch | 2.13.0 |
| TorchVision | 0.28.0 |
| OpenCV | 5.0.0.93 |
| Matplotlib | 3.11.1 |
| MMCV Lite | 2.1.0 |
| MMEngine | 0.10.7 |
| MMDetection | 3.3.0 |

完整依赖记录在项目根目录的 `requirements.txt` 中。

## 3. 创建 Python 环境

```bash
conda create -n face-lab python=3.14 -y
conda activate face-lab
```

安装项目依赖：

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

安装并注册 Jupyter 内核：

```bash
python -m pip install jupyter ipykernel
python -m ipykernel install --user --name face-lab --display-name "face-lab"
```

启动 Jupyter Notebook：

```bash
jupyter notebook
```

## 4. Python 环境验证

运行 `test_env.ipynb`，导入并输出各个库的版本：

```python
import torch
import cv2
import mmcv
import mmengine
import mmdet

print("PyTorch:", torch.__version__)
print("OpenCV:", cv2.__version__)
print("MMCV:", mmcv.__version__)
print("MMEngine:", mmengine.__version__)
print("MMDetection:", mmdet.__version__)
print("Apple MPS 可用:", torch.backends.mps.is_available())
```

验证结果：

```text
PyTorch: 2.13.0
OpenCV: 5.0.0
MMCV: 2.1.0
MMEngine: 0.10.7
MMDetection: 3.3.0
Apple MPS 可用: True
环境安装成功！
```

## 5. Docker 环境

项目根目录包含以下 Docker 配置文件：

- `Dockerfile`
- `requirements.txt`
- `.dockerignore`

构建 Docker 镜像：

```bash
docker build -t face-vision-lab:week1 .
```

运行 Docker 容器：

```bash
docker run --rm face-vision-lab:week1
```

运行结果：

```text
Hello World from Docker!
PyTorch: 2.13.0+cu130
OpenCV: 5.0.0
MMDetection: 3.3.0
```

Docker 镜像中的 Linux 环境无法使用 macOS 的 MPS，因此容器默认使用 CPU。

## 6. OpenCV 示例

OpenCV 图像处理程序位于：

```text
openCV_sample/image_processing.ipynb
```

该程序实现了：

- 读取并显示图片
- 彩色图片灰度化
- 高斯模糊降噪
- Canny 边缘检测
- 保存处理结果

处理后的图片保存在：

```text
openCV_sample/outputs/
```