# WIDER FACE 人脸检测实验报告

## 1. 实验目标

使用 MMDetection 3.3.0 在 WIDER FACE 训练集上微调 RetinaNet R50-FPN，并在验证集上评估人脸检测性能。

## 2. 实验环境

| 项目 | 配置 |
|---|---|
| 操作系统 | 待填写 |
| GPU | 待填写 |
| PyTorch / CUDA | 待填写 |
| MMDetection / MMCV | 3.3.0 / 2.1.0 |
| 训练轮数 | 12 epochs |
| Batch size | 2 |
| 初始学习率 | 0.0025 |

## 3. 数据集

WIDER FACE 共包含 32,203 张图像和 393,703 个标注人脸，覆盖尺度、姿态和遮挡变化。本实验使用官方训练集训练、验证集评估；标注先转换为 COCO JSON。

转换后的实际统计：

| 划分 | 图像数 | 人脸框数 |
|---|---:|---:|
| Train | 待填写 | 待填写 |
| Validation | 待填写 | 待填写 |

## 4. 模型与训练设置

- 模型：RetinaNet R50-FPN
- 初始化：COCO 预训练权重
- 输入与数据增强：继承 MMDetection 官方 RetinaNet 1x 配置
- 优化器：SGD
- 评估条件：IoU = 0.5

## 5. 实验结果

| 指标 | 结果 |
|---|---:|
| bbox mAP@0.5 | 待填写 |
| AR@100 | 待填写 |
| AR@300 | 待填写 |
| AR@1000 | 待填写 |

训练日志与损失曲线：待补充。

## 6. 检测结果展示

从 `work_dirs/retinanet_wider_face/visualizations/` 选择包含小脸、遮挡和大姿态的代表性图片，并说明成功与失败案例。

## 7. 分析与结论

待训练后结合误检、漏检和小脸检测表现填写。
