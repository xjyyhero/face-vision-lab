# 任务 3：WIDER FACE 人脸检测训练

本目录完成阶段二任务 3 的最小训练闭环：理解 MTCNN 与 RetinaFace，转换 WIDER FACE 标注，使用 MMDetection 训练 RetinaNet 基线，并在验证集上计算 AP/AR 和生成检测结果图。

完整实验流程请运行 [`05_wider_face_detection_training.ipynb`](05_wider_face_detection_training.ipynb)。Notebook 负责讲解、训练、评估与可视化；独立配置和转换脚本供 MMDetection 底层复用。

> RetinaFace 与 RetinaNet 不是同一个模型。MMDetection 3.3.0 没有内置 RetinaFace 配置，因此原理部分学习 RetinaFace，训练部分复用 MMDetection 官方 RetinaNet 基线。

## 3.1 算法原理

### MTCNN

MTCNN 将人脸检测和五点定位放在一个三级级联框架中：

1. **P-Net** 在图像金字塔上快速生成候选框，并进行边界框回归；
2. **R-Net** 过滤大量错误候选框，再次校正位置；
3. **O-Net** 输出最终人脸框和双眼、鼻尖、左右嘴角五个关键点。

三个网络共同学习人脸分类、边界框回归和关键点定位。级联结构容易理解且适合 CPU 推理，但需要多次裁剪和前向传播，在大量小脸场景下效率受限。

### RetinaFace

RetinaFace 是基于特征金字塔的单阶段密集人脸定位模型，在多个尺度上联合预测：

- 人脸/背景分类；
- 人脸边界框；
- 五个人脸关键点；
- 额外的像素级三维人脸形状监督（完整模型）。

多尺度特征和联合监督使 RetinaFace 更适合 WIDER FACE 中的小脸、遮挡和大姿态场景。与 MTCNN 相比，它不需要三级候选框级联，GPU 批量推理通常更高效。

参考资料：

- [MTCNN 论文](https://arxiv.org/abs/1604.02878)
- [RetinaFace 论文](https://openaccess.thecvf.com/content_CVPR_2020/html/Deng_RetinaFace_Single-Shot_Multi-Level_Face_Localisation_in_the_Wild_CVPR_2020_paper.html)
- [WIDER FACE 官方页面](https://shuoyang1213.me/WIDERFACE/)

## 3.2 数据与训练

从 WIDER FACE 官方页面下载 `WIDER_train.zip`、`WIDER_val.zip` 和标注文件，解压为：

```text
data/WIDER_FACE/
├── WIDER_train/images/
├── WIDER_val/images/
└── wider_face_annotations/
    └── wider_face_split/
        ├── wider_face_train_bbx_gt.txt
        └── wider_face_val_bbx_gt.txt
```

将官方文本标注转换为 MMDetection 可直接读取的 COCO JSON：

```bash
python face_detection/wider_face_to_coco.py data/WIDER_FACE train
python face_detection/wider_face_to_coco.py data/WIDER_FACE val
```

在包含 MMDetection 源码的目录中执行训练。`MMDET_ROOT` 表示 MMDetection 仓库目录：

```bash
python "$MMDET_ROOT/tools/train.py" \
  face_detection/retinanet_r50_fpn_wider_face.py \
  --work-dir work_dirs/retinanet_wider_face
```

配置继承 MMDetection 3.3.0 官方 RetinaNet R50-FPN 12 epoch 配置，只覆盖人脸类别、WIDER FACE 数据路径、单 GPU 学习率及每张图最多保留的检测框数。

## 3.3 评估与结果展示

验证集评估：

```bash
python "$MMDET_ROOT/tools/test.py" \
  face_detection/retinanet_r50_fpn_wider_face.py \
  work_dirs/retinanet_wider_face/epoch_12.pth
```

保存带检测框的验证集图片：

```bash
python "$MMDET_ROOT/tools/test.py" \
  face_detection/retinanet_r50_fpn_wider_face.py \
  work_dirs/retinanet_wider_face/epoch_12.pth \
  --show-dir work_dirs/retinanet_wider_face/visualizations
```

本配置按照 WIDER FACE 的 IoU 0.5 条件计算 COCO `bbox_mAP`，并输出不同最大检测数量下的 AR。官方 Easy、Medium、Hard 子集 AP 需要在得到预测结果后再运行 WIDER FACE 官方评估脚本。

运行前可先检查标注解析器和完整配置：

```bash
python face_detection/wider_face_to_coco.py --self-test
python "$MMDET_ROOT/tools/misc/print_config.py" \
  face_detection/retinanet_r50_fpn_wider_face.py
```
