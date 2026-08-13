"""MMDetection 3.3 RetinaNet baseline for one-class WIDER FACE detection."""

_base_ = "mmdet::retinanet/retinanet_r50_fpn_1x_coco.py"

data_root = "data/WIDER_FACE/"
metainfo = {"classes": ("face",), "palette": [(220, 20, 60)]}

model = dict(
    bbox_head=dict(num_classes=1),
    test_cfg=dict(
        nms_pre=5000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type="nms", iou_threshold=0.5),
        max_per_img=1000,
    ),
)

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file="annotations/wider_face_train.json",
        data_prefix=dict(img="WIDER_train/images/"),
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file="annotations/wider_face_val.json",
        data_prefix=dict(img="WIDER_val/images/"),
    ),
)
test_dataloader = val_dataloader

# WIDER FACE uses AP at IoU 0.5; AR values provide recall at fixed proposal counts.
val_evaluator = dict(
    ann_file=data_root + "annotations/wider_face_val.json",
    iou_thrs=[0.5],
    proposal_nums=(100, 300, 1000),
)
test_evaluator = val_evaluator

# The inherited COCO schedule is 12 epochs. LR is reduced for one GPU x batch 2.
optim_wrapper = dict(optimizer=dict(lr=0.0025))
load_from = (
    "https://download.openmmlab.com/mmdetection/v2.0/retinanet/"
    "retinanet_r50_fpn_1x_coco/"
    "retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth"
)
