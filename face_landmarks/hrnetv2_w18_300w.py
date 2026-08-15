"""One-epoch MMPose HRNetV2-W18 fine-tuning baseline on 300W."""

_base_ = (
    "mmpose::face_2d_keypoint/topdown_heatmap/300w/"
    "td-hm_hrnetv2-w18_8xb64-60e_300w-256x256.py"
)

data_root = "data/300w/"

# Resource-limited baseline: fine-tune the official 300W model for one epoch.
train_cfg = dict(max_epochs=1, val_interval=1)
train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=dict(data_root=data_root),
)
val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=dict(data_root=data_root),
)
test_dataloader = val_dataloader

optim_wrapper = dict(optimizer=dict(lr=1e-4))
param_scheduler = []
default_hooks = dict(checkpoint=dict(save_best="NME", rule="less", interval=1))

load_from = (
    "https://download.openmmlab.com/mmpose/face/hrnetv2/"
    "hrnetv2_w18_300w_256x256-eea53406_20211019.pth"
)
