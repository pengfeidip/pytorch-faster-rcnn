import torch
import logging

# This is the first config that uses the new version implementation(vectorization)
LR = None
MAX_EPOCHS = 16

train_data_cfg = dict(
    img_dir='/home/server2/4T/liyiqing/dataset/PASCAL_VOC_07/voc2007_trainval/VOC2007/JPEGImages',
    json='/home/server2/4T/liyiqing/projects/migrate-to-simple/'
    'pytorch-faster-rcnn-1-no_difficult/voc2007_trainval_no_difficult.json',
    img_size=(1000,600),
    img_norm=dict(mean=[0.390, 0.423, 0.446], std=[0.282, 0.270, 0.273]),
    loader_cfg=dict(batch_size=1, num_workers=2, shuffle=True),
)

test_data_cfg = dict(
    img_dir='/home/server2/4T/liyiqing/dataset/PASCAL_VOC_07/voc2007_test/VOC2007/JPEGImages',
    loader_cfg=dict(batch_size=1, num_workers=2, shuffle=False),
)


model = dict(
    num_classes=20,
    anchor_scales=[8, 16, 32],
    anchor_aspect_ratios=[0.5, 1.0, 2.0],
    anchor_pos_iou=0.7,
    anchor_neg_iou=0.3,
    anchor_max_pos=128,
    anchor_max_targets=256,
    train_props_pre_nms=12000,
    train_props_post_nms=2000,
    train_props_nms_iou=0.7,
    train_props_min_size=16,
    test_props_pre_nms=6000,
    test_props_post_nms=300,
    test_props_nms_iou=0.5,
    test_props_min_size=16,
    props_pos_iou=0.5,
    props_neg_iou_hi=0.5,
    props_neg_iou_lo=0.1,
    props_max_pos=32,
    props_max_targets=128,
    roi_pool_size=(7, 7),
    transfer_rcnn_fc=False
)


train_cfg = dict(
    max_epochs=MAX_EPOCHS,
    optim=torch.optim.SGD,
    optim_kwargs=dict(lr=LR,momentum=0.9,weight_decay=0.0005),
    rpn_loss_lambda=1.0,
    rcnn_loss_lambda=1.0,
    loss_lambda=1.0,
    log_file='train_16epochs.log',
    lr_scheduler=lambda e : 0.001 if e <=12 else 0.0001,
    log_level=logging.DEBUG,
    device=torch.device('cpu'),
    save_interval=2
)


test_cfg = dict(
    min_score=0.05
)
