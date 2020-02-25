import torch
import logging

# This is the first config that uses the new version implementation(vectorization)
INIT_LR = 0.001
MAX_EPOCHS = 14

train_data_cfg = dict(
    voc_data_dir='/home/server2/4T/liyiqing/dataset/PASCAL_VOC_07/voc2007_all/VOC2007',
    min_size=600,
    max_size=1000,
    loader_cfg=dict(batch_size=1, num_workers=6, shuffle=True),
)

test_data_cfg = dict(
    voc_data_dir='/home/server2/4T/liyiqing/dataset/PASCAL_VOC_07/voc2007_all/VOC2007',
    loader_cfg=dict(batch_size=1, num_workers=2, shuffle=False),
)


model = dict(
    num_classes=20,
    anchor_scales=[8, 16, 32], # [4,8,16,32,64] in mmdet
    anchor_aspect_ratios=[0.5, 1.0, 2.0],
    anchor_pos_iou=0.7,
    anchor_neg_iou=0.3,
    anchor_min_pos_iou=0.0,   # paper=0, set to 0.3 in mmdet 
    anchor_max_pos=128,
    anchor_max_targets=256,
    props_nms_iou=0.7,
    nms_iou=0.3,
    train_props_pre_nms=12000,
    train_props_post_nms=2000,
    train_props_min_size=16,
    test_props_pre_nms=6000,
    test_props_post_nms=300,
    test_props_min_size=16,
    props_pos_iou=0.5,
    props_neg_iou_hi=0.5,
    props_neg_iou_lo=0.0,
    props_max_pos=32,
    props_max_targets=128,
    roi_out_size=(7, 7),
    roi_layer='RoIPool',
    transfer_backbone_cls=True,
    freeze_first_layers=True,
    backbone_type='VGG16',
    rpn_hidden_channels=512,
    rcnn_fc_hidden_channels=4096
)


train_cfg = dict(
    max_epochs=MAX_EPOCHS,
    optim=torch.optim.SGD,
    optim_kwargs=dict(lr=INIT_LR,momentum=0.9,weight_decay=0.0005),
    rpn_loss_lambda=1.0,
    rcnn_loss_lambda=1.0,
    loss_lambda=1.0,
    log_file='train_14epochs.log',
    decay_epoch = {11:0.1},
    log_level=logging.DEBUG,
    device=torch.device('cpu'),
    save_interval=2,
    param_normalize_mean=(0.0, 0.0, 0.0, 0.0),
    param_normalize_std=(0.1, 0.1, 0.2, 0.2),
    wloss_lambda=0.0,   # paper=0.0, 1.0 is validated to be OK
    bloss_lambda=0.0    # paper=0.0, 2.0 is validated to be OK
)


test_cfg = dict(
    min_score=0.05
)
