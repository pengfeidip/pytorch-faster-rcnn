import torch
import logging

train_data_cfg = dict(
    img_dir='/home/lee/datasets/VOCdevkit/VOC2007/JPEGImages',
    json='data/voc2007_trainval.json',
    img_size=(1000,600),
    img_norm=dict(mean=[], std=[])
    loader_cfg=dict(batch_size=1, num_workers=2, shuffle=True),
)

test_data_cfg = dict(
    img_dir = ''
)

model = dict(
    num_classes=20,
    anchor_scales=[128, 256, 512],
    anchor_aspect_ratios=[1.0, 0.5, 2.0], 
    anchor_pos_iou=0.7,
    anchor_neg_iou=0.3,
    anchor_max_pos=128,
    anchor_max_targets=256,
    train_props_pre_nms=12000,
    train_props_post_nms=2000,
    train_props_nms_iou=0.7,
    test_props_pre_nms=6000,
    test_props_post_nms=300,
    test_props_nms_iou=0.5,
    props_pos_iou=0.5,
    props_neg_iou=0.1,
    props_max_pos=32,
    props_max_targets=128,
    roi_pool_size=(7, 7),
    transfer_rcnn_fc=True
)


train_cfg = dict(
    max_epochs=20,
    optim=torch.optim.SGD,
    optim_kwargs=dict(lr=0.001,momentum=0.9,weight_decay=0.0005),
    rpn_loss_lambda=10.0,
    rcnn_loss_lambda=10.0,
    loss_lambda=1.0,
    log_file='train.log',
    log_level=logging.INFO,
    device=torch.device('cpu')
)


test_cfg = dict(
    
)
