
model=dict(
    type='FCOS',
    backbone=dict(type='ResNet', depth=50, frozen_stages=1, out_layers=(1, 2, 3, 4)),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        extra_use_convs=True,
        extra_convs_on_inputs=False,
        num_outs=5,
        relu_before_extra_convs=False
    ),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=21,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        reg_std=600,
        reg_mean=0,
        reg_coef=[1.0, 1.0, 1.0, 1.0, 1.0],
        reg_coef_trainable=True,
        atss_cfg=dict(topk=9, scale=8),
        loss_cls=dict(
            type='FocalLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
        loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)
    )
)

train_cfg = dict(
    allowed_border=-1,
    total_epochs=24,
    log_file='train.log',
    log_level='DEBUG'
)

test_cfg = dict(
    pre_nms=1000,
    min_bbox_size=0,
    min_score=0.05,
    nms_iou=0.6,
    nms_type='strict',
    max_per_img=100
) 

lr_config=dict(
    warmup_iters=500,
    warmup_ratio=0.001,
    lr_decay={17:0.1, 22:0.1},
)

optimizer=dict(type='SGD', lr=0.00075, momentum=0.9, weight_decay=0.0001)
optimizer_config=dict(grad_clip=None)
#optimizer_config=dict(grad_clip=None)

ckpt_config=dict(interval=2)

img_norm = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline=[
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline=[
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
]

data = dict(
    train=dict(
        imgs_per_gpu=2,
        ann_file='/home/server2/4T/liyiqing/dataset/PASCAL_VOC_07/voc2007_trainval/voc2007_trainval_no_difficult.json',
        img_prefix='/home/server2/4T/liyiqing/dataset/PASCAL_VOC_07/mmdet_voc2007/VOC2007/JPEGImages',
        pipeline=train_pipeline,
        loader=dict(batch_size=1, num_workers=6, shuffle=True),
    ),
    test=dict(
        imgs_per_gpu=2,
        ann_file='/home/server2/4T/liyiqing/dataset/PASCAL_VOC_07/voc2007_test/voc2007_test_no_difficult.json',
        img_prefix='/home/server2/4T/liyiqing/dataset/PASCAL_VOC_07/mmdet_voc2007/VOC2007/JPEGImages',
        pipeline=test_pipeline,
        loader=dict(batch_size=1, num_workers=4, shuffle=False),
    )
)


