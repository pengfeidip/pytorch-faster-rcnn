

# Faster RCNN is equivalent to CascadeRCNN with one stage
model=dict(
    type='CascadeRCNN',
    num_stages=1,
    backbone=dict(type='ResNet', depth=50, frozen_stages=1, out_layers=(1, 2, 3, 4)),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[0.0, 0.0, 0.0, 0.0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0/9.0, loss_weight=1.0)),
    roi_extractor=dict(
        type='BasicRoIExtractor',
        roi_layers=[
            dict(type='RoIAlign', spatial_scale=1.0/4.0,  sampling_ratio=2),
            dict(type='RoIAlign', spatial_scale=1.0/8.0,  sampling_ratio=2),
            dict(type='RoIAlign', spatial_scale=1.0/16.0, sampling_ratio=2),
            dict(type='RoIAlign', spatial_scale=1.0/32.0, sampling_ratio=2)],
        output_size=(7, 7),
    ),
    rcnn_head=[
        dict(
            type='RCNNHead',
            in_channels=256,
            roi_out_size=(7, 7),
            fc_channels=[1024, 1024],
            with_avg_pool=False,
            num_classes=21,
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2],
            reg_class_agnostic=False,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
        )
    ]
)


train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou=0.7,
            neg_iou=0.3,
            min_pos_iou=0.3
        ),
        sampler=dict(
            type='RandomSampler',
            max_num=256,
            pos_num=128
        ),
        allowed_border=0
    ),
    rpn_proposal=dict(
        pre_nms=2000,
        post_nms=2000, # used to be post_nms
        max_num=2000,
        nms_iou=0.7,
        min_bbox_size=0,
    ),
    rcnn=[
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou=0.5,
                neg_iou=0.5,
                min_pos_iou=0.5),
            sampler=dict(
                type='RandomSampler',
                max_num=512,
                pos_num=128))
    ],
    stage_loss_weight=[1.0],

    log_level='DEBUG',
    
    total_epochs=14,
    log_file='train.log',
    save_interval=2
)

test_cfg = dict(
    rpn=dict(
        pre_nms=1000,
        post_nms=1000, # same as post_nms
        max_num=1000,
        nms_iou=0.7,
        min_bbox_size=0.0,
    ),
    rcnn=dict(min_score=0.05, nms_iou=0.5, max_per_img=100)
) 

lr_config=dict(
    warmup_iters=500,
    warmup_ratio=1.0/3,
    lr_decay={9:0.1, 12:0.1},
)

optimizer=dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config=dict(grad_clip=dict(max_norm=35, norm_type=2))
ckpt_config=dict(interval=2)


img_norm = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)


train_pipeline=[
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm),
    #dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline=[
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm),
    #dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
]

data = dict(
    train=dict(
        imgs_per_gpu=2,
        ann_file='/home/server2/4T/liyiqing/dataset/PASCAL_VOC_07/voc2007_trainval/voc2007_trainval_no_difficult.json',
        img_prefix='/home/server2/4T/liyiqing/dataset/PASCAL_VOC_07/mmdet_voc2007/VOC2007/JPEGImages',
        pipeline=train_pipeline,
        loader=dict(batch_size=1, num_workers=4, shuffle=True),
    ),
    test=dict(
        imgs_per_gpu=2,
        ann_file='/home/server2/4T/liyiqing/dataset/PASCAL_VOC_07/voc2007_test/voc2007_test_no_difficult.json',
        img_prefix='/home/server2/4T/liyiqing/dataset/PASCAL_VOC_07/mmdet_voc2007/VOC2007/JPEGImages',
        pipeline=test_pipeline,
        loader=dict(batch_size=1, num_workers=4, shuffle=False),
    )
)


