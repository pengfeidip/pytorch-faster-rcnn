
# its name contains caffe, but it does not use caffe models, it simply means ResNet50 with no FPN
# faster rcnn
model=dict(
    type='CascadeRCNN',
    num_stages=1,
    backbone=dict(type='ResNet50', frozen_stages=1, out_layers=(3, ), bn_requires_grad=False),
    rpn_head=dict(
        type='RPNHead',
        in_channels=1024,
        feat_channels=1024,
        anchor_scales=[2, 4, 8, 16, 32],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[16],
        cls_loss_weight=1.0,
        bbox_loss_weight=1.0,
        bbox_loss_beta=1.0/9.0),
    roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer='RoIAlign',
        output_size=(14, 14),
        featmap_strides=[16]
    ),
    shared_head=dict(
        type='ResLayerC5',
        bn_requires_grad=False
    ),
    rcnn_head=[
        dict(
            type='BBoxHead',
            in_channels=2048,
            roi_out_size=(7, 7),
            with_avg_pool=True,
            fc_channels=None,
            num_classes=21,
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2],
            reg_class_agnostic=False,
            bbox_loss_beta=1.0
        ),
        dict(
            type='BBoxHead',
            in_channels=2048,
            roi_out_size=(7, 7),
            with_avg_pool=True,
            fc_channels=None,
            num_classes=21,
            target_means=[.0, .0, .0, .0],
            target_stds=[0.05, 0.05, 0.1, 0.1],
            reg_class_agnostic=True,
            bbox_loss_beta=1.0
        ),
        dict(
            type='BBoxHead',
            in_channels=2048,
            roi_out_size=(7, 7),
            with_avg_pool=True,
            fc_channels=None,
            num_classes=21,
            target_means=[.0, .0, .0, .0],
            target_stds=[0.033, 0.033, 0.067, 0.067],
            reg_class_agnostic=True,
            bbox_loss_beta=1.0
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
        )
    ),
    rpn_proposal=dict(
        pre_nms=12000,
        post_nms=2000,
        nms_iou=0.7,
        min_size=0,
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
                pos_num=128)),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou=0.6,
                neg_iou=0.6,
                min_pos_iou=0.6),
            sampler=dict(
                type='RandomSampler',
                max_num=512,
                pos_num=128)),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou=0.7,
                neg_iou=0.7,
                min_pos_iou=0.7),
            sampler=dict(
                type='RandomSampler',
                max_num=512,
                pos_num=128))
    ],
    stage_loss_weight=[1, 0.5, 0.25],
    
    total_epochs=14,
    optimizer=dict(type='SGD', lr=0.00125, momentum=0.9, weight_decay=0.0001),
    log_file='train.log',
    lr_decay={9:0.1, 12:0.1},
    save_interval=2
)


test_cfg = dict(
    rpn=dict(
        pre_nms=6000,
        post_nms=1000,
        nms_iou=0.7,
        min_size=0.0,
    ),
    rcnn=dict(min_score=0.05, nms_iou=0.5)
) 


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
        ann_file='/home/server2/4T/liyiqing/dataset/PASCAL_VOC_07/voc2007_trainval/voc2007_trainval_no_difficult.json',
        img_prefix='/home/server2/4T/liyiqing/dataset/PASCAL_VOC_07/mmdet_voc2007/VOC2007/JPEGImages',
        pipeline=train_pipeline,
        loader=dict(batch_size=1, num_workers=4, shuffle=True),
    ),
    test=dict(
        ann_file='/home/server2/4T/liyiqing/dataset/PASCAL_VOC_07/voc2007_test/voc2007_test_no_difficult.json',
        img_prefix='/home/server2/4T/liyiqing/dataset/PASCAL_VOC_07/mmdet_voc2007/VOC2007/JPEGImages',
        pipeline=test_pipeline,
        loader=dict(batch_size=1, num_workers=2, shuffle=False),
    )
)

