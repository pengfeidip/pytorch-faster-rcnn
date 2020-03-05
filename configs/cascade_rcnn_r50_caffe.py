
# its name contains caffe, but it does not use caffe models, it simply means ResNet50 with no FPN

model=dict(
    type='CascadeRCNN',
    backbone=dict(type='ResNet50', frozen_stages=1),
    rpn_head=dict(
        type='RPNHead',
        in_channels=1024,
        feat_channels=256,
        anchor_base=16,
        anchor_scales=[4,8,16,32],
        anchor_ratios=[0.5,1.0,2.0],
        cls_loss_weight=1.0,
        bbox_loss_weight=1.0,
        bbox_loss_beta=1.0/9.0),
    roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer='RoIPool',
        output_size=7,
        spatial_scale=1.0/16.0
    ),
    shared_head=dict(
        type='ResLayerC5',
        bn_requires_grad=True
    ),
    rcnn_head=[
        dict(
            type='BBoxHead',
            in_channels=1024,
            fc_channels=[1024, 1024],
            num_classes=21,
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2],
            reg_class_agnostic=False,
            bbox_loss_beta=1.0
        ),
        dict(
            type='BBoxHead',
            in_channels=1024,
            fc_channels=[1024, 1024],
            roi_out_size=(7, 7),
            roi_extractor='RoIPool',
            num_classes=21,
            target_means=[.0, .0, .0, .0],
            target_stds=[0.05, 0.05, 0.1, 0.1],
            reg_class_agnostic=False,
            bbox_loss_beta=1.0
        ),
        dict(
            type='BBoxHead',
            in_channels=1024,
            fc_channels=[1024, 1024],
            roi_out_size=(7, 7),
            roi_extractor='RoIPool',
            num_classes=21,
            target_means=[.0, .0, .0, .0],
            target_stds=[0.033, 0.033, 0.067, 0.067],
            reg_class_agnostic=False,
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
        min_size=16,
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
                max_num=128,
                pos_num=32)),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou=0.6,
                neg_iou=0.6,
                min_pos_iou=0.6),
            sampler=dict(
                type='RandomSampler',
                max_num=128,
                pos_num=32)),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou=0.7,
                neg_iou=0.7,
                min_pos_iou=0.7),
            sampler=dict(
                type='RandomSampler',
                max_num=128,
                pos_num=32))
    ],
    stage_loss_weight=[1, 0.5, 0.25],
    
    total_epochs=14,
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005),
    log_file='train.log',
    lr_decay={11: 0.1},
    save_interval=2
)


test_cfg = dict(
    rpn=dict(
        pre_nms=6000,
        post_nms=300,
        nms_iou=0.7,
        min_size=0.0,
    ),
    rcnn=dict(min_score=0.05, nms_iou=0.3)
) 


data = dict(
    train=dict(
        voc_data_dir='/home/lee/datasets/voc2007_comb/VOC2007',
        min_size=600,
        max_size=1000,
        loader=dict(batch_size=1, num_workers=4, shuffle=True)
    ),
    test=dict(
        voc_data_dir='/home/lee/datasets/voc2007_comb/VOC2007',
        loader=dict(batch_size=1, num_workers=4, shuffle=False)
    )
)
