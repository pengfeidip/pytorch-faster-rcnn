import torch.nn as nn


class GuidedAnchor(nn.Module):
    def __init__(self,
                 in_channels=256,
                 anchor_scales=None,
                 anchor_ratios=None,
                 anchor_strides=None, 
                 anchoring_means=[0.0, 0.0, 0.0, 0.0],
                 anchoring_stds=[0.07, 0.07, 0.14, 0.14],
                 loc_filter=0.01,
                 loss_loc=None,
                 loss_shape=None,
    ):
        self.in_channels=in_channels
        self.anchor_scales=anchor_scales
        self.anchor_ratios=anchor_ratios
        self.anchor_strides=anchor_strides
        self.anchoring_means=anchoring_means
        self.anchoring_stds=anchoring_stds
        self.loc_filter=loc_filter

        from ..builder import build_module
        self.loss_loc=build_module(loss_loc)
        self.loss_shape=build_module(loss_shape)
        
        super(GuidedAnchor, self).__init__()

    def init_layers(self):
        self.loc_layer = nn.Conv2d(self.in_channels, 1, 3, padding=1)
        self.shape_layer = nn.Conv2d(self.in_channels, 2, 3, padding=1)
        # define DCN
        
        pass

    def init_weights(self):
        pass

    def forward(self, feats):
        pass
    pass


class GARPNHead(nn.Module):

    def __init__(self,
                 in_channels=256,
                 feat_channels=256,
                 octave_base_scale=8,
                 scales_per_octave=3,
                 octave_ratios=[0.5, 1.0, 2.0],
                 anchor_strides=[4, 8, 16, 32, 64],
                 anchoring_means=[0.0, 0.0, 0.0, 0.0],
                 anchoring_stds=[0.07, 0.07, 0.14, 0.14],
                 target_means=[0.0, 0.0, 0.0, 0.0],
                 target_stds=[0.07, 0.07, 0.11, 0.11],
                 loc_filter_thr=0.01,
                 loss_loc=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_shape=dict(type='BoundedIoULoss', beta=0.2, loss_weight=1.0),
                 loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                 loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)):
        self.in_channels=in_channels
        self.feat_channels=feat_channels
        self.octave_base_scale=octave_base_scale
        self.scales_per_octave=scales_per_octave
        self.octave_ratios=octave_ratios

        self.anchor_strides=anchor_strides
        self.anchoring_means=anchoring_means
        self.anchoring_stds=anchoring_stds
        self.target_means=target_means
        self.target_stds=target_stds
        self.loc_filter=loc_filter
        
        self.loss_cls_cfg = loss_cls
        self.loss_bbox_cfg = loss_bbox
        self.loss_loc_cfg = loss_loc
        self.loss_shape_cfg = loss_shape_cfg
        
        self.loss_cls = build_module(loss_cls)
        self.loss_bbox = build_module(loss_bbox)

        octave_scales = [2**(i/scales_per_octave) for i in range(scales_per_octave)]
        anchor_scales = [octave_base_scale*octave_scale for octave_scale in octave_scales]

        
        pass
