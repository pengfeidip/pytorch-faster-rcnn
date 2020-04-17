from torch import nn
from .anchor_head import AnchorHead
from mmcv.cnn import normal_init
import numpy as np
import logging

class RetinaHead(AnchorHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs,
                 feat_channels,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 anchor_ratios=[0.5, 1.0, 2.0],
                 anchor_strides=[8, 16, 32, 64, 128],
                 target_means=[0.0, 0.0, 0.0, 0.0],
                 target_stds=[1.0, 1.0, 1.0, 1.0],
                 loss_cls=None,
                 loss_bbox=None):
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_classes = num_classes
        self.cls_channels = num_classes-1

        self.stacked_convs = stacked_convs
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        octave_scales = [2**(i/scales_per_octave) for i in range(scales_per_octave)]
        anchor_scales = [octave_base_scale*octave_scale for octave_scale in octave_scales]

        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_strides = anchor_strides
        self.base_sizes = anchor_strides
        super(RetinaHead, self).__init__(num_classes,
                                            anchor_scales,
                                            anchor_ratios,
                                            anchor_strides,
                                            target_means,
                                            target_stds,
                                            loss_cls,
                                            loss_bbox)
        
        self.init_layers()
        logging.info('Constructed RetinaHead with in_channels={}, feat_channels={}, num_levels={}, num_anchors={}'\
                     .format(in_channels, feat_channels, len(self.anchor_creators), self.num_anchors))
        logging.info('RetinaHead anchor_scales: {}, anchor_ratios: {}, base_sizes: {}'\
                     .format(self.anchor_scales, self.anchor_ratios, self.base_sizes))

    def init_layers(self):
        cls_convs = []
        reg_convs = []
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            cls_convs.append(
                nn.Conv2d(chn, self.feat_channels, 3, padding=1))
            cls_convs.append(nn.ReLU(inplace=True))
            reg_convs.append(
                nn.Conv2d(chn, self.feat_channels, 3, padding=1))
            reg_convs.append(nn.ReLU(inplace=True))
        self.cls_convs = nn.Sequential(*cls_convs)
        self.reg_convs = nn.Sequential(*reg_convs)
        self.retina_cls = nn.Conv2d(self.feat_channels,
                                    self.num_anchors * (self.cls_channels),
                                    3,
                                    padding=1)
        self.retina_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 3, padding=1)

    def init_weights(self):
        for m in self.cls_convs:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        for m in self.reg_convs:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        prior_prob = 0.01
        bias_init = float(-np.log((1-prior_prob)/ prior_prob))
        normal_init(self.retina_cls, std=0.01, bias=bias_init)
        normal_init(self.retina_reg, std=0.01)
        logging.info('Initialized weights for RetinaHead.')


    def forward(self, xs):
        cls_conv_outs = [self.cls_convs(x) for x in xs]
        reg_conv_outs = [self.reg_convs(x) for x in xs]
        cls_outs = [self.retina_cls(x) for x in cls_conv_outs]
        reg_outs = [self.retina_reg(x) for x in reg_conv_outs]
        return cls_outs, reg_outs

    
