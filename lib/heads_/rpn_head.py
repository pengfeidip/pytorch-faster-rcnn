from torch import nn
from ..utils import init_module_normal
from mmcv.cnn import normal_init
from .anchor_head import AnchorHead
import logging

class RPNHead_v2(AnchorHead):
    def __init__(self,
                 in_channels,
                 feat_channels,
                 anchor_scales=[8],
                 anchor_ratios=[0.5, 1.0, 2.0],
                 anchor_strides=[4, 8, 16, 32, 64],
                 target_means=[0.0, 0.0, 0.0, 0.0],
                 target_stds=[1.0, 1.0, 1.0, 1.0],
                 loss_cls=None,
                 loss_bbox=None):
        self.in_channels=in_channels
        self.feat_channels=feat_channels
        
        self.anchor_scales=anchor_scales
        self.anchor_ratios=anchor_ratios
        self.anchor_strides=anchor_strides
        self.loss_cls=loss_cls
        self.loss_bbox=loss_bbox
        
        
        super(RPNHead_v2, self).__init__(
            num_classes=2,
            anchor_scales=anchor_scales,
            anchor_ratios=anchor_ratios,
            anchor_strides=anchor_strides,
            target_means=target_means,
            target_stds=target_stds,
            use_sigmoid=False,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox)
        
        self.init_layers()

    def init_layers(self):
        self.conv = nn.Conv2d(
            self.in_channels, self.feat_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Conv2d(
            self.feat_channels, self.num_anchors*self.num_classes, kernel_size=1)
        self.regressor = nn.Conv2d(self.feat_channels, self.num_anchors*4, kernel_size=1)
        logging.info('Constructed RPNHead with in_channels={}, feat_channels={}, num_levels={}, num_anchors={}'\
                     .format(self.in_channels, self.feat_channels, len(self.anchor_creators), self.num_anchors))
        
    def init_weights(self):
        init_module_normal(self.conv, mean=0.0, std=0.01)
        init_module_normal(self.classifier, mean=0.0, std=0.01)
        init_module_normal(self.regressor, mean=0.0, std=0.01)
        logging.info('Initialized weights for RPNHead.')
        
    def forward(self, xs):
        print('in RPNHead forward, input:')
        for tmp in xs:
            print(tmp.shape)
        conv_outs = [self.relu(self.conv(x)) for x in xs]
        cls_outs = [self.classifier(x) for x in conv_outs]
        reg_outs = [self.regressor(x) for x in conv_outs]
        return cls_outs, reg_outs
