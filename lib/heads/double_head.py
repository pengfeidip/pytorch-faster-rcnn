from .bbox_head import BBoxHead
from .. import utils
from ..utils import init_module_normal
from torch import nn
from .. import debug
import logging, torch
from mmcv.cnn import normal_init, constant_init


class DoubleHead(BBoxHead):
    def __init__(self,
                 in_channels,
                 roi_out_size=7,
                 fc_channels=[1024, 1024],
                 conv_channels=[1024, 1024, 1024],
                 conv_strides=[1, 2, 2],
                 conv_paddings=[1, 0, 0],
                 num_classes=21,
                 target_means=[0.0, 0.0, 0.0, 0.0],
                 target_stds=[0.1, 0.1, 0.2, 0.2],
                 lambda_fc=0.5,
                 lambda_conv=0.5,
                 reg_class_agnostic=False,
                 loss_cls=None,
                 loss_bbox=None):
        super(DoubleHead, self).__init__(
            num_classes,
            target_means,
            target_stds,
            reg_class_agnostic,
            loss_cls,
            loss_bbox)

        self.in_channels=in_channels
        self.roi_out_size=utils.to_pair(roi_out_size)
        self.fc_channels=fc_channels
        self.conv_channels=conv_channels
        self.conv_strides=conv_strides
        self.conv_paddings=conv_paddings
        self.lambda_fc=lambda_fc
        self.lambda_conv=lambda_conv
        
        assert self.roi_out_size[0] == self.roi_out_size[1] == 7
        assert len(conv_channels) == len(conv_strides) == len(conv_paddings)
        
        
        self.init_layers()

    def init_layers(self):
        # the fc head
        spatial_size = self.roi_out_size[0] * self.roi_out_size[1]
        fc_in_channels = spatial_size * self.in_channels
        fc_layers = []
        for fc_channels in self.fc_channels:
            fc_layers.append(nn.Linear(fc_in_channels, fc_channels))
            fc_layers.append(nn.ReLU(inplace=True))
            fc_in_channels = fc_channels
        self.fc_layer = nn.Sequential(*fc_layers)
        self.fc_classifier = nn.Linear(fc_channels, self.cls_channels)
        self.fc_regressor  = nn.Linear(fc_channels, 4 if self.reg_class_agnostic else self.num_classes * 4)

        # the conv head
        conv_layers = []
        conv_in_channels = self.in_channels
        for i in range(len(self.conv_channels)):
            conv_channels = self.conv_channels[i]
            conv_layers.append(nn.Conv2d(conv_in_channels, conv_channels, 3,
                                         padding=self.conv_paddings[i], stride=self.conv_strides[i]))
            conv_layers.append(nn.BatchNorm2d(conv_channels))
            conv_in_channels = conv_channels
        conv_layers.append(nn.Tanh())
        self.conv_layer = nn.Sequential(*conv_layers)
        self.conv_classifier = nn.Linear(conv_channels, self.cls_channels)
        self.conv_regressor  = nn.Linear(conv_channels, 4 if self.reg_class_agnostic else self.num_classes * 4)
        
    def init_weights(self):
        for i, fc in enumerate(self.fc_layer):
            if isinstance(fc, nn.Linear):
                nn.init.xavier_uniform_(fc.weight)
                nn.init.constant_(fc.bias, 0)
                logging.info('Init {} fc layer with xavier and constant'.format(i))
        normal_init(self.fc_classifier, std=0.01)
        normal_init(self.fc_regressor, std=0.001)
        for i, conv in enumerate(self.conv_layer):
            if isinstance(conv, nn.Conv2d):
                normal_init(conv, std=0.01)
                logging.info('Init {} conv layer with normal distribution'.format(i))
            if isinstance(conv, nn.BatchNorm2d):
                constant_init(conv, 1, bias=0)
                logging.info('Init {} conv layer (BN) with constant')
        normal_init(self.conv_classifier, std=0.01)
        normal_init(self.conv_regressor, std=0.001)
        
                

    # for input, rois is a list of roi output from diff imgs
    # for output, cls_outs and reg_outs are list of results for diff imgs
    # notice we concate inputs and then split outputs to save time(not experimentally compared)
    # split case for train and test
    def forward(self, rois):
        roi_sizes = [roi.shape[0] for roi in rois]
        x = torch.cat(rois, dim=0)
        batch_size = x.shape[0]
        fc_x = x.view(batch_size, -1)
        fc_x = self.fc_layer(fc_x)
        fc_cls_out = self.fc_classifier(fc_x)
        fc_reg_out = self.fc_regressor(fc_x)

        conv_x = self.conv_layer(x).view(batch_size, -1)
        conv_cls_out = self.conv_classifier(conv_x)
        conv_reg_out = self.conv_regressor(conv_x)

        fc_cls_outs, fc_reg_outs = [], []
        conv_cls_outs, conv_reg_outs = [], []
        start_idx = 0
        for sz in roi_sizes:
            fc_cls_outs.append(fc_cls_out[start_idx:start_idx+sz])
            fc_reg_outs.append(fc_reg_out[start_idx:start_idx+sz])
            conv_cls_outs.append(conv_cls_out[start_idx:start_idx+sz])
            conv_reg_outs.append(conv_reg_out[start_idx:start_idx+sz])
            start_idx += sz
        if self.training:
            return fc_cls_outs, conv_reg_outs
        else:
            return fc_cls_outs, conv_reg_outs


    # cls_outs and reg_outs are outputs of forward
    # returns cls_loss, reg_loss
    def calc_loss(self, cls_outs, reg_outs, tar_labels, tar_params, train_cfg):
        cls_out = torch.cat(cls_outs, dim=0)
        reg_out = torch.cat(reg_outs, dim=0)
        
        tar_label = torch.cat(tar_labels, dim=0)
        tar_param = torch.cat(tar_params, dim=1)
        
        cls_loss, reg_loss = self.calc_loss_all(cls_out, reg_out, tar_label, tar_param, train_cfg)
        return cls_loss, reg_loss
