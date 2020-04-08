from .bbox_head import BBoxHead_v2
from .. import utils
from ..utils import init_module_normal
from torch import nn
import logging, torch


class RCNNHead(BBoxHead_v2):
    def __init__(self,
                 in_channels,
                 roi_out_size=7,
                 with_avg_pool=False,
                 fc_channels=[1024, 1024],
                 num_classes=21,
                 target_means=[0.0, 0.0, 0.0, 0.0],
                 target_stds=[0.1, 0.1, 0.2, 0.2],
                 reg_class_agnostic=False,
                 use_sigmoid=False,
                 loss_cls=None,
                 loss_bbox=None):
        super(RCNNHead, self).__init__(
            num_classes,
            use_sigmoid,
            target_means,
            target_stds,
            reg_class_agnostic,
            loss_cls,
            loss_bbox)
        
        self.in_channels=in_channels
        self.roi_out_size=utils.to_pair(roi_out_size)
        self.with_avg_pool=with_avg_pool
        self.fc_channels=fc_channels
        
        self.init_layers()

    def init_layers(self):
        cur_channels=self.in_channels
        cur_spatial_size=self.roi_out_size[0]*self.roi_out_size[1]
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_out_size)
            cur_saptial_size=1
        if not self.fc_channels:
            self.fc_channels=[]
            self.with_shared_fcs=False
        else:
            self.with_shared_fcs=True
            fc_channels=list(self.fc_channels)
            self.fc_channels=fc_channels
            fcs=nn.ModuleList()
            for fc_channel in self.fc_channels:
                fcs.append(nn.Linear(cur_channels * cur_spatial_size, fc_channel))
                fcs.append(nn.ReLU(inplace=True))
                cur_spatial_size=1
                cur_channels=fc_channel
            self.shared_fcs=nn.Sequential(*fcs)

        self.classifier=nn.Linear(cur_channels * cur_spatial_size, self.cls_channels)
        self.regressor=nn.Linear(cur_channels * cur_spatial_size,
                                 4 if self.reg_class_agnostic else self.num_classes * 4)
        logging.info('Constructed RCNNHead with num_classes={}'.format(self.num_classes))

    def init_weights(self):
        if self.with_shared_fcs:
            for fc in self.shared_fcs:
                if isinstance(fc, nn.Linear):
                    init_module_normal(fc, mean=0.0, std=0.01)
        init_module_normal(self.classifier, mean=0.0, std=0.01)
        init_module_normal(self.regressor, mean=0.0, std=0.001)
        logging.info('Initialized weights for RCNNHead')

    # for input, rois is a list of roi output from diff imgs
    # for output, cls_outs and reg_outs are list of results for diff imgs
    # notice we concate inputs and then split outputs to save time(not experimentally compared)
    def forward(self, rois):
        roi_sizes = [roi.shape[0] for roi in rois]
        x = torch.cat(rois, dim=0)
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        if self.with_shared_fcs:
            x = self.shared_fcs(x)
        cls_out = self.classifier(x)
        reg_out = self.regressor(x)
        cls_outs, reg_outs = [], []
        start_idx = 0
        for sz in roi_sizes:
            cls_outs.append(cls_out[start_idx:start_idx+sz])
            reg_outs.append(reg_out[start_idx:start_idx+sz])
            start_idx+=sz
        return cls_outs, reg_outs
