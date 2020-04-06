from torch import nn
from .region import AnchorCreator, inside_anchor_mask, ProposalCreator
from .utils import init_module_normal
from . import losses
from . import utils
from . import anchor
import logging
import torchvision, torch
from copy import copy
from mmcv.cnn import normal_init


class BBoxHead_v2(nn.Module):
    def __init__(self,
                 num_classes,
                 use_sigmoid=False,
                 target_means=[0.0, 0.0, 0.0, 0.0],
                 target_stds=[0.1, 0.1, 0.2, 0.2],
                 reg_class_agnostic=False,
                 loss_cls=None,
                 loss_bbox=None):
        self.num_classes=num_classes
        if use_simgoid:
            self.cls_channels=num_classes-1
        else:
            self.cls_channels=num_classes
        self.target_means=target_means
        self.target_stds=target_stds
        from ..registry import build_module
        self.loss_cls=loss_cls
        if isinstance(loss_cls, dict):
            self.loss_cls=build_module(loss_cls)
        self.loss_bbox=loss_bbox
        if isinstance(loss_bbox, dict):
            self.loss_bbox=build_module(loss_bbox)
        
    def init_layers(self):
        raise NotImplementedError('init_layers is not implemented')
    
    def init_weights(self):
        raise NotImplementedError('init_weights is not implemented')
    
    # returns cls_outs and reg_outs which may be multi-image
    def forward(self):
        raise NotImplementedError('forward is not implemented')

    # targets ought to be calculated separately
    def single_image_targets(self):
        # TODO
        pass

    # simple calc loss from targets and avg_factor
    def calc_loss(self):
        pass

    # forward data, calc targets, and then calc_loss
    def loss(self):
        pass

    # a conveinient chain of things need to do forward train mode
    def forward_train(self):
        pass

    # only refine bbox and ignore gt
    def refine_bbox(self):
        pass
    

class BBoxHead(nn.Module):
    def __init__(self,
                 in_channels,
                 roi_out_size=7,
                 with_avg_pool=False,
                 fc_channels=[1024, 1024],
                 num_classes=21,
                 target_means=[0.0, 0.0, 0.0, 0.0],
                 target_stds=[0.1, 0.1, 0.2, 0.2],
                 reg_class_agnostic=False,
                 cls_loss_weight=1.0,
                 bbox_loss_weight=1.0,
                 bbox_loss_beta=1.0):
        super(BBoxHead, self).__init__()
        self.in_channels=in_channels
        self.roi_out_size=utils.to_pair(roi_out_size)
        cur_channels=in_channels
        cur_spatial_size=self.roi_out_size[0]*self.roi_out_size[1]

        self.with_avg_pool=with_avg_pool
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_out_size)
            cur_spatial_size=1

        if not fc_channels:
            self.fc_channels=[]
            self.with_shared_fcs=False
        else:
            self.with_shared_fcs=True
            fc_channels=list(fc_channels)
            self.fc_channels=fc_channels
            fcs = nn.ModuleList()
            for fc_channel in self.fc_channels:
                fcs.append(nn.Linear(cur_channels*cur_spatial_size, fc_channel))
                fcs.append(nn.ReLU(inplace=True))
                cur_spatial_size=1
                cur_channels=fc_channel
            self.shared_fcs = nn.Sequential(*fcs)
            
        self.num_classes = num_classes
        self.target_means = target_means
        self.target_stds = target_stds
        self.reg_class_agnostic = reg_class_agnostic
        self.bbox_loss_beta = bbox_loss_beta
        self.bbox_loss_weight = bbox_loss_weight
        self.cls_loss_weight = cls_loss_weight

        self.classifier = nn.Linear(cur_channels*cur_spatial_size, num_classes)
        self.regressor = nn.Linear(cur_channels*cur_spatial_size,
                                   4 if reg_class_agnostic else self.num_classes*4)
        logging.info('Constructed BBoxHead with num_classes={}'.format(num_classes))

    def init_layers(self):
        pass


    def init_weights(self):
        if self.with_shared_fcs:
            for fc in self.shared_fcs:
                if isinstance(fc, nn.Linear):
                    init_module_normal(fc, mean=0.0, std=0.01)
        init_module_normal(self.classifier, mean=0.0, std=0.01)
        init_module_normal(self.regressor, mean=0.0, std=0.001)
        logging.info('Initialized weights for BBoxHead.')

    # x is roi out
    def forward(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        if self.with_shared_fcs:
            x = self.shared_fcs(x)
        cls_out = self.classifier(x)
        reg_out = self.regressor(x)
        return cls_out, reg_out


    def loss(self, cls_out, reg_out, tar_label, tar_param):
        logging.debug('Calculating loss of BBoxHead...')
        device=cls_out.device
        cls_loss, reg_loss = losses.zero_loss(device), losses.zero_loss(device)
        if tar_label.numel() != 0:
            tar_label = tar_label.long()
            n_classes = cls_out.shape[1]
            n_samples = len(tar_label)
            ce = nn.CrossEntropyLoss()
            cls_loss = ce(cls_out, tar_label)
            if self.reg_class_agnostic:
                reg_out = reg_out.view(-1, 4)
            else:
                reg_out = reg_out.view(-1, 4, n_classes)
                reg_out = reg_out[torch.arange(n_samples), :, tar_label]
            pos_arg = (tar_label>0)
            if pos_arg.sum() == 0:
                logging.warning('BBoxHead recieves no positive samples to train.')
            else:
                pos_reg = reg_out[pos_arg, :]
                reg_loss = losses.smooth_l1_loss_v2(pos_reg, tar_param[:, pos_arg].t(), self.bbox_loss_beta) / n_samples
        else:
            logging.warning('BBoxHead recieves no samples to train, return dummpy losses')

        logging.debug('END of BBoxHead forward_train'.center(50, '='))

        return \
            cls_loss * self.cls_loss_weight, \
            reg_loss * self.bbox_loss_weight


    # use RCNN to refine proposals
    # needs to filter gt proposals
    def refine_props(self, props, labels, reg_out, is_gt):
        assert props.shape[1] == reg_out.shape[0] == is_gt.numel()
        if not self.reg_class_agnostic:
            reg_out = reg_out.view(-1, 4, n_classes)
            ret_out = reg_out[torch.arange(reg_out.shape[0]), :, label]
        is_gt = is_gt.to(dtype=torch.bool)
        props = props[:, ~is_gt]
        reg_out = reg_out.t()[:, ~is_gt]
        param_mean = reg_out.new(self.target_means).view(4, -1)
        param_std  = reg_out.new(self.target_stds).view(4, -1)
        reg_out = reg_out * param_std + param_mean
        refined = utils.param2bbox(props, reg_out)
        # TODO: do we restrict refined proposals by pad_size?
        return refined

    # roi_out: input tensor to BBoxHead
    # props: proposal bboxes
    def forward_test(self, roi_out, props, img_size=None):
        logging.debug('START of BBoxHead forward_test'.center(50, '='))
        logging.debug('received props: {}'.format(props.shape))
        with torch.no_grad():
            cls_out, reg_out = self(roi_out)
            
            soft = torch.softmax(cls_out, dim=1)
            score, label = torch.max(soft, dim=1)
            n_props = cls_out.shape[0]
            n_classes = self.num_classes
            refined = None
            param_mean = reg_out.new(self.target_means).view(-1, 4)
            param_std  = reg_out.new(self.target_stds).view(-1, 4)
            if self.reg_class_agnostic:
                reg_out = reg_out.view(-1, 4)
            else:
                reg_out = reg_out.view(-1, 4, n_classes)
                reg_out = reg_out[torch.arange(n_props), :, label]
            reg_out = reg_out * param_std + param_mean
            refined = utils.param2bbox(props, reg_out.t())
            if img_size is not None:
                h, w = img_size
                refined = torch.stack([refined[0].clamp(0, w), refined[1].clamp(0, h),
                                       refined[2].clamp(0, w), refined[3].clamp(0, h)])
        logging.debug('END of BBoxHead forward_test'.center(50, '='))
        return refined, label, cls_out

