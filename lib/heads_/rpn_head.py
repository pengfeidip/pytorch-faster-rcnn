from torch import nn
from ..utils import init_module_normal
from mmcv.cnn import normal_init
from .anchor_head import AnchorHead
import logging, torch
import torchvision.ops as tvops
from ..utils import class_name
from .. import utils

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
            loss_cls=loss_cls,
            loss_bbox=loss_bbox)
        
        self.init_layers()

    def init_layers(self):
        self.conv = nn.Conv2d(
            self.in_channels, self.feat_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Conv2d(
            self.feat_channels, self.num_anchors*self.cls_channels, kernel_size=1)
        self.regressor = nn.Conv2d(self.feat_channels, self.num_anchors*4, kernel_size=1)
        logging.info('Constructed RPNHead with in_channels={}, feat_channels={}, num_levels={}, num_anchors={}'\
                     .format(self.in_channels, self.feat_channels, len(self.anchor_creators), self.num_anchors))
        
    def init_weights(self):
        init_module_normal(self.conv, mean=0.0, std=0.01)
        init_module_normal(self.classifier, mean=0.0, std=0.01)
        init_module_normal(self.regressor, mean=0.0, std=0.01)
        logging.info('Initialized weights for RPNHead.')
        
    def forward(self, xs):
        conv_outs = [self.relu(self.conv(x)) for x in xs]
        cls_outs = [self.classifier(x) for x in conv_outs]
        reg_outs = [self.regressor(x) for x in conv_outs]
        return cls_outs, reg_outs

    def predict_single_image(self, level_cls_outs, level_reg_outs, level_anchors, img_meta, test_cfg):
        logging.info(' {}: predict one image '.format(class_name(self)).center(50, '*'))
        cls_outs = [lvl_cls_out.view(self.cls_channels, -1) for lvl_cls_out in level_cls_outs]
        reg_outs = [lvl_reg_out.view(4, -1) for lvl_reg_out in level_reg_outs]
        anchors = [anchor.view(4, -1) for anchor in level_anchors]
        num_levels = len(level_cls_outs)
        device = level_cls_outs[0].device
        img_size = img_meta['img_shape'][:2]
        H, W = img_size
        min_size = img_meta['scale_factor'] * test_cfg.min_bbox_size
        cls_scores, cls_labels, pred_bboxes = [], [], []

        for i in range(num_levels):
            cls_out, reg_out = cls_outs[i], reg_outs[i]
            anchor = anchors[i]
            if self.use_sigmoid:
                cls_sig = cls_out.sigmoid()
                cls_score = cls_sig[0]
            else:
                cls_soft = cls_out.softmax(dim=0)
                cls_score = cls_soft[1]
            
            if test_cfg.pre_nms > 0 and test_cfg.pre_nms < len(cls_score):
                _, topk_inds = cls_score.topk(test_cfg.pre_nms)
                cls_score = cls_score[topk_inds]
                reg_out = reg_out[:, topk_inds]
                anchor  = anchor[:, topk_inds]
            pred_bbox = utils.param2bbox(anchor, reg_out)
            pred_bbox = torch.stack([
                torch.clamp(pred_bbox[0], 0.0, W),
                torch.clamp(pred_bbox[1], 0.0, H),
                torch.clamp(pred_bbox[2], 0.0, W),
                torch.clamp(pred_bbox[3], 0.0, H)
            ])
            if min_size > 0:
                non_small = (pred_bbox[2]-pred_bbox[0] + 1 >= min_size) \
                            & (pred_bbox[3]-pred_bbox[1] + 1 >= min_size)
                cls_score = cls_score[non_small]
                pred_bbox = pred_bbox[:, non_small]
                
            keep = tvops.nms(pred_bbox.t(), cls_score, test_cfg.nms_iou)
            cls_score = cls_score[keep]
            pred_bbox = pred_bbox[:, keep]
            if test_cfg.post_nms > 0 and test_cfg.post_nms < len(cls_score):
                cls_score = cls_score[:test_cfg.post_nms]
                pred_bbox = pred_bbox[:, :test_cfg.post_nms]
            cls_scores.append(cls_score)
            pred_bboxes.append(pred_bbox)
        mlvl_cls_score = torch.cat(cls_scores)
        mlvl_pred_bbox = torch.cat(pred_bboxes, dim=1)
        max_num = test_cfg.max_num
        if max_num > 0 and len(mlvl_cls_score) > max_num:
            _, topk_inds = mlvl_cls_score.topk(max_num)
            mlvl_cls_score = mlvl_cls_score[topk_inds]
            mlvl_pred_bbox = mlvl_pred_bbox[:, topk_inds]
                                                                                
        return mlvl_pred_bbox, mlvl_cls_score, None
