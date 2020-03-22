from torch import nn
from .region import AnchorCreator, inside_anchor_mask, ProposalCreator
from .utils import init_module_normal
from . import loss
from . import utils
from . import anchor
import logging
import torchvision, torch
from copy import copy
from mmcv.cnn import normal_init
import numpy as np

class RPNHead(nn.Module):
    def __init__(self,
                 in_channels,
                 feat_channels,
                 anchor_scales=[8],
                 anchor_ratios=[0.5, 1.0, 2.0],
                 anchor_strides=[4, 8, 16, 32, 64],
                 cls_loss_weight=1.0,
                 bbox_loss_weight=1.0,
                 bbox_loss_beta=1.0/9.0):
        super(RPNHead, self).__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_strides = anchor_strides
        self.cls_loss_weight = cls_loss_weight
        self.bbox_loss_weight = bbox_loss_weight
        self.bbox_loss_beta = bbox_loss_beta
        self.base_sizes = tuple(anchor_strides)
        self.anchor_creators = [AnchorCreator(base=base_size,
                                              scales=anchor_scales,
                                              aspect_ratios=anchor_ratios)
                                for base_size in self.base_sizes]
        
        self.num_anchors = len(anchor_scales) * len(anchor_ratios)
        self.conv = nn.Conv2d(in_channels, feat_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Conv2d(feat_channels, self.num_anchors*2, kernel_size=1)
        self.regressor = nn.Conv2d(feat_channels, self.num_anchors*4, kernel_size=1)
        logging.info('Constructed RPNHead with in_channels={}, feat_channels={}, num_levels={}, num_anchors={}'\
                     .format(in_channels, feat_channels, len(self.anchor_creators), self.num_anchors))

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

    def loss(self, tar_cls_out, tar_reg_out, tar_labels, tar_param):
        device=tar_cls_out.device
        cls_loss, reg_loss = loss.zero_loss(device), loss.zero_loss(device)

        # calculate losses
        if tar_labels.numel() != 0:
            ce = nn.CrossEntropyLoss()
            cls_loss = ce(tar_cls_out.t(), tar_labels.long())
            n_samples = len(tar_labels)
            pos_args = (tar_labels==1)
            if pos_args.sum() == 0:
                logging.warning('RPN recieves no positive samples to train.')
            else:
                reg_loss = loss.smooth_l1_loss_v2(tar_reg_out[:, pos_args], tar_param[:, pos_args],
                                                  self.bbox_loss_beta) / n_samples
        else:
            logging.warning('RPN recieves no samples to train, return a dummy zero loss')
        return cls_loss, reg_loss

    def forward_train(self, feats, gt_bbox, img_size, pad_size, train_cfg, scale):
        logging.debug('START of RPNHead forward_train'.center(50, '='))
        from .registry import build_module
        assert len(feats) > 0
        assert len(feats) == len(self.anchor_creators)
        device = feats[0].device
        feat_sizes = [feat.shape[-2:] for feat in feats]
        num_levels = len(self.anchor_creators)
        _ = [ac.to(device=device) for ac in self.anchor_creators]
        
        cls_outs, reg_outs = self(feats)
        logging.debug('cls_out.shape: {}'.format([cls_out.shape for cls_out in cls_outs]))
        logging.debug('reg_out.shape: {}'.format([reg_out.shape for reg_out in reg_outs]))
        cls_outs = [cls_out.view(2, -1) for cls_out in cls_outs]
        reg_outs = [reg_out.view(4, -1) for reg_out in reg_outs]
        cls_out_comb = torch.cat(cls_outs, dim=1)
        reg_out_comb = torch.cat(reg_outs, dim=1)
        
        anchors = [self.anchor_creators[i](pad_size, feat_sizes[i]) for i in range(num_levels)]
        logging.debug('anchors: {}'.format([ac.shape for ac in anchors]))
        anchors = [ac.view(4, -1) for ac in anchors]
        inside_masks = [inside_anchor_mask(ac, pad_size) for ac in anchors]
        logging.debug('inside_masks: {}'.format([iidx.shape for iidx in inside_masks]))
        #in_anchors = [ac[:, iidx] for iidx in inside_idxs ]
        in_anchors = [anchors[i][:, inside_masks[i]] for i in range(num_levels)]
        logging.debug('in_anchors: {}'.format([in_ac.shape for in_ac in in_anchors]))
        # combine anchors from all levels
        in_anchors = torch.cat(in_anchors, dim=1)
        in_mask = torch.cat(inside_masks, dim=0)
        
        logging.debug('inside anchors after cat all levels: {}'.format(in_anchors.shape))
        logging.debug('inside masks after cat all levels: {}'.format(in_mask.shape))

        assigner = build_module(train_cfg.rpn.assigner)
        sampler = build_module(train_cfg.rpn.sampler)
        tar_cls_out, tar_reg_out, tar_labels, tar_anchors, tar_bbox, tar_param \
            = anchor.anchor_target(cls_out_comb, reg_out_comb, in_anchors, in_mask, gt_bbox, assigner, sampler)

        cls_loss, reg_loss = self.loss(tar_cls_out, tar_reg_out, tar_labels, tar_param)

        props_creator = ProposalCreator(**train_cfg.rpn_proposal)
        props, score = [], []
        for i in range(num_levels):
            cur_props, cur_score = props_creator(cls_outs[i],
                                                 reg_outs[i],
                                                 anchors[i],
                                                 img_size,
                                                 scale)
            props.append(cur_props)
            score.append(cur_score)
        props = torch.cat(props, dim=1)
        logging.debug('Proposals by RPNHead: {}'.format(props.shape))
        logging.debug('End of RPNHead forward_train'.center(50, '='))

        return \
            cls_loss * self.cls_loss_weight, \
            reg_loss * self.bbox_loss_weight, \
            props
    
    def forward_test(self, feats, img_size, pad_size, test_cfg, scale):
        assert len(feats) > 0
        device = feats[0].device
        feat_sizes = [feat.shape[-2:] for feat in feats]
        num_levels = len(feats)
        
        _ = [ac.to(device) for ac in self.anchor_creators]
        cls_outs, reg_outs = self(feats)
        cls_outs = [cls_out.view(2, -1) for cls_out in cls_outs]
        reg_outs = [reg_out.view(4, -1) for reg_out in reg_outs]
        
        anchors = [self.anchor_creators[i](pad_size, feat_sizes[i]) for i in range(num_levels)]
        anchors = [ac.view(4, -1) for ac in anchors]

        props_creator = ProposalCreator(**test_cfg.rpn)
        props, score = [], []
        for i in range(num_levels):
            cur_props, cur_score = props_creator(cls_outs[i],
                                                 reg_outs[i],
                                                 anchors[i],
                                                 img_size,
                                                 scale)
            props.append(cur_props)
            score.append(cur_score)
        return torch.cat(props, dim=1), torch.cat(score, dim=0)

    

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
        cls_loss, reg_loss = loss.zero_loss(device), loss.zero_loss(device)
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
                reg_loss = loss.smooth_l1_loss_v2(pos_reg, tar_param[:, pos_arg].t(), self.bbox_loss_beta) / n_samples
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



class RetinaHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs,
                 feat_channels,
                 octave_base_scale=4,
                 scales_per_octave=4,
                 anchor_ratios=[0.5, 1.0, 2.0],
                 anchor_strides=[8, 16, 32, 64, 128],
                 cls_loss_weight=1.0,
                 cls_loss_alpha=0.25,
                 cls_loss_gamma=2.0,
                 bbox_loss_weight=1.0,
                 bbox_loss_beta=1.0/9.0):
        super(RetinaHead, self).__init__()
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
        self.cls_loss_weight = cls_loss_weight
        self.cls_loss_alpha = cls_loss_alpha
        self.cls_loss_gamma = cls_loss_gamma
        self.bbox_loss_weight = bbox_loss_weight
        self.bbox_loss_beta = bbox_loss_beta
        self.base_sizes = tuple(anchor_strides)
        self.anchor_creators = [AnchorCreator(base=base_size,
                                              scales=anchor_scales,
                                              aspect_ratios=anchor_ratios)
                                for base_size in self.base_sizes]
        
        self.num_anchors = len(anchor_scales) * len(anchor_ratios)

        self.init_layers()
        logging.info('Constructed RetinaHead with in_channels={}, feat_channels={}, num_levels={}, num_anchors={}'\
                     .format(in_channels, feat_channels, len(self.anchor_creators), self.num_anchors))

    def init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        cls_convs = []
        reg_convs = []
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            cls_convs.append(
                nn.Conv2d(chn, self.feat_channels, 3, padding=1))
            reg_convs.append(
                nn.Conv2d(chn, self.feat_channels, 3, padding=1))
        self.cls_convs = nn.Sequential(*cls_convs)
        self.reg_convs = nn.Sequential(*reg_convs)
        self.retina_cls = nn.Conv2d(self.feat_channels,
                                    self.num_anchors * (self.cls_channels),
                                    3,
                                    padding=1)
        self.retina_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 3, padding=1)
        print(self)
        # here the classifier and regressor use filter size=3

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m, std=0.01)
        for m in self.reg_convs:
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

    def loss(self, tar_cls_out, tar_reg_out, tar_labels, tar_param):
        device = tar_cls_out.device
        cls_loss, reg_loss = loss.zero_loss(device), loss.zero_loss(device)
        n_pos_samples = (tar_labels>0).sum()
        if tar_labels.numel() != 0:
            cls_loss = loss.focal_loss(tar_cls_out.t(), tar_labels, self.cls_loss_alpha, self.cls_loss_gamma)
            cls_loss = cls_loss / n_pos_samples
            # next calculate regression loss
            tar_labels[tar_labels>0]=1
            pos_args = (tar_labels==1)
            if pos_args.sum() == 0:
                logging.warning('RPN recieves no positive samples to train.')
            else:
                reg_loss = loss.smooth_l1_loss_v2(tar_reg_out[:, pos_args], tar_param[:, pos_args],
                                                  self.bbox_loss_beta) / n_pos_samples
        else:
            logging.warning('RPN recieves no samples to train, return a dummy zero loss')
        return cls_loss, reg_loss
        
        
    def forward_train(self, feats, gt_bbox, gt_label, img_size, pad_size, train_cfg, scale):
        logging.debug('START of RetinaHead forward_train'.center(50, '='))
        logging.debug('img_size={}, pad_size={}'.format(img_size, pad_size))
        from .registry import build_module
        assert len(feats) > 0
        assert len(feats) == len(self.anchor_creators)
        logging.info('features: \n{}'.format('\n'.join([str(x.shape) for x in feats])))
        device = feats[0].device
        feat_sizes = [feat.shape[-2:] for feat in feats]
        num_levels = len(self.anchor_creators)
        _ = [ac.to(device=device) for ac in self.anchor_creators]
        
        cls_outs, reg_outs = self(feats)
        logging.debug('cls_out.shape: \n{}'.format('\n'.join([str(cls_out.shape) for cls_out in cls_outs])))
        logging.debug('reg_out.shape: \n{}'.format('\n'.join([str(reg_out.shape) for reg_out in reg_outs])))
        cls_outs = [cls_out.view(self.cls_channels, -1) for cls_out in cls_outs]
        reg_outs = [reg_out.view(4, -1) for reg_out in reg_outs]
        cls_out_comb = torch.cat(cls_outs, dim=1)
        reg_out_comb = torch.cat(reg_outs, dim=1)
        
        anchors = [self.anchor_creators[i](pad_size, feat_sizes[i]) for i in range(num_levels)]
        logging.debug('anchors: \n{}'.format('\n'.join([str(ac.shape) for ac in anchors])))
        anchors = [ac.view(4, -1) for ac in anchors]
        inside_masks = [inside_anchor_mask(ac, img_size) for ac in anchors]
        logging.debug('inside_masks: \n{}'.format('\n'.join([str(iidx.shape) for iidx in inside_masks])))
        in_anchors = [anchors[i][:, inside_masks[i]] for i in range(num_levels)]
        logging.debug('in_anchors: \n{}'.format('\n'.join([str(in_ac.shape) for in_ac in in_anchors])))
        in_anchors = torch.cat(in_anchors, dim=1)
        in_mask = torch.cat(inside_masks, dim=0)
        
        logging.debug('inside anchors after cat all levels: {}'.format(in_anchors.shape))
        logging.debug('inside masks after cat all levels: {}'.format(in_mask.shape))

        assigner = build_module(train_cfg.assigner)
        sampler = None
        tar_cls_out, tar_reg_out, tar_labels, tar_anchors, tar_bbox, tar_param \
            = anchor.anchor_target_v2(cls_out_comb, reg_out_comb, self.cls_channels, in_anchors, in_mask,
                                      gt_bbox, gt_label, assigner, sampler)
        logging.debug('after anchor_target, tar_cls_out: {}'.format(tar_cls_out.size()))
        logging.debug('after anchor_target, reg_cls_out: {}'.format(tar_reg_out.size()))
        cls_loss, reg_loss = self.loss(tar_cls_out, tar_reg_out, tar_labels, tar_param)
        return \
            cls_loss * self.cls_loss_weight, \
            reg_loss * self.bbox_loss_weight

    
    def forward_test(self, feats, img_size, pad_size, test_cfg, scale):
        logging.debug('In forward_test of RetinaNet, img_size={}, pad_size={}, scale={}'\
                      .format(img_size, pad_size, scale))
        logging.debug('test_cfg: {}'.format(test_cfg))
        assert len(feats) > 0
        device = feats[0].device
        feat_sizes = [feat.shape[-2:] for feat in feats]
        num_levels = len(feats)
        logging.debug('features({}): \n{}'.format(num_levels, '\n'.join([str(feat.shape) for feat in feats])))
        
        _ = [ac.to(device) for ac in self.anchor_creators]
        cls_outs, reg_outs = self(feats)
        cls_outs = [cls_out.view(self.cls_channels, -1) for cls_out in cls_outs]
        reg_outs = [reg_out.view(4, -1) for reg_out in reg_outs]
        logging.debug('cls_outs: \n{}'.format(
            '\n'.join([str(co.shape) for co in cls_outs])))
        logging.debug('reg_outs: \n{}'.format(
            '\n'.join([str(ro.shape) for ro in reg_outs])))

        anchors = [self.anchor_creators[i](pad_size, feat_sizes[i]) for i in range(num_levels)]
        anchors = [ac.view(4, -1) for ac in anchors]
        logging.debug('anchors: \n{}'.format(
            '\n'.join([str(ac.shape) for ac in anchors])))
        
        H, W = img_size
        min_size = scale * test_cfg.min_bbox_size
        logging.info('min_size: {}'.format(min_size))
        cls_scores, cls_labels, pred_bboxes = [], [], []
        for i in range(num_levels):
            logging.debug('predicting in level {}'.format(i))
            cls_out, reg_out = cls_outs[i], reg_outs[i]
            cls_out = cls_out.sigmoid()
            cls_score, cls_label = cls_out.max(0)
            # select top pre_nms candidates based on cls scores
            anchor = anchors[i]
            if test_cfg.pre_nms < len(cls_score):
                _, topk_inds = cls_score.topk(test_cfg.pre_nms)
                cls_score = cls_score[topk_inds]
                cls_label = cls_label[topk_inds]
                reg_out = reg_out[:, topk_inds]
                anchor = anchor[:, topk_inds]
            pred_bbox = utils.param2bbox(anchor, reg_out)
            # filter small bboxes
            pred_bbox = torch.stack([
                torch.clamp(pred_bbox[0], 0.0, W),
                torch.clamp(pred_bbox[1], 0.0, H),
                torch.clamp(pred_bbox[2], 0.0, W),
                torch.clamp(pred_bbox[3], 0.0, H)
            ])
            non_small_bbox = (pred_bbox[2]-pred_bbox[0] >= min_size) \
                             & (pred_bbox[3]-pred_bbox[1] >= min_size)
            cls_score = cls_score[non_small_bbox]
            cls_label = cls_label[non_small_bbox]
            pred_bbox = pred_bbox[:, non_small_bbox]
            cls_scores.append(cls_score)
            cls_labels.append(cls_label)
            pred_bboxes.append(pred_bbox)

        mlvl_cls_score = torch.cat(cls_scores)
        mlvl_cls_label = torch.cat(cls_labels)
        mlvl_pred_bbox = torch.cat(pred_bboxes, dim=1)

        keep_bbox, keep_score, keep_label = utils.multiclass_nms(
            mlvl_pred_bbox, mlvl_cls_score, mlvl_cls_label,
            range(0, self.cls_channels), test_cfg.nms_iou, test_cfg.min_score)

        if len(keep_score) > test_cfg.max_per_img:
            _, topk_inds = keep_score.topk(test_cfg.max_per_img)
            return keep_bbox[:, topk_inds], keep_score[topk_inds], keep_label[topk_inds]
            
        return keep_bbox, keep_score, keep_label
