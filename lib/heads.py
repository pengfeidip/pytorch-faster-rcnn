from torch import nn
from .region import AnchorCreator, inside_anchor_mask, ProposalCreator_v2
from .utils import init_module_normal
from . import loss
from . import utils
import logging
import torchvision, torch


class RPNHead(nn.Module):
    def __init__(self,
                 in_channels,
                 feat_channels,
                 anchor_base=16,
                 anchor_scales=[4, 8, 16, 32],
                 anchor_ratios=[0.5, 1.0, 2.0],
                 cls_loss_weight=1.0,
                 bbox_loss_weight=1.0,
                 bbox_loss_beta=1.0/9.0):
        super(RPNHead, self).__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.anchor_base = anchor_base
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.cls_loss_weight = cls_loss_weight
        self.bbox_loss_weight = bbox_loss_weight
        self.bbox_loss_beta = bbox_loss_beta
        self.anchor_creator = AnchorCreator(base=anchor_base,
                                            scales=anchor_scales,
                                            aspect_ratios=anchor_ratios)
        self.num_anchors = len(anchor_scales) * len(anchor_ratios)
        self.conv = nn.Conv2d(in_channels, feat_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Conv2d(feat_channels, self.num_anchors*2, kernel_size=1)
        self.regressor = nn.Conv2d(feat_channels, self.num_anchors*4, kernel_size=1)
        logging.info('Constructed RPNHead with in_channels={}, feat_channels={}'.format(in_channels, feat_channels))

    def init_weights(self):
        init_module_normal(self.conv, mean=0.0, std=0.01)
        init_module_normal(self.classifier, mean=0.0, std=0.01)
        init_module_normal(self.regressor, mean=0.0, std=0.01)
        logging.info('Initialized weights for RPNHead.')
        
    def forward(self, x):
        x = self.relu(self.conv(x))
        return self.classifier(x), self.regressor(x)

    def forward_train(self, feat, gt_bbox, img_size, train_cfg, scale):
        logging.info('In forward_train of RPNHead')
        from .registry import build_module
        device = feat.device
        self.anchor_creator.to(device=device)
        feat_size = feat.shape[-2:]
        cls_out, reg_out = self(feat)
        logging.debug('cls_out.shape: {}'.format(cls_out.shape))
        logging.debug('reg_out.shape: {}'.format(reg_out.shape))
        anchors = self.anchor_creator(img_size, feat_size)
        logging.debug('anchors: {}'.format(anchors.shape))
        anchors = anchors.view(4, -1)
        inside_idx = inside_anchor_mask(anchors, img_size)
        logging.debug('inside_idx: {}'.format(inside_idx.shape))
        in_anchors = anchors[:, inside_idx]
        logging.debug('in_anchors: {}'.format(in_anchors.shape))
        
        # assign and sample anchors to gt bboxes
        assigner = build_module(train_cfg.rpn.assigner)
        sampler = build_module(train_cfg.rpn.sampler)
        labels, overlap_ious = assigner(in_anchors, gt_bbox)
        logging.debug('labels before sample: {}, {}, {}'\
                      .format((labels==-1).sum(), (labels==0).sum(), (labels>0).sum()))
        labels = sampler(labels)
        logging.debug('labels after sample: {}, {}, {}'\
                      .format((labels==-1).sum(), (labels==0).sum(), (labels>0).sum()))

        # labels_ contains only -1, 0, 1
        # labels contains -1, 0 and positive index of gt bboxes
        labels_ = utils.simplify_label(labels)
        labels = labels - 1
        labels[labels<0]=0
        label_bboxes = gt_bbox[:, labels]

        non_neg_label = (labels_!=-1)
        inside_arg=torch.nonzero(inside_idx)
        chosen = inside_arg[non_neg_label].squeeze()
        cls_out_ = cls_out.view(2, -1)
        reg_out_ = reg_out.view(4, -1)
        assert cls_out_.shape[-1] == reg_out_.shape[-1]
        tar_cls_out = cls_out_[:, chosen]  # 128 chosen places from cls_out 
        tar_reg_out = reg_out_[:, chosen]  # 128 chosen places from reg_out
        tar_labels = labels_[non_neg_label] # 128 labels of 1:positive and 0:negative
        tar_anchors = in_anchors[:, non_neg_label] # 128 chosen anchors 
        tar_bbox = label_bboxes[:, non_neg_label] #128 target bbox where anchors should regress to(only those pos anchors)
        tar_param = utils.bbox2param(tar_anchors, tar_bbox) # deltas where tar_reg_out should regress to(only pos)
        logging.debug('labels chosen to train RPNHead: {}, {}'.format((tar_labels==0).sum(), (tar_labels==1).sum()))
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
            
        # next propose bboxes
        props_creator = ProposalCreator_v2(**train_cfg.rpn_proposal)
        props, score = props_creator(cls_out, reg_out, anchors, img_size, scale)
        logging.debug('props by RPNHead: {}'.format(props.shape))

        return \
            cls_loss * self.cls_loss_weight, \
            reg_loss * self.bbox_loss_weight, \
            props

    def forward_test(self, feat, img_size, test_cfg, scale):
        device = feat.device
        self.anchor_creator.to(device)
        feat_size = feat.shape[-2:]
        cls_out, reg_out = self(feat)
        anchors = self.anchor_creator(img_size, feat_size)
        anchors = anchors.view(4, -1)
        props_creator = ProposalCreator_v2(**test_cfg.rpn)
        props, score = props_creator(cls_out, reg_out, anchors, img_size, scale)
        return props, score

class BBoxHead(nn.Module):
    def __init__(self,
                 in_channels,
                 fc_channels=[1024, 1024],
                 roi_out_size = (7, 7),
                 roi_extractor='RoIPool',
                 num_classes=21,
                 target_means=[0.0, 0.0, 0.0, 0.0],
                 target_stds=[0.1, 0.1, 0.2, 0.2],
                 reg_class_agnostic=False,
                 cls_loss_weight=1.0,
                 bbox_loss_weight=1.0,
                 bbox_loss_beta=1.0):
        super(BBoxHead, self).__init__()
        self.in_channels=in_channels
        if isinstance(fc_channels, int):
            fc_channels = tuple([fc_channels, fc_channels])
        else:
            fc_channels = tuple(fc_channels)
        self.fc_channels=fc_channels
        if isinstance(roi_out_size, int):
            roi_out_size = tuple([roi_out_size, roi_out_size])
        else:
            roi_out_size = tuple(roi_out_size)
        self.roi_out_size = roi_out_size

        # build roi extractor
        if roi_extractor == 'RoIPool':
            self.roi_extractor = torchvision.ops.RoIPool(output_size=roi_out_size,
                                                         spatial_scale=1.0/16.0)
        elif roi_extractor == 'RoIAlign':
            self.roi_extractor = torchvision.ops.RoIAlign(output_size=roi_out_size,
                                                          spatial_scale=1.0/16.0,
                                                          sampling_ratio=-1)
        else:
            raise ValueError('Unknown RoI Extractor type: {}'.format(roi_extractor))

        num_fcs = len(fc_channels)
        roi_out_channels = roi_out_size[0] * roi_out_size[1]
        fcs = nn.ModuleList()
        fcs.append(nn.Linear(roi_out_channels*in_channels, fc_channels[0]))
        fcs.append(nn.ReLU(inplace=True))
        for i in range(1, num_fcs):
            fcs.append(nn.Linear(fc_channels[i-1], fc_channels[i]))
            fcs.append(nn.ReLU(inplace=True))
        self.shared_fcs = nn.Sequential(*fcs)

        self.num_classes = num_classes
        self.target_means = target_means
        self.target_stds = target_stds
        self.reg_class_agnostic = reg_class_agnostic
        self.bbox_loss_beta = bbox_loss_beta
        self.bbox_loss_weight = bbox_loss_weight
        self.cls_loss_weight = cls_loss_weight
        
        self.classifier = nn.Linear(fc_channels[-1], num_classes)
        self.regressor = nn.Linear(fc_channels[-1],
                                   4 if reg_class_agnostic else self.num_classes*4)
        logging.info('Constructed BBoxHead with num_classes={}'.format(num_classes))

    def init_weights(self):
        for fc in self.shared_fcs:
            if isinstance(fc, nn.Linear):
                init_module_normal(fc, mean=0.0, std=0.01)
        init_module_normal(self.classifier, mean=0.0, std=0.01)
        init_module_normal(self.regressor, mean=0.0, std=0.001)
        logging.info('Initialized weights for BBoxHead.')

    # assume props has shape (4, n)
    def forward(self, feat, props):
        device = feat.device
        props_t = props.t()
        batch_idx = torch.zeros(props_t.shape[0], 1, device=props_t.device)
        props_t = torch.cat([batch_idx, props_t], dim=1)
        roi_out = self.roi_extractor(feat, props_t)
        num = roi_out.shape[0]
        roi_out = roi_out.view(num, -1)
        fc_out = self.shared_fcs(roi_out)
        cls_out = self.classifier(fc_out)
        reg_out = self.regressor(fc_out)
        return cls_out, reg_out

    # props: proposals proposed by RPNHead in train mode setting
    # train_cfg: only one of the many training cfg for RCNN heads
    def forward_train(self, feat, props_bbox, gt_bbox, gt_label, train_cfg):
        logging.debug('In forward_train of BBoxHead')
        logging.debug('props_bbox.shape: {}'.format(props_bbox.shape))
        logging.debug('gt_bbox: {}'.format(gt_bbox))
        logging.debug('gt_label: {}'.format(gt_label))
        from .registry import build_module
        device = feat.device
        assigner = build_module(train_cfg.assigner)
        sampler = build_module(train_cfg.sampler)
        gt_bbox = gt_bbox.to(props_bbox.dtype)
        props_bbox = torch.cat([gt_bbox, props_bbox], dim=1)

        labels, overlaps_ious = assigner(props_bbox, gt_bbox)
        logging.debug('labels after assigner: -1:{}, 0:{}, >0:{}'\
                      .format((labels==-1).sum(), (labels==0).sum(), (labels>0).sum()))
        labels = sampler(labels)
        logging.debug('labels after sampler: -1:{}, 0:{}, >0:{}'\
                      .format((labels==-1).sum(), (labels==0).sum(), (labels>0).sum()))
        pos_places = (labels > 0)
        neg_places = (labels == 0)
        chosen_places = (labels>=0)
        
        labels = labels - 1
        labels[labels<0] = 0
        label_bboxes = gt_bbox[:, labels]
        label_cls = gt_label[labels]
        # it is very important to set neg places to 0 as 0 means background
        label_cls[neg_places] = 0

        tar_props = props_bbox[:, chosen_places]
        tar_label = label_cls[chosen_places] # class of each gt label, 0 means background
        tar_bbox = label_bboxes[:, chosen_places]
        # calc target param which reg_out regress to
        tar_param = utils.bbox2param(tar_props, tar_bbox)
        # for debug
        logging.debug('mean of tar_param of RCNN: {}'.format(tar_param.mean(dim=1)))
        logging.debug('std  of tar_param of RCNN: {}'.format(tar_param.std(dim=1)))
        param_mean = tar_param.new(self.target_means).view(4, 1)
        param_std  = tar_param.new(self.target_stds).view(4, 1)
        tar_param = (tar_param - param_mean) / param_std # normalize the regression values

        cls_out, reg_out = self(feat, tar_props)
        # next calculate cls_loss and reg_loss
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

        # next find proposals
        return \
            cls_loss * self.cls_loss_weight, \
            reg_loss * self.bbox_loss_weight

    # use RCNN to refine proposals
    def refine_props(self, feat, props, img_size=None):
        with torch.no_grad():
            cls_out, reg_out = self(feat, props)
            soft = torch.softmax(cls_out, dim=1)
            score, label = torch.max(soft, dim=1)
            n_props = cls_out.shape[0]
            n_classes = self.num_classes
            refined = None
            param_mean = reg_out.new(self.target_means).view(-1, 4)
            param_std  = reg_out.new(self.target_stds).view(-1, 4)
            if not self.reg_class_agnostic:
                reg_out = reg_out.view(-1, 4, n_classes)
                reg_out = reg_out[torch.arange(n_props), :, label]
            reg_out = reg_out * param_std + param_mean
            refined = utils.param2bbox(props, reg_out.t())
            if img_size is not None:
                h, w = img_size
                refined = torch.stack([refined[0].clamp(0, w), refined[1].clamp(0, h),
                                       refined[2].clamp(0, w), refined[3].clamp(0, h)])
        return refined, label, score
        

    def forward_test(self, feat, props, img_size=None):
        return self.refine_props(feat, props, img_size)

