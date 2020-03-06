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
        logging.debug('START of RPNHead forward_train'.center(50, '='))
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
        logging.debug('labels before sample(-1, 0, >0): {}, {}, {}'\
                      .format((labels==-1).sum(), (labels==0).sum(), (labels>0).sum()))
        labels = sampler(labels)
        logging.debug('labels after sample(-1, 0, >0): {}, {}, {}'\
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
        logging.debug('labels chosen to train RPNHead neg={}, pos={}'\
                      .format((tar_labels==0).sum(), (tar_labels==1).sum()))
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
        logging.debug('Proposals by RPNHead: {}'.format(props.shape))
        logging.debug('End of RPNHead forward_train'.center(50, '='))

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
        
        """
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
        """
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
    def refine_props(self, props, reg_out, is_gt, img_size=None):
        assert props.shape[1] == reg_out.shape[0] == is_gt.numel()
        is_gt = is_gt.to(dtype=torch.bool)
        props = props[:, ~is_gt]
        reg_out = reg_out.t()[:, ~is_gt]
        param_mean = reg_out.new(self.target_means).view(4, -1)
        param_std  = reg_out.new(self.target_stds).view(4, -1)
        reg_out = reg_out * param_std + param_mean
        refined = utils.param2bbox(props, reg_out)
        return refined

    # roi_out: input tensor to BBoxHead
    # props: proposal bboxes
    def forward_test(self, roi_out, props, img_size=None):
        logging.debug('START of BBoxHead forward_test'.center(50, '='))
        logging.debug('received props: {}'.format(props.shape))
        with torch.no_grad():
            cls_out, reg_out = self(roi_out)
            logging.info('cls_out {}'.format(cls_out))
            
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

