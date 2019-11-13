from . import region, modules, loss
import torch
import torch.nn as nn

FASTER_ANCHOR_SCALES = [128, 256, 512]
FASTER_ANCHOR_ASPECT_RATIOS = [1.0, 0.5, 2.0]
FASTER_ROI_POOL_SIZE = (7, 7)

class FasterRCNNModule(nn.Module):
    r"""
    It consists of a Backbone, a RPN and a RCNN.
    It contains region related utilities to generate anchors, region proposals, rois etc.
    It is a reguler nn.Module.
    """
    def __init__(self,
                 num_classes=20,
                 anchor_scales=FASTER_ANCHOR_SCALES,
                 anchor_aspect_ratios=FASTER_ANCHOR_ASPECT_RATIOS,
                 anchor_pos_iou=0.7,
                 anchor_neg_iou=0.3,
                 anchor_max_pos=128,
                 anchor_max_targets=256,
                 train_props_pre_nms=12000,
                 train_props_post_nms=2000,
                 train_props_nms_iou=0.7,
                 test_props_pre_nms=6000,
                 test_props_post_nms=300,
                 test_props_nms_iou=0.7,
                 props_pos_iou=0.5,
                 props_neg_iou=0.1,
                 props_max_pos=32,
                 props_max_targets=128,
                 roi_pool_size=FASTER_ROI_POOL_SIZE,
                 transfer_rcnn_fc=True
    ):
        super(FasterRCNNModule, self).__init__()
        self.num_classes=num_classes
        self.anchor_scales=anchor_scales,
        self.anchor_aspect_ratios=anchor_aspect_ratios
        self.anchor_pos_iou=anchor_pos_iou
        self.anchor_neg_iou=anchor_neg_iou
        self.anchor_max_pos=anchor_max_pos
        self.anchor_max_targets=anchor_max_targets
        self.train_props_pre_nms=train_props_pre_nms
        self.train_props_post_nms=train_props_post_nms
        self.train_props_nms_iou=train_props_nms_iou
        self.test_props_pre_nms=test_props_pre_nms
        self.test_props_post_nms=test_props_post_nms
        self.test_props_nms_iou=test_props_nms_iou
        self.props_pos_iou=props_pos_iou
        self.props_neg_iou=props_neg_iou
        self.props_max_pos=props_max_pos
        self.props_max_targets=props_max_targets
        self.roi_pool_size=roi_pool_size

        # next init region related utilities
        self.anchor_gen = region.AnchorGenerator(anchor_scales, anchor_aspect_ratios)
        self.anchor_target_gen = region.AnchorTargetCreator(self.anchor_gen)
        self.train_props_gen = region.ProposalCreator(
            self.anchor_gen,
            self.train_props_pre_nms,
            self.train_props_post_nms,
            self.train_props_nms_iou)
        self.test_props_gen = region.ProposalCreator(
            self.anchor_gen,
            self.test_props_pre_nms,
            self.test_props_post_nms,
            self.test_props_nms_iou)
        self.props_target_gen = region.ProposalTargetCreator()
        self.roi_crop = region.ROICropping()
        self.roi_pool = region.ROIPooling(output_size=roi_pool_size)
        # next init networks
        self.backbone = modules.VGGBackbone()
        self.rpn = modules.RPN(num_classes=num_classes,
                               num_anchors=len(anchor_scales)*len(anchor_aspect_ratios))
        vgg16 = self.backbone.vgg16[0]
        fc1_state_dict=vgg16.classifier[0].state_dict() if transfer_rcnn_fc else None
        fc2_state_dict=vgg16.classifier[3].state_dict() if transfer_rcnn_fc else None
        self.rcnn = modules.RCNN(num_classes,
                                 fc1_state_dict,
                                 fc2_state_dict)

        self.training = True

    # It assumes that x only contains one image,
    # i.e. it only supports train/test one image at a time.
    # X.shape may look like (1, 3, 600, 1000), the size is already resized to
    # longer=1000 and shorter=600
    # the purpose of ground truth 'gt' is to generate rcnn output for training mode
    #
    # Pipeline for training:
    #
    #                         gt-----
    #                               |
    # feature->rpn_out->props(train)-->props_targets-->roi_crops->roi_pool->rcnn_out
    #       |                                       ^
    #       |                                       |
    #       -----------------------------------------
    #
    # Pipeline for testing:
    # feature->rpn_out->props(test)-->roi_crops->roi_pool->rcnn_out
    #       |                      ^
    #       |                      |
    #       ------------------------
    
    def forward(self, x, gt):
        img_size = x.shape[-2:]
        feat = self.backbone(x)
        feat_size = feat.shape[-2:]

        rpn_cls_out, rpn_reg_out = self.rpn(feat)
        anchor_targets = None
        if self.training:
            anchor_targets = self.anchor_target_gen.targets(img_size, feat_size, gt)

        if self.training:
            props = self.train_props_gen.proposals_filtered(
                rpn_cls_out, rpn_reg_out, img_size, feat_size)
        else:
            props = self.test_props_gen.proposals_filtered(
                rpn_cls_out, rpn_reg_out, img_size, feat_size)
                
        props_targets = None
        if self.training:
            props_targets = self.props_target_gen.targets(props, gt)

        if self.training:
            roi_crops, props_targets = self.roi_crop.crop(img_size, feat, props_targets)
            roi_pool_out = self.roi_pool(roi_crops)
            rcnn_cls_out, rcnn_reg_out = self.rcnn(roi_pool_out)
        else:
            roi_crops, props = self.roi_crop.crop(img_size, feat, props)
            roi_pool_out = self.roi_pool(roi_crops)
            rcnn_cls_out, rcnn_reg_out = self.rcnn(roi_pool_out)
        # anchor_targets, props_targets will be None for test mode
        return \
            rpn_cls_out, rpn_reg_out, rcnn_cls_out, rcnn_reg_out,\
            anchor_targets, props, props_targets


    def train(self):
        self.training = True
    def eval(self):
        self.training = False
