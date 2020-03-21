from . import utils
import logging
import torch


def anchor_target(cls_out, reg_out, in_anchors, in_mask, gt_bbox,
                  assigner, sampler=None, target_means=None, target_stds=None):
    from .registry import build_module
    # assign and sample anchors to gt bboxes
    if isinstance(assigner, dict):
        assigner = build_module(assigner)

    labels, overlap_ious = assigner(in_anchors, gt_bbox)
    logging.debug('labels before sample(-1, 0, >0): {}, {}, {}'\
                  .format((labels==-1).sum(), (labels==0).sum(), (labels>0).sum()))
    if sampler is not None:
        if isinstance(sampler, dict):
            sampler = build_module(sampler)
        labels = sampler(labels)
    logging.debug('labels after sample(-1, 0, >0): {}, {}, {}'\
                  .format((labels==-1).sum(), (labels==0).sum(), (labels>0).sum()))
    pos_places = (labels>0)
    num_pos_places = pos_places.sum()
    logging.debug('average overlap iou after sampling: {} with {} pos anchors'\
                  .format(None if num_pos_places==0 else \
                          overlap_ious[pos_places].sum()/num_pos_places, num_pos_places))

    # labels_ contains only -1, 0, 1
    # labels contains -1, 0 and positive index of gt bboxes
    labels_ = utils.simplify_label(labels)
    labels = labels - 1
    labels[labels<0]=0
    label_bboxes = gt_bbox[:, labels]

    non_neg_label = (labels_!=-1)
    inside_arg=torch.nonzero(in_mask)
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
    if target_means is not None and target_stds is not None:
        param_mean = tar_param.new(target_means).new(4, 1)
        param_std  = tar_param.new(target_stds).new(4, 1)
        tar_param = (tar_param - param_mean) / param_std
    logging.debug('labels chosen to train RPNHead neg={}, pos={}'\
                  .format((tar_labels==0).sum(), (tar_labels==1).sum()))
    return tar_cls_out, tar_reg_out, tar_labels, tar_anchors, tar_bbox, tar_param


# It finds positive, negative and ignored anchors based on assigner and sampler and return only
# positive and negative targets.
# If gt_label is None, the returned tar_labels contains only 1 and 0 where 1 means positive and 0 negative.
# If gt_label is not None, the returned tar_labels contains 0 and positive specific labels of assigned gt.
# Since 0 is reserved as negative(background), gt labels can contain 0.
def anchor_target_v2(cls_out, reg_out, in_anchors, in_mask, gt_bbox, gt_label=None,
                     assigner=None, sampler=None, target_means=None, target_stds=None):
    assert assiner is not None
    from .registry import build_module
    # assign and sample anchors to gt bboxes
    if isinstance(assigner, dict):
        assigner = build_module(assigner)

    labels, overlap_ious = assigner(in_anchors, gt_bbox)
    logging.debug('labels before sample(-1, 0, >0): {}, {}, {}'\
                  .format((labels==-1).sum(), (labels==0).sum(), (labels>0).sum()))
    if sampler is not None:
        if isinstance(sampler, dict):
            sampler = build_module(sampler)
        labels = sampler(labels)
    logging.debug('labels after sample(-1, 0, >0): {}, {}, {}'\
                  .format((labels==-1).sum(), (labels==0).sum(), (labels>0).sum()))
    pos_places = (labels>0)
    num_pos_places = pos_places.sum()
    logging.debug('average overlap iou after sampling: {} with {} pos anchors'\
                  .format(None if num_pos_places==0 else \
                          overlap_ious[pos_places].sum()/num_pos_places, num_pos_places))

    # labels_ contains only -1, 0, 1
    # labels contains -1, 0 and positive index of gt bboxes
    labels_ = utils.simplify_label(labels)
    labels = labels - 1
    labels[labels<0]=0
    label_bboxes = gt_bbox[:, labels]
    
    non_neg_label = (labels_!=-1)
    inside_arg=torch.nonzero(in_mask)
    chosen = inside_arg[non_neg_label].squeeze()
    cls_out_ = cls_out.view(2, -1)
    reg_out_ = reg_out.view(4, -1)
    assert cls_out_.shape[-1] == reg_out_.shape[-1]
    tar_cls_out = cls_out_[:, chosen]  # 128 chosen places from cls_out 
    tar_reg_out = reg_out_[:, chosen]  # 128 chosen places from reg_out
    if gt_label is None:
        # It is an RPN
        tar_labels = labels_[non_neg_label]
    else:
        # It is an anchor head
        label_gt_labels = gt_label[labels]
        tar_labels = label_gt_labels[non_neg_label]

    # tar_labels = labels_[non_neg_label] # 128 labels of 1:positive and 0:negative
    tar_anchors = in_anchors[:, non_neg_label] # 128 chosen anchors 
    tar_bbox = label_bboxes[:, non_neg_label] #128 target bbox where anchors should regress to(only those pos anchors)
    tar_param = utils.bbox2param(tar_anchors, tar_bbox) # deltas where tar_reg_out should regress to(only pos)
    if target_means is not None and target_stds is not None:
        param_mean = tar_param.new(target_means).new(4, 1)
        param_std  = tar_param.new(target_stds).new(4, 1)
        tar_param = (tar_param - param_mean) / param_std
    logging.debug('labels chosen to train RPNHead neg={}, pos={}'\
                  .format((tar_labels==0).sum(), (tar_labels==1).sum()))
    return tar_cls_out, tar_reg_out, tar_labels, tar_anchors, tar_bbox, tar_param