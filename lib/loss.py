import torch
from . import region

"""
targets is a list of 256 pairs of anchor and gt_bbox, which is a 
dict of {
  'anchr': anchor,
  'gt_bbox': BBox,
  'gt_label': 0/1,
  'category': category,
  'iou': iou btw gt and anchor
}
where anchor is a dict of keys: 
  'center', 'feat_loc', 'scale_idx', 'ar_idx', 'bbox', 'id'.
"""
def rpn_loss(rpn_cls_res, rpn_reg_res, anchor_generator, targets, lamb):
    cls_out, cls_labels = [], []
    reg_out, reg_params = [], []
    num_scales = len(anchor_generator.scales)
    num_ars = len(anchor_generator.aspect_ratios)
    for tar in targets:
        anchor = tar['anchor']
        gt_bbox = tar['gt_bbox']
        anchor_bbox = anchor['bbox']
        
        feat_i, feat_j = anchor['feat_loc'].y, anchor['feat_loc'].x
        scale_idx, ar_idx = anchor['scale_idx'], anchor['ar_idx']
        anchor_idx = scale_idx * num_ars + ar_idx
        objectness = rpn_cls_res[0, anchor_idx*2:anchor_idx*2 + 2,
                                 feat_i, feat_j]
        adjustment = rpn_reg_res[0, anchor_idx*4:anchor_idx*4 + 4,
                                 feat_i, feat_j]
        gt_label = tar['gt_label']
        cls_out.append(objectness)
        cls_labels.append(gt_label)
        if gt_label != 0:
            gt_params = region.xywh2param(gt_bbox.get_xywh(), anchor_bbox)
            reg_out.append(adjustment)
            reg_params.append(gt_params)
    
    cls_out_tsr = torch.stack(cls_out)
    cls_labels_tsr = torch.tensor(cls_labels)

    ce_loss = torch.nn.CrossEntropyLoss()
    cls_loss = ce_loss(cls_out_tsr, cls_labels_tsr)

    reg_out_tsr = torch.stack(reg_out)
    reg_params_tsr = torch.tensor(reg_params)
    
    sm_loss = torch.nn.SmoothL1Loss()
    reg_loss = sm_loss(reg_out_tsr, reg_params_tsr)
    return cls_loss + lamb * reg_loss

    
