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
class RPNLoss(object):
    def __init__(self, anchor_generator, lamb):
        self.anchor_generator = anchor_generator
        self.lamb = lamb
    def __call__(self, rpn_cls_out, rpn_reg_out, targets):
        device = rpn_cls_out.device
        cls_out, cls_labels = [], []
        reg_out, reg_params = [], []
        num_scales = len(self.anchor_generator.scales)
        num_ars = len(self.anchor_generator.aspect_ratios)
        for tar in targets:
            anchor = tar['anchor']
            gt_bbox = tar['gt_bbox']
            anchor_bbox = anchor['bbox']
            
            feat_i, feat_j = anchor['feat_loc'].y, anchor['feat_loc'].x
            scale_idx, ar_idx = anchor['scale_idx'], anchor['ar_idx']
            anchor_idx = scale_idx * num_ars + ar_idx
            objectness = rpn_cls_out[0, anchor_idx*2:anchor_idx*2 + 2,
                                     feat_i, feat_j]
            adjustment = rpn_reg_out[0, anchor_idx*4:anchor_idx*4 + 4,
                                     feat_i, feat_j]
            gt_label = tar['gt_label']
            cls_out.append(objectness)
            cls_labels.append(gt_label)
            if gt_label != 0:
                gt_params = region.xywh2param(gt_bbox.get_xywh(), anchor_bbox)
                reg_out.append(adjustment)
                reg_params.append(gt_params)
                
        cls_loss, reg_loss = torch.tensor(0, device=device), torch.tensor(0, device=device)
        cls_out_tsr = torch.stack(cls_out)
        cls_labels_tsr = torch.tensor(cls_labels, device=device)
        ce_loss = torch.nn.CrossEntropyLoss()
        if cls_out_tsr.numel() !=0 and cls_labels_tsr.numel() != 0:
            cls_loss = ce_loss(cls_out_tsr, cls_labels_tsr)
        reg_out_tsr = torch.stack(reg_out)
        reg_params_tsr = torch.tensor(reg_params, device=device)
        sm_loss = torch.nn.SmoothL1Loss()
        if reg_out_tsr.numel() != 0 and reg_params_tsr.numel() != 0:
            reg_loss = sm_loss(reg_out_tsr, reg_params_tsr)
        return cls_loss + self.lamb * reg_loss
        
        

class RCNNLoss(object):
    def __init__(self, lamb):
        self.lamb = lamb
    def __call__(self, cls_out, reg_out, props_targets):
        device = cls_out.device
        cls_loss, reg_loss = torch.tensor(0), torch.tensor(0)
        ce_loss = torch.nn.CrossEntropyLoss()
        labels = torch.tensor([tar['category'] for tar in props_targets], device=device)
        if cls_out.numel() != 0 and labels.numel() != 0:
            cls_loss = ce_loss(cls_out, labels)

        smL1_loss = torch.nn.SmoothL1Loss()
        pos_reg_out = []
        gt_params = []
        for i, tar in enumerate(props_targets):
            if tar['gt_bbox'] is None:
                continue
            adj_bbox = tar['adj_bbox']
            category = tar['category']
            gt_bbox = tar['gt_bbox']
            pos_reg_out.append(reg_out[i][category*4:(category+1)*4])
            gt_params.append(region.xywh2param(gt_bbox.get_xywh(), adj_bbox))
            
        if len(gt_params) != 0:
            pos_reg_out_tsr = torch.stack(pos_reg_out)
            gt_params_tsr = torch.tensor(gt_params, device=device)
            if pos_reg_out_tsr.numel() !=0 and gt_params_tsr.numel() != 0:
                reg_loss = smL1_loss(pos_reg_out_tsr, gt_params_tsr)
        return cls_loss + self.lamb * reg_loss


### delete the following later ###
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

    cls_loss, reg_loss = torch.tensor(0), torch.tensor(0)
    cls_out_tsr = torch.stack(cls_out)
    cls_labels_tsr = torch.tensor(cls_labels)
    ce_loss = torch.nn.CrossEntropyLoss()
    if cls_out_tsr.numel() !=0 and cls_labels_tsr.numel() != 0:
        cls_loss = ce_loss(cls_out_tsr, cls_labels_tsr)
    reg_out_tsr = torch.stack(reg_out)
    reg_params_tsr = torch.tensor(reg_params)
    sm_loss = torch.nn.SmoothL1Loss()
    if reg_out_tsr.numel() != 0 and reg_params_tsr.numel() != 0:
        reg_loss = sm_loss(reg_out_tsr, reg_params_tsr)
    return cls_loss + lamb * reg_loss
        

def head_loss(cls_out, reg_out, adj_bboxes, gt_bboxes, category_labels, lamb):
    cls_loss, reg_loss = torch.tensor(0), torch.tensor(0)
    ce_loss = torch.nn.CrossEntropyLoss()
    labels = torch.tensor(category_labels)
    if cls_out.numel() != 0 and labels.numel() != 0:
        cls_loss = ce_loss(cls_out, labels)

    smL1_loss = torch.nn.SmoothL1Loss()
    pos_reg_out = []
    gt_params = []
    for i, gt_bbox in enumerate(gt_bboxes):
        if gt_bbox is None:
            continue
        adj_bbox = adj_bboxes[i]
        category = category_labels[i]
        pos_reg_out.append(reg_out[i][category*4:(category+1)*4])
        gt_params.append(region.xywh2param(gt_bbox.get_xywh(), adj_bbox))
    if len(gt_params) != 0:
        pos_reg_out_tsr = torch.stack(pos_reg_out)
        gt_params_tsr = torch.tensor(gt_params)
        if pos_reg_out_tsr.numel() !=0 and gt_params_tsr.numel() != 0:
            reg_loss = smL1_loss(pos_reg_out_tsr, gt_params_tsr)
    return cls_loss + lamb * reg_loss
