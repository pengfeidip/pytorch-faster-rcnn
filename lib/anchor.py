from . import utils
import logging
import torch
import numpy as np

# It finds positive, negative and ignored anchors based on assigner and sampler and return only
# positive and negative targets.
# If gt_label is None, the returned tar_labels contains only 1 and 0 where 1 means positive and 0 negative.
# If gt_label is not None, the returned tar_labels contains 0 and positive specific labels of assigned gt.
# Since 0 is reserved as negative(background), gt labels can't contain 0.
def anchor_target(cls_out, reg_out, cls_channels, in_anchors, in_mask, gt_bbox, gt_label=None,
                  assigner=None, sampler=None, target_means=None, target_stds=None):
    '''
    Args:
        cls_out: [1, n], class predict of all anchors
        reg_out: [4, n], bbox predict of all anchors
        in_anchors: [4, m], anchors that are considered, m <= n
        in_mask: [1, n], mask of considered anchors, it has m 0's and (n-m) 1's
    '''
    assert assigner is not None
    from .builder import build_module
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
    neg_places, zero_places, pos_places = (labels<0), (labels==0), (labels>0)
    non_neg_places = (~neg_places)
    num_pos_places = pos_places.sum()
    logging.debug('average overlap iou after  sampling: {} with {} pos anchors'\
                  .format(None if num_pos_places==0 else \
                          overlap_ious[pos_places].sum()/num_pos_places, num_pos_places))

    # labels_ contains only -1, 0, 1
    # labels contains -1, 0 and positive index of gt bboxes
    labels_ = utils.simplify_label(labels)
    labels = labels - 1
    labels[labels<0]=0
    label_bboxes = gt_bbox[:, labels]
    
    inside_arg=torch.nonzero(in_mask)
    chosen = inside_arg[non_neg_places].squeeze()
    cls_out_ = cls_out.view(cls_channels, -1)
    reg_out_ = reg_out.view(4, -1)
    assert cls_out_.shape[-1] == reg_out_.shape[-1]
    # notice tar_cls_out and tar_reg_out has larger sizes than labels, as labels only consider in_anchors
    tar_cls_out = cls_out_[:, chosen]  # 128 chosen places from cls_out
    tar_reg_out = reg_out_[:, chosen]  # 128 chosen places from reg_out
    if gt_label is None:
        # It is an RPN
        tar_labels = labels_[non_neg_places]
    else:
        # It is an anchor head
        label_gt_labels = gt_label[labels]
        label_gt_labels[zero_places] = 0
        tar_labels = label_gt_labels[non_neg_places]

    # tar_labels = labels_[non_neg_label] # 128 labels of 1:positive and 0:negative
    tar_anchors = in_anchors[:, non_neg_places] # 128 chosen anchors 
    tar_bbox = label_bboxes[:, non_neg_places] #128 target bbox where anchors should regress to(only those pos anchors)
    tar_param = utils.bbox2param(tar_anchors, tar_bbox) # deltas where tar_reg_out should regress to(only pos)
    if target_means is not None and target_stds is not None:
        param_mean = tar_param.new(target_means).view(4, 1)
        param_std  = tar_param.new(target_stds).view(4, 1)
        tar_param = (tar_param - param_mean) / param_std
    logging.debug('labels chosen to train network neg={}, pos={}'\
                  .format((tar_labels==0).sum(), (tar_labels>0).sum()))
    return tar_cls_out, tar_reg_out, tar_labels, tar_anchors, tar_bbox, tar_param



class AnchorCreator(object):

    def __init__(self, base=16, scales=[8, 16, 32],
                 aspect_ratios=[0.5, 1.0, 2.0], center_lt=False, device=torch.device('cpu')):

        self.device = device
        self.center_lt = center_lt
        self.base = base
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.num_anchors = len(scales)*len(aspect_ratios)
        anchor_ws, anchor_hs = [], []
        for s in scales:
            for ar in aspect_ratios:
                anchor_ws.append(base * s * np.sqrt(ar))
                anchor_hs.append(base * s / np.sqrt(ar))
        self.anchor_ws = torch.tensor(anchor_ws, device=device, dtype=torch.float32)
        self.anchor_hs = torch.tensor(anchor_hs, device=device, dtype=torch.float32)
        print('AnchorCreator: center_lt={}'.format(self.center_lt))
        
    def to(self, device):
        if self.device == device:
            return True
        self.device = device
        self.anchor_ws = self.anchor_ws.to(device)
        self.anchor_hs = self.anchor_hs.to(device)

    def __call__(self, stride, grid):
        with torch.no_grad():
            grid_h, grid_w = grid
            grid_dist_h, grid_dist_w = stride, stride
        
            center_h = torch.linspace(0, grid_dist_h * grid_h, grid_h+1,
                                      device=self.device, dtype=torch.float32)[:-1]
            if not self.center_lt:
                center_h = center_h + grid_dist_h/2
            center_w = torch.linspace(0, grid_dist_w * grid_w, grid_w+1,
                                      device=self.device, dtype=torch.float32)[:-1]
            if not self.center_lt:
                center_w = center_w + grid_dist_w/2
            mesh_h, mesh_w = torch.meshgrid(center_h, center_w)
            # NOTE that the corresponding is h <-> y and w <-> x
            anchor_hs = self.anchor_hs.view(-1, 1, 1)
            anchor_ws = self.anchor_ws.view(-1, 1, 1)
            x_min = mesh_w - anchor_ws / 2
            x_max = mesh_w + anchor_ws / 2
            y_min = mesh_h - anchor_hs / 2
            y_max = mesh_h + anchor_hs / 2
            anchors = torch.stack([x_min, y_min, x_max, y_max])
        return anchors


