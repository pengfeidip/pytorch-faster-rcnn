import torch, torchvision
import torch.nn as nn
import copy, random, logging
import numpy as np
import time, sys, os
import os.path as osp
from . import config, utils

#####################################################
### new implementation using vertorized computing ###
#####################################################

class AnchorCreator(object):

    MAX_CACHE_ANCHOR = 1000
    CACHE_REPORT_PERIOD = 500
    def __init__(self, base=16, scales=[8, 16, 32],
                 aspect_ratios=[0.5, 1.0, 2.0], device=torch.device('cuda:0')):
        self.device = device
        self.base = base
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.cached = {}
        self.count = 0
        anchor_ws, anchor_hs = [], []
        for s in scales:
            for ar in aspect_ratios:
                anchor_ws.append(base * s * np.sqrt(ar))
                anchor_hs.append(base * s / np.sqrt(ar))
        self.anchor_ws = torch.tensor(anchor_ws, device=device, dtype=torch.float32)
        self.anchor_hs = torch.tensor(anchor_hs, device=device, dtype=torch.float32)

    def to(self, device):
        self.device = device
        self.anchor_ws.to(device)
        self.anchor_hs.to(device)

    def report_cache(self):
        count_info = [[k, v[0]] for k,v in self.cached.items()]
        count_info.sort(key=lambda x:x[1], reverse=True)
        top_count = count_info[:10]
        top_str = ', '.join([':'.join([str_id, str(ct)]) for str_id, ct in top_count])
        rep_str = '\n'.join([
            'AnchorCreator count: {}'.format(self.count),
            'Cache size: {}'.format(len(self.cached)),
            'Top 10 used anchor count: {}'.format(top_str)
        ])
        logging.info(rep_str)

    def __call__(self, img_size, grid):
        str_id = '|'.join([
            ','.join([str(x) for x in img_size]),
            ','.join([str(x) for x in grid])
        ])
        # check if the anchor is in the cached
        if str_id in self.cached:
            self.cached[str_id][0] += 1
            return self.cached[str_id][1]
        anchors = self._create_anchors_(img_size, grid)
        if len(self.cached) < self.MAX_CACHE_ANCHOR:
            self.cached[str_id] = [1, anchors]
        self.count += 1
        if self.count % self.CACHE_REPORT_PERIOD == 0:
            self.report_cache()
        return anchors
        
    def _create_anchors_(self, img_size, grid):
        assert len(img_size) == 2 and len(grid) == 2
        imag_h, imag_w = img_size
        grid_h, grid_w = grid
        grid_dist_h, grid_dist_w = imag_h/grid_h, imag_w/grid_w
        
        center_h = torch.linspace(0, imag_h, grid_h+1,
                                  device=self.device, dtype=torch.float32)[:-1] + grid_dist_h/2
        center_w = torch.linspace(0, imag_w, grid_w+1,
                                  device=self.device, dtype=torch.float32)[:-1] + grid_dist_w/2
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

def find_inside_index(anchors, img_size):
    H, W = img_size
    inside = (anchors[0,:]>=0) & (anchors[1,:]>=0) & \
             (anchors[2,:]<=W) & (anchors[3,:]<=H)
    return inside

def random_sample_label(labels, pos_num, tot_num):
    assert pos_num <= tot_num
    pos_args = utils.index_of(labels==1)
    if len(pos_args[0]) > pos_num:
        dis_idx = np.random.choice(
            pos_args[0].cpu().numpy(), size=(len(pos_args[0]) - pos_num), replace=False)
        labels[dis_idx] = -1
    real_n_pos = min(len(pos_args[0]), pos_num)
    n_negs = tot_num - real_n_pos
    neg_args = utils.index_of(labels==0)
    if len(neg_args[0]) > n_negs:
        dis_idx = np.random.choice(
            neg_args[0].cpu().numpy(), size=(len(neg_args[0]) - n_negs), replace=False)
        labels[dis_idx] = -1
    return labels

class AnchorTargetCreator(object):
    def __init__(self, pos_iou=0.7, neg_iou=0.3, max_pos=128, max_targets=256):
        self.pos_iou = pos_iou
        self.neg_iou = neg_iou
        self.max_pos = max_pos
        self.max_targets = max_targets

    def __call__(self, img_size, feat_size, anchors, gt_bbox):
        assert anchors.shape[0] == 4 and gt_bbox.shape[0] == 4
        # TODO: find out why there is a diff btw old and new version
        with torch.no_grad():
            gt_bbox = gt_bbox.to(torch.float32)
            n_anchors, n_gts = anchors.shape[1], gt_bbox.shape[1]
            labels = torch.full((n_anchors,), -1, device=anchors.device, dtype=torch.int)
            iou_tab = utils.calc_iou(anchors, gt_bbox)
            max_anchor_iou, max_anchor_arg = torch.max(iou_tab, dim=0)
            max_gt_iou, max_gt_arg = torch.max(iou_tab, dim=1)
            # first label negative anchors, some of them might be replaced with positive later
            labels[(max_gt_iou < self.neg_iou)] = 0
            # next label positive anchors
            labels[max_anchor_arg] = 1
            labels[(max_gt_iou >= self.pos_iou)] = 1
            labels = random_sample_label(labels, self.max_pos, self.max_targets)
            bbox_labels = gt_bbox[:,max_gt_arg]
            param = utils.bbox2param(anchors, bbox_labels)
        return labels, param, bbox_labels
        

class ProposalCreator(object):
    def __init__(self, max_pre_nms, max_post_nms, nms_iou, min_size):
        self.max_pre_nms = max_pre_nms
        self.max_post_nms = max_post_nms
        self.nms_iou = nms_iou
        self.min_size = min_size

    def __call__(self, rpn_cls_out, rpn_reg_out, anchors, img_size, scale=1.0):
        assert anchors.shape[0] == 4 and len(anchors.shape) == 2
        n_anchors = anchors.shape[1]
        #min_size = scale * self.min_size # this is the value from simple-faster-rcnn
        min_size = 17 # this is the old version value which is basically 1 in feature map
        H, W = img_size
        with torch.no_grad():
            cls_out = rpn_cls_out.view(2, -1)
            reg_out = rpn_reg_out.view(4, -1)
            scores = torch.softmax(cls_out, 0)[1]
            props_bbox = utils.param2bbox(anchors, reg_out)
            props_bbox = torch.stack([
                torch.clamp(props_bbox[0], 0.0, W),
                torch.clamp(props_bbox[1], 0.0, H),
                torch.clamp(props_bbox[2], 0.0, W),
                torch.clamp(props_bbox[3], 0.0, H)
            ])
            small_area_idx = utils.index_of(
                (props_bbox[2] - props_bbox[0]) * (props_bbox[3] - props_bbox[1]) < min_size
            )
            scores[small_area_idx] = -1
            sort_args = torch.argsort(scores, descending=True)
            sort_args = sort_args[sort_args!=-1]
            top_sort_args = sort_args[:self.max_pre_nms]
            
            props_bbox = props_bbox[:, top_sort_args]
            top_scores = scores[top_sort_args]
            keep = torchvision.ops.nms(props_bbox.t(), top_scores, self.nms_iou)

            keep = keep[:self.max_post_nms]
        return props_bbox[:, keep], top_scores[keep]
        

class ProposalTargetCreator(object):
    r"""
    From selected ROIs(around 2000, by ProposalCreator),
    choose 128 samples for training Head.
    """
    def __init__(self,
                 max_pos=32,
                 max_targets=128,
                 pos_iou=0.5,
                 neg_iou_hi=0.5,
                 neg_iou_lo=0.1):
        self.max_pos = max_pos
        self.max_targets = max_targets
        self.pos_iou = pos_iou
        self.neg_iou_hi = neg_iou_hi
        self.neg_iou_lo = neg_iou_lo

    def __call__(self, props_bbox, gt_bbox, gt_label):
        # TODO: this version does not add gt to train classifier in RCNN
        with torch.no_grad():
            gt_bbox = gt_bbox.to(torch.float32)
            n_props, n_gts = props_bbox.shape[1], gt_bbox.shape[1]
            iou_tab = utils.calc_iou(props_bbox, gt_bbox)
            logging.debug('ProposalTargetCreator: max_iou={}, min_iou={}'.format(
                iou_tab.max(), iou_tab.min()))
            max_gt_iou, max_gt_arg = torch.max(iou_tab, dim=1)
            label = torch.full((n_props,), -1, device = props_bbox.device, dtype=torch.int)
            label[max_gt_iou > self.pos_iou] = 1
            label[(max_gt_iou < self.neg_iou_hi) & (max_gt_iou >= self.neg_iou_lo)] = 0
            label = random_sample_label(label, self.max_pos, self.max_targets)
            pos_idx, neg_idx = (label==1), (label==0)
            chosen_idx = pos_idx | neg_idx
            # find class label of each roi, 0 is background
            roi_label = gt_label[max_gt_arg]
            roi_label[neg_idx] = 0
            # find gt bbox for each roi
            roi_gt_bbox = gt_bbox[:,max_gt_arg]
            roi_param = utils.bbox2param(props_bbox, roi_gt_bbox)
        # next only choose rois of non-negative
        return props_bbox[:,chosen_idx], roi_label[chosen_idx], roi_param[:,chosen_idx]


def image2feature(bbox, img_size, feat_size):
    """
    transfer bbox size from image to feature
    """
    h_rat, w_rat = [feat_size[i]/img_size[i] for i in range(2)]
    return bbox * torch.tensor([[w_rat], [h_rat], [w_rat], [h_rat]],
                               device=bbox.device, dtype=torch.float32)
    
    
class ROICropping(object):
    def __init__(self):
        pass

    def __call__(self, feature, props, image_size):
        if props.numel() == 0:
            logging.warning('ROICropping reveives zero proposals')
            return []
        _, n_chanel, h, w = feature.shape
        feat_size = feature.shape[-2:]
        # process of cropping participates in the computation graph
        bbox_feat = image2feature(props, image_size, feat_size).round().int()
        crops = [feature[0, :, y_min:y_max+1, x_min:x_max+1] \
                 for x_min, y_min, x_max, y_max in bbox_feat.t()]
        return crops


class ROIPooling(nn.Module):
    def __init__(self, out_size):
        super(ROIPooling, self).__init__()
        self.out_size = out_size
        self.adaptive_pool = nn.AdaptiveMaxPool2d(out_size)
        
    def forward(self, rois):
        if len(rois) == 0:
            logging.warning('ROIPooling reveives an empty list of rois')
            return None
        return torch.stack([self.adaptive_pool(x) for x in rois])
