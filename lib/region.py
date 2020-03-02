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
    '''
    It creates anchors based on image size(H, W) and feature size(h, w).
    
    Args:
        img_size: tuple of (H, W)
        grid: feature size, tupe of (h, w)
    Returns:
        anchors: a tensor of shapw (4, num_anchors, h, w)
    '''
    MAX_CACHE_ANCHOR = 1000
    CACHE_REPORT_PERIOD = 100
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
        self.anchor_ws = self.anchor_ws.to(device)
        self.anchor_hs = self.anchor_hs.to(device)

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

def inside_anchor_mask(anchors, img_size):
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
    '''
    It assigns gt bboxes to anchors based on some rules.
    Args:
        anchors: tensor of shape (4, n), n is number of anchors
        gt_bbox: tensor of shape (4, m), m is number of gt bboxes
    Returns:
        labels: consists of 1=positive anchor, 0=negative anchor, -1=ignore
        bbox_labels: gt bboxes assigned to each anchor
    '''
    def __init__(self, pos_iou=0.7, neg_iou=0.3, min_pos_iou=0.3, max_pos=128, max_targets=256):
        self.pos_iou = pos_iou
        self.neg_iou = neg_iou
        self.max_pos = max_pos
        self.min_pos_iou = min_pos_iou
        self.max_targets = max_targets
        self.assigner = MaxIoUAssigner(pos_iou=pos_iou, neg_iou=neg_iou, min_pos_iou=min_pos_iou)
        self.sampler = RandomSampler(max_num=max_targets, pos_num=max_pos)

    def __call__(self, anchors, gt_bbox):
        labels, overlap_ious = self.assigner(anchors, gt_bbox)
        labels = self.sampler(labels)
        
        labels_ = labels.clone().detach()
        labels_[labels_>0] = 1

        labels = labels - 1
        labels[labels<0] = 0
        label_bboxes = gt_bbox[:,labels]
        return labels_, label_bboxes, overlap_ious
    

class AnchorTargetCreator_v2(object):
    def __init__(self, assigner, sampler):
        from .builder import build_module
        self.assigner = build_module(assigner)
        self.sampler = build_module(sampler)

    def __call__(self, anchors, gt_bbox):
        labels, overlap_ious = self.assigner(anchors, gt_bbox)
        labels = self.sampler(labels)
        
        labels_ = labels.clone().detach()
        labels_[labels_>0] = 1

        labels = labels - 1
        labels[labels<0] = 0
        label_bboxes = gt_bbox[:,labels]
        return labels_, label_bboxes, overlap_ious

    
class MaxIoUAssigner(object):
    '''
    It assigns gt bboxes to anchors based on some rules.
    Args:
        anchors: tensor of shape (4, n), n is number of anchors
        gt_bbox: tensor of shape (4, m), m is number of gt bboxes
    Returns:
        labels: consists of >0:positive anchor, 0:negative anchor, -1:ignore
        bbox_labels: gt bboxes assigned to each anchor
    '''
    def __init__(self, pos_iou, neg_iou, min_pos_iou):
        self.pos_iou = pos_iou
        self.neg_iou = neg_iou
        self.min_pos_iou = min_pos_iou

    def __call__(self, bboxes, gt_bboxes):
        assert bboxes.shape[0] == 4 and gt_bboxes.shape[0] == 4
        num_gts = gt_bboxes.shape[-1]
        with torch.no_grad():
            gt_bbox = gt_bboxes.to(torch.float32)
            n_bboxes, n_gts = bboxes.shape[1], gt_bboxes.shape[1]
            # first label everything as -1(ignore)
            labels_ = torch.full((n_bboxes,), -1, device=bboxes.device, dtype=torch.long)
            # calculate iou table, it has shape [num_anchors, num_gt_bboxes]
            iou_tab = utils.calc_iou(bboxes, gt_bboxes)
            # for each gt, find the anchor with max iou overlap with it
            max_bbox_iou, max_bbox_arg = torch.max(iou_tab, dim=0)
            # for each anchor, find the gt with max iou overlap with it
            max_gt_iou, max_gt_arg = torch.max(iou_tab, dim=1)
            # first label negative bboxes, some of them might be replaced with positive later
            labels_[(max_gt_iou < self.neg_iou)] = 0
            # next to label positive bboxes
            labels_[(max_gt_iou >= self.pos_iou)] = 1

            # find all the bbox with the same max iou overlap with a gt, but the max iou must be >= min_pos_iou
            equal_max_bbox = (iou_tab == max_bbox_iou) & (max_bbox_iou >= self.min_pos_iou)
            # find the bbox index
            _, max_equal_arg = torch.max(equal_max_bbox, dim=1)
            # find where the equal happens
            equal_places = (equal_max_bbox.sum(1) > 0)
            # update max_gt_arg and max_gt_iou
            max_gt_arg[equal_places] = max_equal_arg[equal_places]
            max_gt_iou = iou_tab[torch.arange(n_bboxes), max_gt_arg]
            # update labels_
            labels_[equal_places] = 1
            labels = labels_.clone().detach()
            labels[labels_==1] = (max_gt_arg+1)[labels_==1]
        return labels, max_gt_iou

class RandomSampler(object):
    def __init__(self, max_num, pos_num):
        assert pos_num <= max_num
        self.max_num = max_num
        self.pos_num = pos_num
        
    def __call__(self, labels):
        # labels is vector returned by an assigner where
        # -1:ignore, 0:negative, >0:positive
        labels_ = labels.clone().detach()
        labels_[labels>0] = 1
        labels_ = random_sample_label(labels_, self.pos_num, self.max_num)
        pos_places = (labels_==1)
        labels_[pos_places] = labels[pos_places]
        return labels_


class ProposalCreator(object):
    '''
    It propose regions that potentially contain objects.
    Args:
        rpn_cls_out: output of the classifer of RPN
        rpn_reg_out: output of the regressor of RPN
        anchors: (4, n) where n is number of anchors
    Returns:
        props_bbox: tensor of shape (4, n)
        top_scores: objectness score
    '''
    def __init__(self, max_pre_nms, max_post_nms, nms_iou, min_size):
        self.max_pre_nms = max_pre_nms
        self.max_post_nms = max_post_nms
        self.nms_iou = nms_iou
        self.min_size = min_size

    def __call__(self, rpn_cls_out, rpn_reg_out, anchors, img_size, scale=1.0):
        assert anchors.shape[0] == 4 and len(anchors.shape) == 2
        n_anchors = anchors.shape[1]
        min_size = scale * self.min_size # this is the value from simple-faster-rcnn
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
            small_area_mask = (props_bbox[2] - props_bbox[0] < min_size) \
                              | (props_bbox[3] - props_bbox[1] < min_size)
            num_small_area = small_area_mask.sum()
            scores[small_area_mask] = -1
            sort_args = torch.argsort(scores, descending=True)
            if num_small_area > 0:
                sort_args = sort_args[:-num_small_area]
            top_sort_args = sort_args[:self.max_pre_nms]
            
            props_bbox = props_bbox[:, top_sort_args]
            top_scores = scores[top_sort_args]
            keep = torchvision.ops.nms(props_bbox.t(), top_scores, self.nms_iou)
            keep = keep[:self.max_post_nms]
        return props_bbox[:, keep], top_scores[keep]

class ProposalCreator_v2(object):
    def __init__(self,
                 pre_nms,
                 post_nms,
                 nms_iou,
                 min_size):
        self.pre_nms = pre_nms
        self.post_nms = post_nms
        self.nms_iou = nms_iou
        self.min_size = min_size

    def __call__(self, rpn_cls_out, rpn_reg_out, anchors, img_size, scale=1.0):
        assert anchors.shape[0] == 4 and len(anchors.shape) == 2
        n_anchors = anchors.shape[1]
        min_size = scale * self.min_size # this is the value from simple-faster-rcnn
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
            small_area_mask = (props_bbox[2] - props_bbox[0] < min_size) \
                              | (props_bbox[3] - props_bbox[1] < min_size)
            num_small_area = small_area_mask.sum()
            scores[small_area_mask] = -1
            sort_args = torch.argsort(scores, descending=True)
            if num_small_area > 0:
                sort_args = sort_args[:-num_small_area]
            top_sort_args = sort_args[:self.pre_nms]
            
            props_bbox = props_bbox[:, top_sort_args]
            top_scores = scores[top_sort_args]
            keep = torchvision.ops.nms(props_bbox.t(), top_scores, self.nms_iou)
            keep = keep[:self.post_nms]
        return props_bbox[:, keep], top_scores[keep]
        
        

class ProposalTargetCreator_(object):
    """
    Choose regions to train RCNN.
    Args:
        props_bbox: region proposals with shape (4, n) where n=number of regions
        gt_bbox: gt bboxes with shape (4, m) where m=number of gt bboxes
        gt_label: gt lables with shape (4,) where m=number of labels
    Returns:
        props_bbox: chosen bbox
        roi_label: class labels of each chosen roi
        roi_gt_bbox: gt assigned to each props_bbox
    """
    def __init__(self,
                 max_pos=32,
                 max_targets=128,
                 pos_iou=0.5,
                 neg_iou_hi=0.5,
                 neg_iou_lo=0.0):
        self.max_pos = max_pos
        self.max_targets = max_targets
        self.pos_iou = pos_iou
        self.neg_iou_hi = neg_iou_hi
        self.neg_iou_lo = neg_iou_lo
        
    def __call__(self, props_bbox, gt_bbox, gt_label):
        with torch.no_grad():
            gt_bbox = gt_bbox.to(props_bbox.dtype)
            # add gt to train RCNN
            props_bbox = torch.cat([gt_bbox, props_bbox], dim=1)
            n_props, n_gts = props_bbox.shape[1], gt_bbox.shape[1]
            iou_tab = utils.calc_iou(props_bbox, gt_bbox)
            max_gt_iou, max_gt_arg = torch.max(iou_tab, dim=1)
            label = torch.full((n_props,), -1, device = props_bbox.device, dtype=torch.int)
            label[max_gt_iou >= self.pos_iou] = 1
            label[(max_gt_iou < self.neg_iou_hi) & (max_gt_iou >= self.neg_iou_lo)] = 0
            label = random_sample_label(label, self.max_pos, self.max_targets)
            pos_idx, neg_idx = (label==1), (label==0)
            chosen_idx = pos_idx | neg_idx
            # just for logging purpose
            chosen_iou = max_gt_iou[chosen_idx]
            logging.debug('ProposalTargetCreator: max_iou={}, min_iou={}'.format(
                chosen_iou.max(), chosen_iou.min()))
            # find class label of each roi, 0 is background
            roi_label = gt_label[max_gt_arg]
            roi_label[neg_idx] = 0
            # find gt bbox for each roi
            roi_gt_bbox = gt_bbox[:,max_gt_arg]
        # next only choose rois of non-negative
        return props_bbox[:,chosen_idx], roi_label[chosen_idx], roi_gt_bbox[:, chosen_idx]

# using MaxIoUAssigner and RandomSampler
class ProposalTargetCreator(object):
    """
    Choose regions to train RCNN.
    Args:
        props_bbox: region proposals with shape (4, n) where n=number of regions
        gt_bbox: gt bboxes with shape (4, m) where m=number of gt bboxes
        gt_label: gt lables with shape (4,) where m=number of labels
    Returns:
        props_bbox: chosen bbox
        label_cls: class labels of each chosen roi
        label_bbox: gt assigned to each props_bbox
    """
    def __init__(self,
                 max_pos=32,
                 max_targets=128,
                 pos_iou=0.5,
                 neg_iou=0.5,
                 min_pos_iou=0.5):
        self.max_pos = max_pos
        self.max_targets = max_targets
        self.pos_iou = pos_iou
        self.neg_iou = neg_iou
        self.min_pos_iou = min_pos_iou
        self.assigner = MaxIoUAssigner(pos_iou=pos_iou, neg_iou=neg_iou, min_pos_iou=min_pos_iou)
        self.sampler = RandomSampler(max_num=max_targets, pos_num=max_pos)
        
    def __call__(self, props_bbox, gt_bbox, gt_label):
        gt_bbox = gt_bbox.to(props_bbox.dtype)
        props_bbox = torch.cat([gt_bbox, props_bbox], dim=1)

        
        labels, overlaps_ious = self.assigner(props_bbox, gt_bbox)
        labels = self.sampler(labels)
        pos_places = (labels > 0)
        neg_places = (labels == 0)
        chosen_places = (labels>=0)

        n_props_bbox = props_bbox.shape[1]
        n_gts = gt_label.numel()
        is_gt = labels.new_zeros(n_props_bbox)
        is_gt[:n_gts]=1
        
        labels = labels - 1
        labels[labels<0] = 0
        label_bboxes = gt_bbox[:, labels]
        label_cls = gt_label[labels]
        # it is very important to set neg places to 0 as 0 means background
        label_cls[neg_places] = 0
        is_gt_chosen = is_gt[chosen]
        logging.debug('Chosen gt: {}, number of gt: {}'.format(is_gt_chosen.sum(), n_gts))
        return props_bbox[:, chosen_places], label_cls[chosen_places], \
            label_bboxes[:, chosen_places], is_gt_chosen

    
def image2feature(bbox, img_size, feat_size):
    """
    transfer bbox size from image to feature
    """
    h_rat, w_rat = [feat_size[i]/img_size[i] for i in range(2)]
    return bbox * torch.tensor([[w_rat], [h_rat], [w_rat], [h_rat]],
                               device=bbox.device, dtype=torch.float32)
    
class ROICropping(object):
    '''
    It crops feature based on proposals. First proposals are resized to feature size.
    Args:
        feature: feature map from a backbone
        props: a batch of bboxes with shape (4, n)
        image_size: (h, w)
    '''
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
    '''
    Map tensors of different sizes to a fixed sized tensor.
    '''
    def __init__(self, out_size):
        super(ROIPooling, self).__init__()
        self.out_size = out_size
        self.adaptive_pool = nn.AdaptiveMaxPool2d(out_size)
        
    def forward(self, rois):
        if len(rois) == 0:
            logging.warning('ROIPooling receives an empty list of rois')
            return None
        zero_area, pos_area = [], []
        for roi in rois:
            if roi.numel()==0:
                zero_area.append(roi)
            else:
                pos_area.append(roi)
        if len(zero_area)>0:
            logging.warning('Encounter'
                            ' {} rois with 0 area, ignore this batch!'.format(len(zero_area)))
            return None
        if len(pos_area)==0:
            logging.warning('No rois with positive area, ignore this batch!')
            return None
        return torch.stack([self.adaptive_pool(x) for x in pos_area])
