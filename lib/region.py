import torch, torchvision
import torch.nn as nn
import copy, random, logging
import numpy as np
import time, sys, os
import os.path as osp
from . import utils


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
    MAX_CACHE_ANCHOR = 2000
    CACHE_REPORT_PERIOD = 200
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
        with torch.no_grad():
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
    with torch.no_grad():
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
        

class SingleRoIExtractor(nn.Module):
    def __init__(self, roi_layer='RoIPool', output_size=7, featmap_strides=[16], finest_scale=56):
        super(SingleRoIExtractor, self).__init__()
        output_size=utils.to_pair(output_size)
        self.roi_layer=roi_layer
        assert roi_layer in ['RoIPool', 'RoIAlign'], 'Unknown roi_layer type: {}'.format(roi_layer)
        self.output_size=output_size
        self.featmap_strides=featmap_strides
        self.finest_scale=finest_scale
        logging.info('Initialized SingleRoIExtractor with roi_layer={}, output_size={}, featmap_strides={}'\
                     .format(roi_layer, output_size, len(featmap_strides)))


    def map_props_to_levels(self, props, num_lvls):
        # borrow from mmdet
        with torch.no_grad():
            scale = torch.sqrt(
                (props[2]-props[0]+1) * (props[3]-props[1]+1)
            )
            tar_lvls = torch.floor(torch.log2(scale/self.finest_scale+1e-6))
            tar_lvls = tar_lvls.clamp(0, num_lvls-1).long()
        return tar_lvls

    def forward_one_level(self, feat, props, spatial_scale):
        props_t = props.t()
        batch_idx = torch.zeros(props_t.shape[0], 1, device=props.device)
        props_t = torch.cat([batch_idx, props_t], dim=1)
        if self.roi_layer == 'RoIPool':
            return torchvision.ops.roi_pool(feat, props_t, self.output_size, spatial_scale)
        else:
            return torchvision.ops.roi_align(feat, props_t, self.output_size, spatial_scale, 2)
            
    def forward(self, feats, props):
        '''
        Args:
            feat(Tensor): a list of feature maps from a backbone or neck
            props(Tensor(4, n)): proposals
        '''
        assert len(feats) > 0
        assert len(feats) >= len(self.featmap_strides)
        num_lvls = len(self.featmap_strides)
        if num_lvls==1:
            return self.forward_one_level(feats[0], props, 1.0/self.featmap_strides[0])

        out_channels = feats[0].size(1)
        roi_outs = feats[0].new_zeros(
            props.size(1), out_channels, *self.output_size)
        tar_lvls = self.map_props_to_levels(props, num_lvls)
        for i in range(num_lvls):
            feat = feats[i]
            spatial_scale = 1.0/ self.featmap_strides[i]
            cur_props_mask = (tar_lvls==i)
            cur_props = props[:, cur_props_mask]
            cur_roi_out = self.forward_one_level(feat, cur_props, spatial_scale)
            roi_outs[cur_props_mask] = cur_roi_out
        return roi_outs
        
