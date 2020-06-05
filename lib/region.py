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
    It creates anchors based on image size(H, W) and feature size(h, w), which is different than normal implementation.
    
    Args:
        img_size: tuple of (H, W)
        grid: feature size, tupe of (h, w)
    Returns:
        anchors: a tensor of shape (4, num_anchors, h, w)
    '''
    MAX_CACHE_ANCHOR = 10
    CACHE_REPORT_PERIOD = 200
    def __init__(self, base=16, scales=[8, 16, 32],
                 aspect_ratios=[0.5, 1.0, 2.0], device=torch.device('cpu')):
        self.device = device
        self.base = base
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.num_anchors = len(scales)*len(aspect_ratios)
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
        if self.device == device:
            return True
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

# inside grid mask, using img_size, not pad_size
def inside_grid_mask(num_anchors, input_size, img_size, grid_size, device=torch.device('cpu')):
    assert img_size[0] <= input_size[0] and img_size[1] <= input_size[1]
    h_ratio, w_ratio = [grid_size[i]/input_size[i] for i in range(2)]
    assert h_ratio <= 1 and w_ratio <= 1
    in_h = min(grid_size[0], int(img_size[0]*h_ratio)+1)
    in_w = min(grid_size[1], int(img_size[1]*w_ratio)+1)
    flags = torch.zeros((num_anchors, grid_size[0], grid_size[1]), device=device)
    flags[:, :in_h, :in_w] = 1
    return flags.view(-1)

    
def inside_anchor_mask(anchors, img_size, allowed_border=0):
    with torch.no_grad():
        H, W = img_size
        if allowed_border < 0:
            return anchors.new_full((anchors.shape[1], ), True, dtype=torch.bool)
        else:
            return \
                (anchors[0,:] >= -allowed_border) & \
                (anchors[1,:] >= -allowed_border) & \
                (anchors[2,:] <  W + allowed_border) & \
                (anchors[3,:] <  H + allowed_border)

def map_gt2level(scale, strides, gt_bbox):
    '''
    scale: a number
    strides: [4, 8, 16, 32, 64], strides of feature maps
    gt_bbox: [4, n]
    '''
    gt_sides = torch.sqrt((gt_bbox[2] - gt_bbox[1]) * (gt_bbox[3] - gt_bbox[1]))
    min_scale = scale * strides[0]
    gt_sides = gt_sides / min_scale
    return torch.log2(gt_sides).floor().long().clamp(0, len(strides)-1)

        
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
        max_gt_iou: max overlap of each bbox with gt
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
        
    def __call__(self, labels, overlaps_iou=None, props_bbox=None, gt_bbox=None):
        # labels is vector returned by an assigner where
        # -1:ignore, 0:negative, >0:positive
        labels_ = labels.clone().detach()
        labels_[labels>0] = 1
        labels_ = random_sample_label(labels_, self.pos_num, self.max_num)
        pos_places = (labels_==1)
        labels_[pos_places] = labels[pos_places]
        return labels_

class IoUBalancedNegSampler(object):
    def __init__(self, max_num, pos_num, num_bins=3, max_iou=0.5, floor_thr=-1, floor_fraction=0):
        # floor_thr and floor_fraction are not supported
        self.max_num = max_num
        self.pos_num = pos_num
        self.num_bins=num_bins
        self.max_iou=max_iou
        assert max_num >= pos_num
        assert max_iou >0 and max_iou <=1

    def __call__(self, labels, overlaps, props_bbox, gt_bbox):
        # res_labels = labels.clone().detach()
        neg_ious = overlaps[labels==0]
        
        logging.debug('IoUBalancedNegSampler: mean iou of tot neg samples: {}'.format(
            neg_ious.sum()/neg_ious.shape[0] if neg_ious.shape[0]>0 else None))
        pos_places = (labels > 0).nonzero()
        if pos_places.shape[0] > self.pos_num:
            pos_places = utils.random_select(pos_places, self.pos_num)
        num_neg = self.max_num - pos_places.shape[0]

        num_per_bin = int(num_neg/self.num_bins)
        bin_size = self.max_iou / self.num_bins
        starts = [i*bin_size for i in range(self.num_bins)]
        ends = [s+bin_size for s in starts]
        neg_chosen, neg_places = 0, []
        for i, se in enumerate(zip(starts[::-1], ends[::-1])):
            s, e = se
            cur_negs = (labels==0) & (overlaps >= s) & (overlaps < e)
            cur_negs = cur_negs.nonzero()
            cur_allowed = num_per_bin if i < self.num_bins - 1 else num_neg - neg_chosen
            if cur_negs.shape[0] > cur_allowed:
                cur_negs = utils.random_select(cur_negs, cur_allowed)
            neg_chosen += cur_negs.shape[0]
            neg_places.append(cur_negs)
        tot_chosen = torch.cat([pos_places]+neg_places)
        if tot_chosen.shape[0] < self.max_num:
            logging.warning('Sampler can not sample max number of samples, instead: {}'.format(tot_chosen.shape[0]))
        res_labels = labels.clone().detach()
        res_labels[:] = -1
        res_labels[tot_chosen] = labels[tot_chosen]
        neg_ious = overlaps[torch.cat(neg_places)]
        logging.debug('IoUBalancedNegSampler: mean iou of neg samples after sampling: {}'.format(
            neg_ious.sum()/neg_ious.shape[0] if neg_ious.shape[0]>0 else None))
        return res_labels

def ApproxMaxIoUAssigner(object):
    def __init__(self, pos_iou=0.7, neg_iou=0.3, min_pos_iou=0.3):
        pass


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
            props_bbox = utils.param2bbox(anchors, reg_out, img_size=img_size)
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



# provide more flexible roi extractor where users can choose different roi layer for different feature levels
class BasicRoIExtractor(nn.Module):
    '''
    Args:
        feats: list([N, C, H, W]): features of different levels
        rois:  Tensor([4, K]): K rois
    '''

    def __init__(self, roi_layers, output_size=(7, 7), finest_scale=56):
        assert isinstance(roi_layers, list)
        self.output_size = utils.to_pair(output_size)
        self.finest_scale=finest_scale
        for roi_layer in roi_layers:
            roi_layer['output_size'] = output_size
        from .builder import build_module
        self.roi_layers = [build_module(roi_layer) for roi_layer in roi_layers]
        super(BasicRoIExtractor, self).__init__()
        
    def map_rois_to_levels(self, rois, num_lvls):
        # borrow from mmdet
        with torch.no_grad():
            scale = torch.sqrt(
                (rois[2]-rois[0]+1) * (rois[3]-rois[1]+1)
            )
            tar_lvls = torch.floor(torch.log2(scale/self.finest_scale+1e-6))
            tar_lvls = tar_lvls.clamp(0, num_lvls-1).long()
        return tar_lvls

    def _attach_idx_to_rois_(self, rois, idx):
        n_rois = rois.shape[1]
        idx = rois.new_full((1, n_rois), idx)
        return torch.cat([idx, rois], dim=0)

    def forward_single_level(self, feat, rois, i):
        assert feat.dim() in (3, 4)
        if feat.dim() == 3:
            feat = feat.unsqueeze(0)
        rois = self._attach_idx_to_rois_(rois, 0)
        return self.roi_layers[i](feat, rois.t())

    # feats: [1, C, H, W]
    # rois:  [4, K]
    def forward_single_image(self, feats, rois):
        n_lvls = len(self.roi_layers)
        n_rois = rois.shape[1]
        assert n_lvls <= len(feats)
        if n_lvls == 1:
            return self.forward_one_level(feats[0], rois, 0)

        out_channels = feats[0].shape[-3]
        roi_outs = feats[0].new_full((n_rois, out_channels, *self.output_size), 0)
        tar_lvls = self.map_rois_to_levels(rois, n_lvls)
        for i in range(n_lvls):
            feat = feats[i]
            cur_rois_places = (tar_lvls == i)
            cur_rois = rois[:, cur_rois_places]
            cur_roi_out = self.forward_single_level(feat, cur_rois, i)
            roi_outs[cur_rois_places] = cur_roi_out
        return roi_outs

    def forward(self, level_feats, rois_list):
        n_lvls = len(self.roi_layers)
        assert n_lvls > 0 and n_lvls <= len(level_feats)
        n_imgs = len(rois_list)
        feats_list = [[lvl_feat[i] for lvl_feat in level_feats] for i in range(n_imgs)]
        return utils.multi_apply(self.forward_single_image, feats_list, rois_list)
        

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
        assert feat.dim() in (3, 4)
        if feat.dim() == 3:
            feat = feat.unsqueeze(0)
        props_t = props.t()
        batch_idx = torch.zeros(props_t.shape[0], 1, device=props.device)
        props_t = torch.cat([batch_idx, props_t], dim=1)
        if self.roi_layer == 'RoIPool':
            return torchvision.ops.roi_pool(feat, props_t, self.output_size, spatial_scale)
        else:
            return torchvision.ops.roi_align(feat, props_t, self.output_size, spatial_scale, 2)

    # level_feats: feats from all levels, each level may contain multi-image feats
    def forward(self, level_feats, props_list):
        num_levels = len(level_feats)
        num_imgs = len(props_list)

        feats_list = [[lvl_feat[i] for lvl_feat in level_feats] for i in range(num_imgs)]
        return utils.multi_apply(self.forward_single_image, feats_list, props_list)
            
    def forward_single_image(self, feats, props):
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

        out_channels = feats[0].size(-3)
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
