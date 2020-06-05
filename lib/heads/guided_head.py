import torch.nn as nn
import numpy as np
from mmdet.ops.dcn import DeformConv
from mmcv.cnn import normal_init
import logging, torch
import torchvision.ops as tvops
from .. import debug, utils, region, losses
from ..utils import multi_apply, unpack_multi_result
from ..anchor import anchor_target


# not used
def map_gt_level(anchor_scale, anchor_strides, gt_bbox):
    raise ValueError('Old version of map gt to levels')
    # borrow from mmdet
    scales = torch.sqrt((gt_bbox[2] - gt_bbox[0] + 1) * (gt_bbox[3] - gt_bbox[1] + 1))
    min_anchor_size = scales.new_full((1, ), float(anchor_scale * anchor_strides[0]))
    target_lvls = torch.floor(torch.log2(scales) - torch.log2(min_anchor_size) + 0.5)
    target_lvls = target_lvls.clamp(0, len(anchor_strides)-1).long()
    return target_lvls

# TODO:do we optimize area by adjusting +1 after ending index or by taking floor and start and +1 at end?
def paint_value(canvas, bbox, scale, val):
    assert canvas.dim() == 2
    bbox = bbox * scale
    bbox = bbox.round().long()
    canvas[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1] = val
    return canvas

def paint_bbox(canvas, bbox, scale, bbox_val):
    # canvas: [4, 300, 200]
    # bbox: [x_min, y_min, x_max, y_max]
    # scale: bbox will be scaled to this
    # bbox_val: [x_min, y_min, x_max, y_max] the value to paint to canvas
    assert canvas.dim() == 3
    bbox = bbox * scale
    bbox = bbox.round().long()
    canvas[:, bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1] = bbox_val.view(4, 1, 1)
    return canvas

# it is actually multi-level guided anchor
class GuidedAnchor(nn.Module):
    def __init__(self,
                 in_channels=256,
                 out_channels=256,
                 anchor_scales=None,
                 anchor_ratios=None,
                 anchor_strides=None, 
                 anchoring_means=[0.0, 0.0, 0.0, 0.0],
                 anchoring_stds=[0.07, 0.07, 0.14, 0.14],
                 deformable_groups=4,
                 sigma=8,
                 loc_filter_thr=0.01,
                 loss_loc=None,
                 loss_shape=None):
        super(GuidedAnchor, self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.anchor_scales=anchor_scales
        self.anchor_ratios=anchor_ratios
        self.anchor_strides=anchor_strides
        self.anchoring_means=anchoring_means
        self.anchoring_stds=anchoring_stds
        self.deformable_groups=deformable_groups
        self.sigma=sigma
        self.loc_filter_thr=loc_filter_thr
        
        from ..builder import build_module
        self.loss_loc=build_module(loss_loc)
        self.loss_shape=build_module(loss_shape)

        self.square_anchor_creators = [
            region.AnchorCreator(base=base,
                                 scales=[anchor_scales[0]],
                                 aspect_ratios=[1.0])
            for base in self.anchor_strides
        ]

        self.approx_anchor_creators = [
            region.AnchorCreator(base=base,
                                 scales=anchor_scales,
                                 aspect_ratios=anchor_ratios)
            for base in self.anchor_strides]
        
        self.init_layers()

    # finished
    def init_layers(self):
        self.loc_layer = nn.Conv2d(self.in_channels, 1, 1)
        self.shape_layer = nn.Conv2d(self.in_channels, 2, 1)

        offset_channels = 3 * 3 * 2 * self.deformable_groups
        self.offset_layer = nn.Conv2d(2, offset_channels, 1, bias=False)
        # Do we turn bias on?
        self.adapt_layer = DeformConv(self.in_channels, self.out_channels, 3, padding=1,
                                      deformable_groups=self.deformable_groups)
        self.relu=nn.ReLU(inplace=True)
        logging.info('Init layers of GuidedAnchor')

    # finished
    def init_weights(self):
        normal_init(self.offset_layer, std=0.1)
        normal_init(self.adapt_layer, std=0.01)
        
        normal_init(self.shape_layer, std=0.01)
        prior_prob = 0.01
        bias_init = float(-np.log((1-prior_prob)/prior_prob))
        normal_init(self.loc_layer, std=0.01, bias=bias_init)  # borrow from mmdet?
        logging.info('Init weights of GuidedAnchor')
        
    # finished        
    # loc_outs:   [[1, 200, 300], [1, 100, 150], ...]
    # shape_outs: [[2, 200, 300], [2, 100, 150], ...]
    def create_rpn_anchors_single_image(self, loc_outs, shape_outs, square_anchors, img_meta, input_size):
        loc_outs = [x.sigmoid() for x in loc_outs]
        device = loc_outs[0].device
        num_lvls = len(loc_outs)
        grid_sizes = [lo.shape[-2:] for lo in loc_outs]
        img_size = img_meta['img_shape'][:2]
        in_grid_masks = [region.inside_grid_mask(1, input_size, img_size, grid_size, device)
                         for grid_size in grid_sizes]
        in_grid_masks = [in_grid_masks[i].view(gsz).bool() for i, gsz in enumerate(grid_sizes)]
        in_masks = [(lo>self.loc_filter_thr) & in_grid_masks[i] for i, lo in enumerate(loc_outs)]
        anchors = []
        for i in range(num_lvls):
            wh = shape_outs[i]
            xy = torch.full_like(wh, 0)
            param = torch.cat([xy, wh], dim=0)
            anchor = utils.param2bbox(square_anchors[i].view(4, -1), param.view(4, -1),
                                      self.anchoring_means, self.anchoring_stds)
            anchors.append(anchor)
        return anchors, in_masks
    
    # finished
    # create final anchors for RPN prediction
    def create_rpn_anchors(self, loc_outs, shape_outs, img_metas):
        # infer anchor from loc and shape predictions
        device = loc_outs[0].device
        input_size = utils.input_size(img_metas)
        input_size = tuple(input_size)
        grid_sizes = [lo.shape[-2:] for lo in loc_outs]
        square_anchors = self.create_square_anchors(input_size, grid_sizes, device)
        square_anchors = tuple(square_anchors)
        num_imgs = len(img_metas)
        loc_outs_img, shape_outs_img = [], []
        for i in range(num_imgs):
            loc_outs_img.append([loc[i] for loc in loc_outs])
            shape_outs_img.append([sro[i] for sro in shape_outs])
        return unpack_multi_result(multi_apply(
            self.create_rpn_anchors_single_image,
            loc_outs_img,
            shape_outs_img,
            square_anchors,
            img_metas,
            input_size))

    # finished
    def create_square_anchors(self, in_size, grid_sizes, device):
        for ac in self.square_anchor_creators:
            ac.to(device)
        return [actr(in_size, grid_sizes[i])
                for i, actr in enumerate(self.square_anchor_creators)]

    # finished
    def create_approx_anchors(self, in_size, grid_sizes, device):
        for ac in self.approx_anchor_creators:
            ac.to(device)
        return [actr(in_size, grid_sizes[i])
                for i, actr in enumerate(self.approx_anchor_creators)]
        
    # finished
    def shape_target(self, shape_outs, gt_bboxes, gt_labels, img_metas, cfg):
        device = shape_outs[0].device
        input_size = utils.input_size(img_metas)
        input_size = tuple(input_size)
        grid_sizes = [so.shape[-2:] for so in shape_outs]
        approx_anchors = self.create_approx_anchors(input_size, grid_sizes, device)
        approx_anchors = tuple(approx_anchors)
        square_anchors = self.create_square_anchors(input_size, grid_sizes, device)
        square_anchors = tuple(square_anchors)
        num_imgs = len(img_metas)
        shape_outs_img = []
        for i in range(num_imgs):
            shape_outs_img.append([shape[i] for shape in shape_outs])
        return unpack_multi_result(multi_apply(self.shape_target_single_image_v2,
                                               square_anchors,
                                               approx_anchors,
                                               shape_outs_img,
                                               gt_bboxes,
                                               gt_labels,
                                               img_metas,
                                               input_size,
                                               cfg))

    def shape_target_single_image_v2(self, square_anchors, approx_anchors, shape_outs,
                                     gt_bbox, gt_label, img_meta, input_size, cfg):
        # return tar_shape_out, tar_bbox, tar_sqr_anchors, tar_label
        logging.debug('IN shape_target_single_image_v2'.center(30, '*'))
        device = shape_outs[0].device
        gt_lvl = region.map_gt2level(self.anchor_scales[0], self.anchor_strides, gt_bbox)
        logging.debug('mapped gt to levels: {}'.format(gt_lvl.tolist()))
        ctr_ratio = cfg.get('shape_center_ratio', 0.5)
        scales = [1.0/x for x in self.anchor_strides]
        tar_masks = [so.new_full(so.shape[-2:], 0, dtype=torch.long, device=device) \
                     for so in shape_outs]
        tar_bboxes = [so.new_full([4] + list(so.shape[-2:]), 0, dtype=torch.float, device=device) \
                     for so in shape_outs]

        debug_tar_places = []
        for i in range(gt_bbox.shape[1]):
            lvl = gt_lvl[i]
            bbox = gt_bbox[:, i]
            x1, y1, x2, y2 = bbox.float()
            w, h = x2-x1+1, y2-y1+1
            w, h = w*ctr_ratio, h*ctr_ratio
            ctr_x, ctr_y = (x2+x1)/2, (y2+y1)/2
            ctr_bbox = torch.tensor([ctr_x-w/2, ctr_y-h/2, ctr_x+w/2, ctr_y+h/2], dtype=torch.float)
            paint_value(tar_masks[lvl], ctr_bbox, scales[lvl], 1)
            debug_tar_places.append(tar_masks[lvl].sum().item())
            # TODO: do we do the following instead?
            # tar_bboxes[lvl][:, tar_masks[lvl]] = bbox
            # TODO: what order should we paint bbox values to tar_bboxes?
            # Stride from low to hight or inverse?
            paint_bbox(tar_bboxes[lvl], ctr_bbox, scales[lvl], bbox)
        logging.debug('target places for each gt: {}'.format(debug_tar_places))
        # tar_bboxes: [[4, 200, 300], [4, 100, 150], ...]
        # tar_masks:  [[200, 300], [100, 150], ...]
        tar_masks  = [tm.view(-1) for tm in tar_masks]
        tar_bboxes = [tb.view(4, -1) for tb in tar_bboxes]
        tar_mask = torch.cat(tar_masks)
        tar_bbox = torch.cat(tar_bboxes, dim=1)

        # if we do targets in this way, assigner is not needed
        from ..builder import build_module
        sampler = cfg.ga_sampler
        if sampler is not None:
            sampler = build_module(sampler)
            labels = sampler(tar_mask)

        non_neg_places = (labels>=0)
        tar_bbox = tar_bbox[:, non_neg_places]
        tar_label = labels[non_neg_places]
        sqr_anchors = [sa.view(4, -1) for sa in square_anchors]
        sqr_anchor = torch.cat(sqr_anchors, dim=1)
        tar_sqr_anchor = sqr_anchor[:, non_neg_places]
        shape_outs = [so.view(2, -1) for so in shape_outs]
        shape_out = torch.cat(shape_outs, dim=1)
        tar_shape_out = shape_out[:, non_neg_places]
        logging.debug('tar_labels: {}'.format(utils.count_tensor(tar_label)))
        
        return tar_shape_out, tar_bbox, tar_sqr_anchor, tar_label

    
    # finished
    def shape_target_single_image(self, square_anchors, approx_anchors, shape_outs,
                                  gt_bbox, gt_label, img_meta, input_size, cfg):
        '''
        Args:
            approx_anchors: [Tensor[4, 9, 200, 301], Tensor[4, 9, 100, 151], ...]
            square_anchors: [Tensor[4, 1, 200, 301], Tensor[4, 1, 100, 151], ...]
            shape_outs: [Tensor[2, 200, 301], Tensor[2, 100, 151], ...]
                        the first 2 channels are for w and h
        '''
        from ..builder import build_module
        # here shape_outs are shape reformed outs
        device = shape_outs[0].device
        grid_sizes = [so.shape[-2:] for so in shape_outs]
        img_size = img_meta['img_shape'][:2]

        assert cfg.allowed_border == -1 # assert this to make it a little easier
        num_anchors = approx_anchors[0].shape[1]
        # only flatten spatial dim, i.e. w, h
        anchors = [acs.view(4, num_anchors, -1) for acs in approx_anchors]
        anchors = torch.cat(anchors, dim=-1) # [4, 9, 80297]
        sqr_anchors = [sacs.view(4, -1) for sacs in square_anchors]
        sqr_anchors = torch.cat(sqr_anchors, dim=-1)
        
        # get grid that is within current image area, it is needed for multi-image mode
        in_masks = [region.inside_grid_mask(1, input_size, img_size, grid_size, device)
                    for grid_size in grid_sizes]
        in_mask = torch.cat(in_masks)

        # now we want to find the anchor(out of 9 anchors) that overlap gt
        # the most for all grid position
        max_ious = []
        for i in range(num_anchors):
            cur_anchor = anchors[:, i, :]
            iou_tab = utils.calc_iou(gt_bbox, cur_anchor)
            max_iou, max_gt_idx = iou_tab.max(dim=0)
            max_ious.append(max_iou)
        max_ious = torch.stack(max_ious) # [9, 80297]
        # find out which anchor overlaps the most with gt
        max_iou_with_gt, max_anchor_with_gt = max_ious.max(0)
        max_anchors = anchors[:, max_anchor_with_gt, torch.arange(anchors.shape[-1])]

        assigner = build_module(cfg.ga_assigner)
        in_anchors = max_anchors[:, in_mask.bool()]
        labels, overlap_ious = assigner(in_anchors, gt_bbox)
        sampler = cfg.ga_sampler
        if sampler is not None:
            if isinstance(sampler, dict):
                sampler = build_module(sampler)
            labels = sampler(labels)
            
        neg_places, zero_places, pos_places = (labels<0), (labels==0), (labels>0)
        non_neg_places = (~neg_places)

        # select target shape outs before in_mask
        shape_outs = [so.view(2, -1) for so in shape_outs]
        shape_out = torch.cat(shape_outs, dim=1)
        inside_args = torch.nonzero(in_mask)
        chosen = inside_args[non_neg_places].squeeze()
        tar_shape_out = shape_out[:, chosen]
        tar_sqr_anchors = sqr_anchors[:, chosen]

        # select target bbox and target label(1 and 0) after in_mask
        tar_idx = labels - 1
        tar_idx[tar_idx < 0] = 0
        tar_bbox = gt_bbox[:, tar_idx]
        tar_bbox = tar_bbox[:, non_neg_places]
        tar_label = labels.detach().clone()
        tar_label[pos_places] = 1
        tar_label = tar_label[non_neg_places]
        return tar_shape_out, tar_bbox, tar_sqr_anchors, tar_label


    # finished
    def loc_target(self, loc_outs, gt_bboxes, gt_labels, img_metas, cfg):
        num_imgs = len(gt_bboxes)
        input_size = tuple(utils.input_size(img_metas)) # make it not splitable for multi_apply
        loc_outs_img = []
        for i in range(num_imgs):
            loc_outs_img.append([loc_out[i] for loc_out in loc_outs])
        return unpack_multi_result(multi_apply(self.loc_target_single_image,
                                               loc_outs_img,
                                               gt_bboxes,
                                               gt_labels,
                                               img_metas,
                                               input_size,
                                               cfg))
    # finished
    def loc_target_single_image(self, loc_outs, gt_bbox, gt_label, img_meta, input_size, cfg):
        '''
        Args:
            loc_outs: list([1, 267, 200], [1, 134, 100], ...)
            gt_bbox: [4, n], n gt bboxes
            gt_label: [n], n gt labels
            img_meta: 'img_shape', 'scale_factor' etc
            cfg: 'sampler', 'assigner', 'center_ratio', 'ignore_ratio', etc
        '''
        loc_outs = [lo.squeeze() for lo in loc_outs]
        device = loc_outs[0].device
        scales = [1.0 / x for x in self.anchor_strides]
        num_lvls = len(loc_outs)
        img_size = img_meta['img_shape'][:2]
        grid_sizes = [lo.shape[-2:] for lo in loc_outs]

        # first set all targets to -1 which is ignore, this mainly targets out-of-grid locations
        targets = [lo.new_full(lo.shape[-2:], -1, dtype=torch.long) for lo in loc_outs]
        
        # then set all in grid areas to 0 which is negative
        in_masks = [region.inside_grid_mask(1, input_size, img_size, grid_size, device)
                    for grid_size in grid_sizes]
        for tar_map, in_mask in zip(targets, in_masks):
            in_mask = in_mask.view(tar_map.size()).bool()
            tar_map[in_mask.bool()] = 0

        gt_lvl = region.map_gt2level(self.anchor_scales[0], self.anchor_strides, gt_bbox)
        logging.debug('mapped gt in levels: {}'.format(gt_lvl))
        # second set all ignore areas
        ig_thr = cfg.ignore_ratio
        for i in range(gt_bbox.shape[1]):
            lvl = gt_lvl[i]
            bbox = gt_bbox[:, i]
            x1, y1, x2, y2 = bbox.float()
            w, h = x2 - x1 + 1, y2 - y1 + 1
            w, h = w*cfg.ignore_ratio, h*cfg.ignore_ratio
            ctr_x, ctr_y = (x2 + x1) / 2, (y2 + y1) / 2
            ig_bbox = [ctr_x - w/2, ctr_y - h/2, ctr_x + w/2, ctr_y + h/2]
            ig_bbox = torch.tensor(ig_bbox, dtype=torch.float)
            paint_value(targets[lvl], ig_bbox, scales[lvl], -1)
            if lvl - 1 >= 0:
                paint_value(targets[lvl-1], ig_bbox, scales[lvl-1], -1)
            if lvl + 1 <  num_lvls:
                paint_value(targets[lvl+1], ig_bbox, scales[lvl+1], -1)
        # third, set all positive areas
        for i in range(gt_bbox.shape[1]):
            lvl = gt_lvl[i]
            bbox = gt_bbox[:, i]
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1 + 1, y2 - y1 + 1
            w, h = w*cfg.center_ratio, h*cfg.center_ratio
            ctr_x, ctr_y = (x2 + x1) / 2, (y2 + y1) / 2
            ctr_bbox = [ctr_x - w/2, ctr_y - h/2, ctr_x + w/2, ctr_y + h/2]
            ctr_bbox = torch.tensor(ctr_bbox, dtype=torch.float)
            paint_value(targets[lvl], ctr_bbox, scales[lvl], 1)
        return targets, loc_outs
    
    # finished
    def forward(self, feats):
        assert len(feats) == len(self.anchor_strides)
        loc_outs = [self.loc_layer(feat) for feat in feats]
        shape_outs = [self.shape_layer(feat) for feat in feats]
        offsets = [self.offset_layer(so.detach()) for so in shape_outs]
        adapt_feats = [self.relu(self.adapt_layer(feats[i], offsets[i])) for i in range(len(feats))]
        return loc_outs, shape_outs, adapt_feats

    # finished
    def loss(self, loc_outs, shape_outs, 
             adapt_feats, gt_bboxes, gt_labels, img_metas, cfg):
        print('IN LOSS')
        device = loc_outs[0].device
        logging.debug('START: Loc Target'.center(50, '='))
        tar_locs, tar_loc_outs = self.loc_target(loc_outs, gt_bboxes, gt_labels, img_metas, cfg)
        logging.debug('END: Loc Target'.center(50, '='))
        debug.tensor_shape(tar_locs, 'tar_locs')
        debug.tensor_shape(tar_loc_outs, 'tar_loc_outs')
        
        logging.debug('START: Shape target'.center(50, '='))
        tar_shape_outs, tar_bboxes, tar_sqr_anchors, tar_labels = self.shape_target(
            shape_outs, gt_bboxes, gt_labels, img_metas, cfg)
        logging.debug('END: Shape target'.center(50, '='))
        debug.tensor_shape(tar_shape_outs, 'tar_shape_outs')
        debug.tensor_shape(tar_labels, 'tar_labels')

        # first calc loc loss
        num_imgs = len(img_metas)
        for i in range(num_imgs):
            tar_locs[i] = [x.view(-1) for x in tar_locs[i]]
            tar_locs[i] = torch.cat(tar_locs[i])
            tar_loc_outs[i] = [x.view(-1) for x in tar_loc_outs[i]]
            tar_loc_outs[i] = torch.cat(tar_loc_outs[i])
        tar_locs = torch.cat(tar_locs)
        logging.debug('Chosen loc targets: {}'.format(utils.count_tensor(tar_locs)))
        tar_loc_outs = torch.cat(tar_loc_outs)

        non_neg_places = (tar_locs >= 0)
        loc_loss = self.loss_loc(tar_loc_outs[non_neg_places].unsqueeze(1), tar_locs[non_neg_places])
        loc_loss = loc_loss / (tar_locs==1).sum()

        # next calc shape loss
        tar_labels = torch.cat(tar_labels)
        pos_places = (tar_labels==1)
        print('num of pos places', pos_places.sum())
        tar_shape_outs = torch.cat(tar_shape_outs, dim=1)
        tar_bboxes = torch.cat(tar_bboxes, dim=1)
        tar_sqr_anchors = torch.cat(tar_sqr_anchors, dim=1)
        tar_params = utils.bbox2param(tar_sqr_anchors, tar_bboxes,
                                     self.anchoring_means, self.anchoring_stds)
        pos_tar_params = tar_params[:, pos_places]
        print('pos_tar_params values:')
        print(pos_tar_params.t())

        logging.debug('pos shape tar params.sum(1): {}'.format(pos_tar_params.sum(1)))
        exit()



        avg_factor = pos_places.sum().item()
        shape_loss = losses.zero_loss(device)

        if avg_factor > 0:
            shape_loss = self.loss_shape(tar_shape_outs[0][pos_places], tar_params[2][pos_places]) + \
                self.loss_shape(tar_shape_outs[1][pos_places], tar_params[3][pos_places])
            shape_loss = shape_loss / avg_factor
        ga_loss = {'loc_loss': loc_loss, 'shape_loss': shape_loss}
        return ga_loss



class GARPNHead(nn.Module):
    def __init__(self,
                 in_channels=256,
                 feat_channels=256,
                 octave_base_scale=8,
                 scales_per_octave=3,
                 octave_ratios=[0.5, 1.0, 2.0],
                 anchor_strides=[4, 8, 16, 32, 64],
                 anchoring_means=[0.0, 0.0, 0.0, 0.0],
                 anchoring_stds=[0.07, 0.07, 0.14, 0.14],
                 target_means=[0.0, 0.0, 0.0, 0.0],
                 target_stds=[0.07, 0.07, 0.11, 0.11],
                 deformable_groups=4,
                 loc_filter_thr=0.01,
                 loss_loc=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_shape=dict(type='BoundedIoULoss', beta=0.2, loss_weight=1.0),
                 loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                 loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)):
        super(GARPNHead, self).__init__()
        self.in_channels=in_channels
        self.feat_channels=feat_channels
        self.octave_base_scale=octave_base_scale
        self.scales_per_octave=scales_per_octave
        self.octave_ratios=octave_ratios

        self.anchor_strides=anchor_strides
        self.anchoring_means=anchoring_means
        self.anchoring_stds=anchoring_stds
        self.target_means=target_means
        self.target_stds=target_stds
        self.loc_filter_thr=loc_filter_thr
        
        self.loss_cls_cfg = loss_cls
        self.loss_bbox_cfg = loss_bbox
        self.loss_loc_cfg = loss_loc
        self.loss_shape_cfg = loss_shape

        from ..builder import build_module
        self.loss_cls = build_module(loss_cls)
        self.loss_bbox = build_module(loss_bbox)

        octave_scales = [2**(i/scales_per_octave) for i in range(scales_per_octave)]
        anchor_scales = [octave_base_scale*octave_scale for octave_scale in octave_scales]
        self.anchor_scales=anchor_scales
        self.guided_anchor = GuidedAnchor(
            in_channels,
            in_channels,
            anchor_scales=anchor_scales,
            anchor_ratios=octave_ratios,
            anchor_strides=anchor_strides,
            anchoring_means=anchoring_means,
            anchoring_stds=anchoring_stds,
            deformable_groups=deformable_groups,
            loc_filter_thr=loc_filter_thr,
            loss_loc=loss_loc,
            loss_shape=loss_shape)
        
        self.num_anchors = 1
        self.num_classes = 2
        self.cls_channels = 1
        self.use_sigmoid = loss_cls.get('use_sigmoid', False)

        logging.info('Finish init of GARPNHead, num_anchors={}, num_classes={}, cls_channels={}, use_sigmoid={}'.format(self.num_anchors, self.num_classes, self.cls_channels, self.use_sigmoid))
        self.init_layers()

    def init_layers(self):
        self.conv = nn.Conv2d(self.in_channels, self.feat_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Conv2d(self.feat_channels, self.num_anchors*self.cls_channels, 1)
        self.regressor = nn.Conv2d(self.feat_channels, self.num_anchors*4, 1)


    def init_weights(self):
        self.guided_anchor.init_weights()
        normal_init(self.conv, std=0.01)
        normal_init(self.classifier, std=0.01)
        normal_init(self.regressor, std=0.01)
        logging.info('Initialized weights for GARPNHead')


    def forward_conv(self, adapt_feats):
        conv_xs = [self.relu(self.conv(af)) for af in adapt_feats]
        cls_outs = [self.classifier(cxs) for cxs in conv_xs]
        reg_outs = [self.regressor(cxs) for cxs in conv_xs]
        return cls_outs, reg_outs

    # finished
    def forward(self, feats):
        loc_outs, shape_outs, adapt_feats = self.guided_anchor(feats)
        cls_outs, reg_outs = self.forward_conv(adapt_feats)
        return cls_outs, reg_outs, loc_outs, shape_outs, adapt_feats
 
    def rpn_target_single_image(self, cls_outs, reg_outs, anchors, in_masks,
                                gt_bbox, gt_label, img_meta, cfg):

        cls_outs = utils.inplace_apply(cls_outs, lambda x:x.view(1, -1))
        reg_outs = utils.inplace_apply(reg_outs, lambda x:x.view(4, -1))
        anchors = utils.inplace_apply(anchors, lambda x:x.view(4, -1))
        in_masks = utils.inplace_apply(in_masks, lambda x:x.view(1, -1))
        cls_out = torch.cat(cls_outs, dim=1)
        reg_out = torch.cat(reg_outs, dim=1)
        anchors = torch.cat(anchors, dim=1)
        in_mask = torch.cat(in_masks, dim=1).squeeze()
        
        in_anchors = anchors[:, in_mask.bool()]
        return anchor_target(cls_out, reg_out, 1, in_anchors, in_mask, gt_bbox, None,
                             assigner=cfg.assigner, sampler=cfg.sampler,
                             target_means=self.target_means, target_stds=self.target_stds)
        

    def rpn_target(self, cls_outs, reg_outs, anchors, in_masks,
                   gt_bboxes, gt_labels, img_metas, cfg):
        
        num_imgs = len(img_metas)
        cls_outs_img = utils.split_by_image(cls_outs)
        reg_outs_img = utils.split_by_image(reg_outs)
        rpn_tars = unpack_multi_result(multi_apply(
            self.rpn_target_single_image,
            cls_outs_img,
            reg_outs_img,
            anchors,
            in_masks,
            gt_bboxes,
            gt_labels,
            img_metas,
            cfg))
        return rpn_tars

        
    def loss(self, cls_outs, reg_outs, loc_outs, shape_outs,
             adapt_feats, gt_bboxes, gt_labels, img_metas, cfg):
        num_imgs = len(img_metas)
        ga_loss = self.guided_anchor.loss(
            loc_outs, shape_outs, adapt_feats, gt_bboxes, gt_labels, img_metas, cfg)

        anchors, in_masks = self.guided_anchor.create_rpn_anchors(loc_outs, shape_outs, img_metas)
        rpn_tars = self.rpn_target(
            cls_outs, reg_outs, anchors, in_masks, gt_bboxes, gt_labels, img_metas, cfg)

        tar_cls_outs, tar_reg_outs, tar_labels, tar_anchors, tar_bboxes, tar_params = rpn_tars
        tar_label = torch.cat(tar_labels)
        pos_places = (tar_label==1)
        tar_cls_out = torch.cat(tar_cls_outs, dim=1)
        tar_reg_out = torch.cat(tar_reg_outs, dim=1)[:, pos_places]
        tar_param = torch.cat(tar_params, dim=1)[:, pos_places]
        
        avg_factor = tar_label.shape[0]
        tar_cls_out = tar_cls_out.t()
        cls_loss = self.loss_cls(tar_cls_out, tar_label) / avg_factor
        reg_loss = self.loss_bbox(tar_reg_out, tar_param) / avg_factor

        
        rpn_loss = {'cls_loss':cls_loss, 'reg_loss':reg_loss}
        rpn_loss.update(ga_loss)
        return rpn_loss

    def predict_bboxes_single_image(self, cls_outs, reg_outs, anchors, in_masks, img_meta, cfg):
        # cfg: pre_nms, pos_nms, max_num, nms_iou, min_bbox_size
        num_lvls = len(cls_outs)
        device = cls_outs[0].device
        img_size = img_meta['img_shape'][:2]
        H, W = img_size
        min_size = img_meta['scale_factor'] * cfg.min_bbox_size
        
        cls_outs = utils.inplace_apply(cls_outs, lambda x:x.view(1, -1))
        reg_outs = utils.inplace_apply(reg_outs, lambda x:x.view(4, -1))
        anchors  = utils.inplace_apply(anchors,  lambda x:x.view(4, -1))
        in_masks = utils.inplace_apply(in_masks, lambda x:x.view(-1))

        cls_scores, cls_labels, pred_bboxes = [], [], []

        for i in range(num_lvls):
            in_mask = in_masks[i]
            cls_out, reg_out = cls_outs[i][:, in_mask], reg_outs[i][:, in_mask]
            anchor = anchors[i][:, in_mask]
            cls_sig = cls_out.sigmoid()
            cls_score = cls_sig[0]
            if cfg.pre_nms > 0 and cfg.pre_nms < len(cls_score):
                _, topk_inds = cls_score.topk(cfg.pre_nms)
                cls_score = cls_score[topk_inds]
                reg_out = reg_out[:, topk_inds]
                anchor = anchor[:, topk_inds]
            pred_bbox = utils.param2bbox(anchor, reg_out, self.target_means, self.target_stds, img_size)
            if min_size > 0:
                non_small = (pred_bbox[2] - pred_bbox[0] + 1 >= min_size) \
                            & (pred_bbox[3] - pred_bbox[1] + 1 >= min_size)
                cls_score = cls_score[non_small]
                pred_bbox = pred_bbox[:, non_small]
            keep = tvops.nms(pred_bbox.t(), cls_score, cfg.nms_iou)
            cls_score = cls_score[keep]
            pred_bbox = pred_bbox[:, keep]

            if cfg.post_nms > 0 and cfg.post_nms < len(cls_score):
                cls_score = cls_score[:cfg.pos_nms]
                pred_bbox = pred_bbox[:, :cfg.pos_nms]
            cls_scores.append(cls_score)
            pred_bboxes.append(pred_bbox)
        mlvl_cls_score = torch.cat(cls_scores)
        mlvl_pred_bbox = torch.cat(pred_bboxes, dim=1)
        max_num = cfg.max_num
        if max_num > 0 and len(mlvl_cls_score) > max_num:
            _, topk_inds = mlvl_cls_score.topk(max_num)
            mlvl_cls_score = mlvl_cls_score[topk_inds]
            mlvl_pred_bbox = mlvl_pred_bbox[:, topk_inds]
        return mlvl_pred_bbox, mlvl_cls_score, None


    def predict_bboxes_from_output(self, cls_outs, reg_outs, loc_outs, shape_outs,
                                   adapt_feats, img_metas, cfg):
        '''
        cfg: pre_nms, pos_nms, max_num, nms_iou, min_bbox_size
        '''
        # create anchors for all images
        with torch.no_grad():
            anchors, in_masks = self.guided_anchor.create_rpn_anchors(
                loc_outs, shape_outs, img_metas)
            cls_outs_img = utils.split_by_image(cls_outs)
            reg_outs_img = utils.split_by_image(reg_outs)
            predict_res = unpack_multi_result(multi_apply(
                self.predict_bboxes_single_image,
                cls_outs_img,
                reg_outs_img,
                anchors,
                in_masks,
                img_metas,
                cfg))

        return predict_res
        
    def predict_bboxes(self, feats, img_metas, cfg):
        forward_res = self.forward(feats)
        pred_res = self.predict_bboxes_from_output(*(forward_res+(img_metas, cfg)))
        return pred_res
