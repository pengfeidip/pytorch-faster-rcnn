import torch.nn as nn
from mmdet.ops.dcn import DeformConv
from mmcv.cnn import normal_init
import logging, torch
from .. import debug, utils, region
from ..utils import multi_apply, unpack_multi_result

# shape_outs [[1, 2, 267, 200], [1, 2, 134, 100], ...]
# anchors [[4, num_anchors, 267, 200], [4, num_anchors, 134, 100], ...]
# img_metas, mainly need img_shape
#
def _shape_target_single_image(shape_outs, anchors, img_meta, assigner, sampler,
                               gt_bbox, gt_label, target_means=None, target_stds=None):
    from ..builder import build_module
    img_size = img_meta['img_shape'][:2]
    grid_sizes = [so.shape[-2:] for so in shape_outs]
    num_anchors = anchors.shape[1]
    
    pass

def map_gt_level(anchor_scale, anchor_strides, gt_bbox):
    # borrow from mmdet
    scales = torch.sqrt((gt_bbox[2] - gt_bbox[0] + 1) * (gt_bbox[3] - gt_bbox[1] + 1))
    min_anchor_size = scales.new_full((1, ), float(anchor_scale * anchor_strides[0]))
    target_lvls = torch.floor(torch.log2(scales) - torch.log2(min_anchor_size) + 0.5)
    target_lvls = target_lvls.clamp(0, len(anchor_strides)-1).long()
    return target_lvls

def paint_bbox(canvas, bbox, scale, val):
    assert canvas.dim() == 2
    bbox = bbox * scale
    bbox = bbox.round().long()
    canvas[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1] = val
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

        self.anchor_creators = [
            region.AnchorCreator(base=base, scales=anchor_scales,
                                 aspect_ratios=anchor_ratios)
            for base in self.anchor_strides]

        self.init_layers()

                

    def init_layers(self):
        self.loc_layer = nn.Conv2d(self.in_channels, 1, 3, padding=1)
        self.shape_layer = nn.Conv2d(self.in_channels, 2, 3, padding=1)

        offset_channels = 3 * 3 * 2 * self.deformable_groups
        self.offset_layer = nn.Conv2d(2, offset_channels, 1, bias=False)
        # Do we turn bias on?
        self.adapt_layer = DeformConv(self.in_channels, self.out_channels, 3, padding=1,
                                      deformable_groups=self.deformable_groups)
        self.relu=nn.ReLU(inplace=True)



    def init_weights(self):
        normal_init(self.offset_layer, std=0.1)
        normal_init(self.adapt_layer, std=0.01)

    def create_anchor(self):
        # infer anchor from loc and shape predictions
        pass

    
    def create_level_anchors(self, in_size, grid_sizes, device):
        for ac in self.anchor_creators:
            ac.to(device)
        return [actr(in_size, grid_sizes[i]) for i, actr in enumerate(
            self.anchor_creators)]

    def shape_target_single_image(self, lvl_anchors, shape_outs,
                                  gt_bbox, gt_label, img_meta, input_size, cfg):
        from ..builder import build_module
        # here shape_outs are shape reformed outs
        device = shape_outs[0].device
        grid_sizes = [so.shape[-2:] for so in shape_outs]
        img_size = img_meta['img_shape'][:2]

        assert cfg.allowed_border == -1 # assert this to make it a little easier
        num_anchors = lvl_anchors[0].shape[1]
        # only flatten spatial dim, i.e. w, h
        anchors = [acs.view(4, num_anchors, -1) for acs in lvl_anchors]
        anchors = torch.cat(anchors, dim=-1) # [4, 9, 80297]
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

        # select target bbox and target label(1 and 0) after in_mask
        tar_idx = labels - 1
        tar_idx[tar_idx < 0] = 0
        tar_bbox = gt_bbox[:, tar_idx]
        tar_bbox = tar_bbox[:, non_neg_places]
        tar_label = labels.detach().clone()
        tar_label[pos_places] = 1
        tar_label = tar_label[non_neg_places]
        return tar_shape_out, tar_bbox, tar_label


    def shape_target(self, shape_reformed_outs, gt_bboxes, gt_labels, img_metas, cfg):
        device = shape_reformed_outs[0].device
        input_size = utils.input_size(img_metas)
        input_size = tuple(input_size)
        grid_sizes = [so.shape[-2:] for so in shape_reformed_outs]
        lvl_anchors = self.create_level_anchors(input_size, grid_sizes, device)
        lvl_anchors = tuple(lvl_anchors)
        num_imgs = len(img_metas)
        shape_outs_img = []
        for i in range(num_imgs):
            shape_outs_img.append([shape[i] for shape in shape_reformed_outs])
        return unpack_multi_result(multi_apply(self.shape_target_single_image,
                                               lvl_anchors,
                                               shape_outs_img,
                                               gt_bboxes,
                                               gt_labels,
                                               img_metas,
                                               input_size,
                                               cfg))


    def loc_target(self, loc_outs, gt_bboxes, gt_labels, img_metas, cfg):
        num_imgs = len(gt_bboxes)
        loc_outs_img = []
        for i in range(num_imgs):
            loc_outs_img.append([loc_out[i] for loc_out in loc_outs])
        return unpack_multi_result(multi_apply(self.loc_target_single_image,
                                                loc_outs_img,
                                                gt_bboxes,
                                                gt_labels,
                                                img_metas,
                                                cfg))
            
    def loc_target_single_image(self, loc_outs, gt_bbox, gt_label, img_meta, cfg):
        '''
        Args:
            loc_outs: list([1, 267, 200], [1, 134, 100], ...)
            gt_bbox: [4, n], n gt bboxes
            gt_label: [n], n gt labels
            img_meta: 'img_shape', 'scale_factor' etc
            cfg: 'sampler', 'assigner', 'center_ratio', 'ignore_ratio', etc
        '''
        # first set all targets to 0 which is negative. -1:ignore, 0:negative, 1:positive
        device = loc_outs[0].device
        targets = [lo.new_full(lo.shape[-2:], 0) for lo in loc_outs]
        scales = [1.0 / x for x in self.anchor_strides]
        num_lvls = len(targets)

        gt_lvl = map_gt_level(self.anchor_scales[0], self.anchor_strides, gt_bbox)
        # second set all ignore areas
        ig_thr = cfg.ignore_ratio
        for i in range(gt_bbox.shape[1]):
            lvl = gt_lvl[i]
            bbox = gt_bbox[:, i]
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1 + 1, y2 - y1 + 1
            w, h = w*cfg.ignore_ratio, h*cfg.ignore_ratio
            ctr_x, ctr_y = (x2 + x1) / 2, (y2 + y1) / 2
            ig_bbox = [ctr_x - w/2, ctr_y - h/2, ctr_x + w/2, ctr_y + h/2]
            ig_bbox = torch.tensor([x.round() for x in ig_bbox], dtype=torch.float)
            paint_bbox(targets[lvl], ig_bbox, scales[lvl], -1)
            if lvl - 1 >= 0:
                paint_bbox(targets[lvl-1], ig_bbox, scales[lvl-1], -1)
            if lvl + 1 <  num_lvls:
                paint_bbox(targets[lvl+1], ig_bbox, scales[lvl+1], -1)
        # third, set all positive areas
        for i in range(gt_bbox.shape[1]):
            lvl = gt_lvl[i]
            bbox = gt_bbox[:, i]
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1 + 1, y2 - y1 + 1
            w, h = w*cfg.center_ratio, h*cfg.center_ratio
            ctr_x, ctr_y = (x2 + x1) / 2, (y2 + y1) / 2
            ctr_bbox = [ctr_x - w/2, ctr_y - h/2, ctr_x + w/2, ctr_y + h/2]
            ctr_bbox = torch.tensor([x.round() for x in ig_bbox], dtype=torch.float)
            paint_bbox(targets[lvl], ctr_bbox, scales[lvl], 1)
        return targets, cfg

    def forward(self, feats):
        assert len(feats) == len(self.anchor_strides)
        loc_outs = [self.loc_layer(feat).sigmoid() for feat in feats]
        shape_outs = [self.shape_layer(feat) for feat in feats]
        offsets = [self.offset_layer(so.detach()) for so in shape_outs]
        adapt_feats = [self.relu(self.adapt_layer(feats[i], offsets[i])) for i in range(len(feats))]
        shape_reformed_outs = [self.sigma*self.anchor_strides[i]*(shape_outs[i].exp()) for i in range(len(feats))]
        return loc_outs, shape_outs, shape_reformed_outs, adapt_feats


    def loss(self, loc_outs, shape_outs, shape_reformed_outs, adapt_feats, gt_bboxes, gt_labels, img_metas):
        pass


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

    def forward_ga(self, feats):
        return self.guided_anchor(feats)

    def forward_conv(self, adapt_feats):
        conv_xs = [self.relu(self.conv(af)) for af in adapt_feats]
        cls_outs = [self.classifier(cxs) for cxs in conv_xs]
        reg_outs = [self.regressor(cxs) for cxs in conv_xs]
        return cls_outs, reg_outs
        
    def forward(self, feats):
        loc_outs, shape_outs, shape_reformed_outs, adapt_feats = self.forward_ga(feats)
        cls_outs, reg_outs = self.forward_conv(adapt_feats)
        return cls_outs, reg_outs, loc_outs, shape_outs, shape_reformed_outs, adapt_feats

    def loss(self, cls_outs, reg_outs, loc_outs, shape_outs,
             shape_reformed_outs, adapt_feats, gt_bboxes, gt_labels, img_metas, cfg):
        print('Reached loss'.center(50, '*'))
        print('cls_outs:')
        debug.tensor_shape(cls_outs)
        print('reg_outs:')
        debug.tensor_shape(reg_outs)
        print('loc_outs:')
        debug.tensor_shape(loc_outs)
        print('shape_outs:')
        debug.tensor_shape(shape_outs)
        print('shape_reformed_outs:')
        debug.tensor_shape(shape_reformed_outs)
        print('adapt_feats:')
        debug.tensor_shape(adapt_feats)
        print('cfg')
        print(cfg)

        shape_tars = self.guided_anchor.shape_target(shape_reformed_outs, gt_bboxes, gt_labels, img_metas, cfg)
        print('shape_tars')
        tar_shape_outs, tar_bbox, tar_label = shape_tars
        num_imgs = len(img_metas)
        for i in range(num_imgs):
            print('img:', i)
            print('tar_shape_out:', tar_shape_outs[i].shape)
            print('tar_bbox:', tar_bbox[i].shape)
            print('tar_label:', utils.count_tensor(tar_label[i]))

        loc_tars = self.guided_anchor.loc_target(
            loc_outs, gt_bboxes, gt_labels, img_metas, cfg)
        loc_tars, cfg = loc_tars
        for i in range(num_imgs):
            print('img:', i)
            debug.tensor_shape(loc_tars[i])
        exit()
        return {}

    def predict_bboxes(cls_outs, reg_outs, loc_outs, shape_outs, shape_reformed_outs, adapt_feats, cfg):
        pass
        


    
