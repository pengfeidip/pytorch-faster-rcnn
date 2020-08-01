from torch import nn
from mmcv.cnn import normal_init
import numpy as np
import logging, torch

from .. import utils, debug, region, anchor, losses

# turn length representation to class representation, according to paper Generalized Focal Loss
def length2class(length, cls_channels, stride):
    '''
    length: a tensor of length
    cls_channels: integer, number of discretization
    stride: real number, size of the discretization length
    '''
    len_shape = length.shape
    len_flat = length.view(-1).float()
    numel = len_flat.numel()
    max_len = (cls_channels - 1) * stride
    len_flat = len_flat.clamp(0, max_len)
    left_pt = (len_flat / stride).long() 
    right_pt = left_pt + 1
    left_bound, right_bound = left_pt * stride, right_pt * stride
    right_prob = (len_flat - left_bound) / stride
    left_prob = (right_bound - len_flat) / stride
    distr = len_flat.new_full((numel, cls_channels), 0, dtype=torch.float)
    distr[torch.arange(numel), left_pt] = left_prob
    distr[torch.arange(numel), right_pt] = right_prob
    return distr.view(*len_shape, -1), left_pt

# turn class representation to length representation, according to paper Generalized Focal Loss
def class2length(cls_score, stride):
    '''
    cls_score: [..., cls_channels], i.e. cls channels are in the last dim
    stride: real number
    '''
    cls_channels = cls_score.shape[-1]
    cls_shape = cls_score.shape[:-1]
    cls_flat = cls_score.view(-1, cls_channels)
    rand_var = cls_score.new([i*stride for i in range(cls_channels)])
    inner_prod = (cls_score * rand_var).sum(-1)
    return inner_prod


# create tensor of shape [grid_heigh, grid_width, dim] with all values equal to 'value'
def make_level_blanks(grids, dim, value, dtype, device):
    return [torch.full(list(grid)+[dim], value, dtype=dtype, device=device) \
            for grid in grids]

# get places where l,t,r,b values are all positive
def positive_ltrb(ltrb):
    pos_mask = ltrb > 0
    return pos_mask.all(dim=-1)

# from the FCOS paper
def centerness(ltrb):
    ltrb = ltrb + 1e-6
    l, t, r, b = [ltrb[..., i] for i in range(4)]
    return torch.sqrt((torch.min(l, r)/torch.max(l, r))*(torch.min(t, b)/torch.max(t, b)))

# transform from ltrb representation to xyxy representation
# ltrb: [grid_h, grid_w, 4]
def ltrb2bbox(ltrb, stride):
    grid_size = ltrb.shape[1:]
    full_idx = utils.full_index(grid_size).to(
        device=ltrb.device, dtype=ltrb.dtype)
    
    coor = full_idx * stride + stride / 2
    bbox = torch.stack([
        coor[:, :, 1] - ltrb[0, :, :],
        coor[:, :, 0] - ltrb[1, :, :],
        ltrb[2, :, :] + coor[:, :, 1],
        ltrb[3, :, :] + coor[:, :, 0]
    ])
    return bbox

# transform xyxy representation to ltrb representation
def bbox2ltrb(bbox, grid, stride):
    full_idx = utils.full_index(grid).to(device=bbox.device).to(dtype=torch.float)
    coor = full_idx * stride + stride/2.0
    ltrb = torch.stack([
        coor[:, :, 1] - bbox[0],
        coor[:, :, 0] - bbox[1],
        bbox[2] - coor[:, :, 1],
        bbox[3] - coor[:, :, 0]
    ], dim=-1)
    return ltrb

# 
def paint_value(canvas, bbox, scale, val):
    bbox = bbox * scale
    bbox = bbox.round().long()
    canvas[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1] = val
    return canvas

# with center (x, y)
def simple_ltrb2bbox(ltrb, ctr_xy):
    x, y = ctr_xy
    return torch.stack([
        x - ltrb[0],
        y - ltrb[1],
        x + ltrb[2],
        y + ltrb[3]
    ])

def topk_by_center(anchors, bbox, k):
    h, w = anchors.shape[-2:]
    anchors_flat = anchors.view(4, -1)
    anchors_ctr = torch.stack(list(utils.center_of(anchors_flat)))
    bbox_ctr = torch.stack(list(utils.center_of(bbox))).view(-1, 1)
    diff = anchors_ctr - bbox_ctr
    l2 = diff.norm(dim=0)
    k_val, k_inds = l2.topk(k, largest=False)
    top_anchors = anchors_flat[:, k_inds]
    k_inds_x, k_inds_y = k_inds % w, k_inds / w
    return k_inds_x, k_inds_y, top_anchors, k_inds_x.numel()

    
'''
TODO:
We mix FCOS, ATSS, Generalized Focal Loss(QFL, DFL) in one FCOSHead, which can be chaotic.
The logic is that GFL only support ATSS, and one can use QFL, DFL or both.
'''
class FCOSHead(nn.Module):
    def __init__(self,
                 num_classes=21,
                 in_channels=256,
                 stacked_convs=4,
                 feat_channels=256,
                 strides=[8, 16, 32, 64, 126],
                 anchor_center_lt=False,
                 reg_std=300,
                 reg_mean=0,
                 reg_coef=[1.0, 1.0, 1.0, 1.0, 1.0],
                 reg_coef_trainable=False,
                 atss_cfg=None,
                 loss_cls=None,
                 loss_bbox=None,
                 loss_dfl=None,
                 loss_centerness=None):
        super(FCOSHead, self).__init__()
        from ..builder import build_module
        # common settings
        self.num_classes=num_classes
        self.cls_channels=num_classes-1
        self.in_channels=in_channels
        self.stacked_convs=stacked_convs
        self.feat_channels=feat_channels
        self.strides=strides
        self.anchor_center_lt=anchor_center_lt
        self.reg_std=reg_std
        self.reg_mean=reg_mean
        self.reg_coef=reg_coef
        self.reg_coef_trainable=reg_coef_trainable
        
        # first check if using ATSS
        self.atss_cfg=atss_cfg
        if atss_cfg is not None:
            self.use_atss=True
            self.anchor_creators=[
                anchor.AnchorCreator(base=stride, scales=[atss_cfg.scale], aspect_ratios=[1.0],
                                     center_lt=anchor_center_lt) \
                for stride in strides]
        else:
            assert loss_cls.type != 'QualityFocalLoss' and loss_bbox.type != 'DistributionFocalLoss'
            self.use_atss=False
            self.level_scale_thr = [0, 64, 128, 256, 512, 1e6]  # for original FCOS setting
        
        # next check loss_cls, there are two choices:
        # FocalLoss or QualityFocalLoss(Generalized Focal Loss)
        assert loss_cls.type in ['FocalLoss', 'QualityFocalLoss']
        if loss_cls.type == 'QualityFocalLoss':
            assert self.use_atss, 'Generalized Focal Loss only support ATSS sampler'
            self.use_centerness = False
            self.use_qfl = True
            if loss_centerness is not None:
                logging.warning('Found loss cfg for centerness while QFL loss is present, will ignore centerness.')
        else:
            self.use_centerness = True
            self.use_qfl = False
        self.loss_cls=build_module(loss_cls)
        
        # next check loss_bbox
        if loss_bbox is None:
            assert self.use_dfl
            self.loss_bbox = None
        else:
            assert loss_bbox.type == 'GIoULoss', 'Bbox loss only support GIoULoss for FCOSHead'
            self.loss_bbox=build_module(loss_bbox)

        # check if use DFL
        if loss_dfl is not None:
            assert loss_dfl.type == 'DistributionFocalLoss'
            self.loss_dfl = build_module(loss_dfl)
            self.use_dfl = True
        else:
            self.use_dfl = False

        # build loss_centerness if use_centerness
        if self.use_centerness:
            self.loss_centerness=build_module(loss_centerness)

        self.use_gfl = self.use_qfl or self.use_dfl
        logging.debug('use_atss: {}, use_centerness: {}, use_qfl: {}, use_dfl: {}'\
                      .format(self.use_atss, self.use_centerness,
                              self.use_qfl, self.use_dfl))
        self.init_layers()
        
    def init_layers(self):
        cls_convs = []
        reg_convs = []
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            cls_convs.append(
                nn.Conv2d(chn, self.feat_channels, 3, padding=1))
            cls_convs.append(nn.ReLU(inplace=True))
            reg_convs.append(
                nn.Conv2d(chn, self.feat_channels, 3, padding=1))
            reg_convs.append(nn.ReLU(inplace=True))
        self.cls_convs = nn.Sequential(*cls_convs)
        self.reg_convs = nn.Sequential(*reg_convs)

        self.fcos_cls = nn.Conv2d(
            self.feat_channels,
            self.cls_channels,
            3,
            padding=1)

        if self.use_dfl:
            self.fcos_reg = nn.Conv2d(self.feat_channels,
                                      self.loss_dfl.cls_channels * 4, 3, padding=1)
            # notice it is 16 * 4 which means cls channel is at first
        else:
            self.fcos_reg = nn.Conv2d(
                self.feat_channels, 4, 3, padding=1)
        # check if use centerness branch
        if self.use_centerness:
            self.fcos_center = nn.Conv2d(
                self.feat_channels, 1, 3, padding=1)


        if self.use_dfl:
            reg_coef = torch.stack([
                torch.ones(self.loss_dfl.cls_channels)*x for x in self.reg_coef]).float()
            self.reg_coef = nn.Parameter(reg_coef, 
                                         requires_grad=self.reg_coef_trainable)
        else:
            self.reg_coef = nn.Parameter(
                torch.tensor(self.reg_coef, dtype=torch.float),
                requires_grad=self.reg_coef_trainable)

    def init_weights(self):
        for m in self.cls_convs:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        for m in self.reg_convs:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        prior_prob = 0.01
        bias_init = float(-np.log((1-prior_prob)/ prior_prob))
        normal_init(self.fcos_cls, std=0.01, bias=bias_init)
        normal_init(self.fcos_reg, std=0.01)
        if self.use_centerness:
            normal_init(self.fcos_center, std=0.01)
        logging.info('Initialized weights for RetinaHead.')

    def forward(self, xs):
        cls_conv_outs = [self.cls_convs(x) for x in xs]
        reg_conv_outs = [self.reg_convs(x) for x in xs]
        cls_outs = [self.fcos_cls(x) for x in cls_conv_outs]
        ctr_outs = None
        if self.use_centerness:
            ctr_outs = [self.fcos_center(x) for x in reg_conv_outs]
        if self.use_dfl:
            reg_outs = []
            for i, x in enumerate(reg_conv_outs):
                x = self.fcos_reg(x)
                coef = self.reg_coef[i].view(-1, 1)
                coef = torch.cat([coef for _ in range(4)], dim=1)
                reg_outs.append(x * coef.view(-1, 1, 1))
        else:
            reg_outs = [torch.exp(self.fcos_reg(x)*self.reg_coef[i]) \
                        for i, x in enumerate(reg_conv_outs)]
        return cls_outs, reg_outs, ctr_outs

    def single_image_targets_atss(self, cls_outs, reg_outs, ctr_outs,
                                  lvl_anchors, gt_bboxes, gt_labels, img_meta, train_cfg):
        logging.debug('Use ATSS to find targets'.center(50, '*'))
        logging.debug('GT lables for current image: {}'.format(gt_labels.tolist()))
        logging.debug('GT bboxes for current image: {}'.format(gt_bboxes.tolist()))
        assert len(cls_outs) == len(reg_outs) \
            == len(self.strides) == len(lvl_anchors) 
        device = cls_outs[0].device
        grids = [x.shape[-2:] for x in cls_outs]
        img_size = img_meta['img_shape'][:2]
        img_h, img_w = img_size
        scales = [1/stride for stride in self.strides]

        num_gt = gt_bboxes.shape[1]
        num_lvl = len(self.strides)

        # first create holders for all target values
        cls_tars = make_level_blanks(grids, 1, -1, dtype=torch.long,  device=device)
        reg_tars = make_level_blanks(grids, 4, -1, dtype=torch.float, device=device)
        ctr_tars = make_level_blanks(grids, 1, -1, dtype=torch.float, device=device)
        max_ious = make_level_blanks(grids, 1,  0, dtype=torch.float, device=device)

        # then find areas inside image
        for i in range(num_lvl):
            paint_value(cls_tars[i], torch.tensor([0, 0, img_w, img_h], dtype=torch.float),
                        scales[i], 0)
            paint_value(ctr_tars[i], torch.tensor([0, 0, img_w, img_h], dtype=torch.float),
                        scales[i], 0)

        # assign positive grids on each level for each gt, we assign centerness later
        for i in range(num_gt):
            gt_bbox = gt_bboxes[:, i]
            gt_label = gt_labels[i]
            ltrb_list = []
            topk_list = [] # a list of (x_inds, y_inds, k_anchors, num)
            for j in range(num_lvl):
                ltrb = bbox2ltrb(gt_bbox, grids[j], self.strides[j])
                ltrb_list.append(ltrb)
                topk_list.append(topk_by_center(lvl_anchors[j], gt_bbox, self.atss_cfg.topk))
                ###
            
            # cat topk information from all levels
            x_inds, y_inds, close_anchors = torch.cat([x[0] for x in topk_list]), \
                                            torch.cat([x[1] for x in topk_list]), \
                                            torch.cat([x[2] for x in topk_list], dim=1)
            num_topk = [x[-1] for x in topk_list]
            logging.debug('number of topk by center distance: '.format(num_topk))
            
            ious = utils.calc_iou(close_anchors, gt_bbox).view(-1)
            mean, std = ious.mean(), ious.std()
            iou_thr = mean + std
            pos_mask = ious > iou_thr  # positive samples by iou
            tot_topk = 0
            # next update positive information level by level
            for lvl, lvl_topk in enumerate(num_topk):
                lvl_x, lvl_y, lvl_iou, lvl_pos_mask = x_inds[tot_topk:tot_topk+lvl_topk],\
                                                      y_inds[tot_topk:tot_topk+lvl_topk],\
                                                      ious[tot_topk:tot_topk+lvl_topk],\
                                                      pos_mask[tot_topk:tot_topk+lvl_topk]
                exist_iou = max_ious[lvl][lvl_y, lvl_x, 0]
                lvl_pos_ltrb = positive_ltrb(ltrb_list[lvl])[lvl_y, lvl_x]
                
                # lvl_mask marks places where positive information should be updated
                # the places are:
                #   1, positive by iou_thr filtering
                #   2, current iou be bigger than the existing iou to chose gt with larger iou
                #   3, center inside gt, by choose places where ltrb > 0
                lvl_mask = (lvl_iou >= exist_iou) & lvl_pos_mask & lvl_pos_ltrb
                lvl_x_chosen = lvl_x[lvl_mask]
                lvl_y_chosen = lvl_y[lvl_mask]

                cls_tars[lvl][lvl_y_chosen, lvl_x_chosen, :] = gt_label  # update gt label
                reg_tars[lvl][lvl_y_chosen, lvl_x_chosen, :] \
                    = ltrb_list[lvl][lvl_y_chosen, lvl_x_chosen, :]  # update reg ltrb
                max_ious[lvl][lvl_y_chosen, lvl_x_chosen, 0] = lvl_iou[lvl_mask]  # update max iou
                tot_topk += lvl_topk

        # next calc centerness from ltrb
        for lvl in range(num_lvl):
            pos_mask = cls_tars[lvl] > 0
            cur_ctr = centerness(reg_tars[lvl])
            cur_ctr = cur_ctr.unsqueeze(-1)
            ctr_tars[lvl][pos_mask] = cur_ctr[pos_mask]

        # for debug
        return cls_tars, reg_tars, ctr_tars, max_ious


    def single_image_targets(self, cls_outs, reg_outs, ctr_outs,
                             gt_bboxes, gt_labels, img_meta, train_cfg):
        gt_bboxes, gt_labels = utils.sort_bbox(gt_bboxes, labels=gt_labels, descending=True)
        device = cls_outs[0].device
        grids = [x.shape[-2:] for x in cls_outs]
        img_size = img_meta['img_shape'][:2]
        img_h, img_w = img_meta['img_shape'][:2]
        scales = [1/stride for stride in self.strides]

        num_gt  = gt_bboxes.shape[1]
        num_lvl = len(self.strides)

        # first create holders for all target values
        cls_tars = make_level_blanks(grids, 1, -1, dtype=torch.long,  device=device)
        reg_tars = make_level_blanks(grids, 4, -1, dtype=torch.float, device=device)
        ctr_tars = make_level_blanks(grids, 1, -1, dtype=torch.float, device=device) 

        # then find areas inside image
        for i in range(num_lvl):
            paint_value(cls_tars[i], torch.tensor([0, 0, img_w, img_h], dtype=torch.float),
                        scales[i], 0)
            paint_value(ctr_tars[i], torch.tensor([0, 0, img_w, img_h], dtype=torch.float),
                        scales[i], 0)

        # assign positive grids on each level for each gt, we assign centerness later
        for i in range(num_gt):
            gt_bbox = gt_bboxes[:, i]
            gt_label = gt_labels[i]
            for j in range(num_lvl):
                ltrb = bbox2ltrb(gt_bbox, grids[j], self.strides[j])
                pos_ltrb = positive_ltrb(ltrb)
                max_ltrb, _ = ltrb.max(2)
                cur_scale = pos_ltrb & \
                            (max_ltrb >= self.level_scale_thr[j]) & \
                            (max_ltrb <  self.level_scale_thr[j+1])
                cls_tars[j][cur_scale] = gt_label
                reg_tars[j][cur_scale] = ltrb[cur_scale]

        
        for i in range(num_lvl):
            pos_mask = cls_tars[i] > 0
            cur_ctr = centerness(reg_tars[i])
            cur_ctr = cur_ctr.unsqueeze(-1)
            ctr_tars[i][pos_mask] = cur_ctr[pos_mask]

        return cls_tars, reg_tars, ctr_tars

    # after finding targets, it applies designated loss settings
    def calc_loss(self, cls_outs, reg_outs, ctr_outs, cls_tars, reg_tars, ctr_tars, max_ious):
        logging.debug('IN Calculation Loss'.center(50, '*'))
        logging.debug('reg_coef: {}'.format(self.reg_coef))
        logging.debug('reg_mean: {}, reg_std: {}'.format(self.reg_mean, self.reg_std))

        # combine targets from all images and calculate loss at once

        num_imgs = len(cls_tars)
        cls_outs = torch.cat([utils.concate_grid_result(x, False) for x in cls_outs], dim=-1)
        reg_outs = torch.cat([utils.concate_grid_result(x, False) for x in reg_outs], dim=-1)
        if ctr_outs is not None:
            ctr_outs = torch.cat([utils.concate_grid_result(x, False) for x in ctr_outs], dim=-1)
        cls_tars = torch.cat([utils.concate_grid_result(x, True)  for x in cls_tars], dim=0)
        reg_tars = torch.cat([utils.concate_grid_result(x, True)  for x in reg_tars], dim=0)
        ctr_tars = torch.cat([utils.concate_grid_result(x, True)  for x in ctr_tars], dim=0)
        max_ious = torch.cat([utils.concate_grid_result(x, True)  for x in max_ious], dim=0)

        chosen_mask = (cls_tars >=0).squeeze()
        pos_mask = (cls_tars > 0).squeeze()
        
        # first calc cls loss
        num_pos_cls = pos_mask.sum().item()
        if self.use_qfl:
            cls_loss = self.loss_cls(cls_outs[:, chosen_mask].t(),
                                     max_ious[chosen_mask].squeeze(),
                                     cls_tars[chosen_mask].squeeze(), avg_factor=num_pos_cls)
        else:
            cls_loss = self.loss_cls(cls_outs[:, chosen_mask].t(),
                                     cls_tars[chosen_mask].squeeze()) / num_pos_cls

        cls_as_weight, _ = cls_outs.detach()[:, pos_mask].sigmoid().max(0)
        #cls_as_weight = max_ious[pos_mask].squeeze()
        
        # next calc ctr loss
        if self.use_centerness:
            ctr_outs = ctr_outs.view(-1, 1) # [m, 1]
            ctr_tars = ctr_tars.squeeze()   # [m]
            num_pos_ctr = num_pos_cls
            pos_ctr_outs = ctr_outs[pos_mask, :]
            pos_ctr_tars = ctr_tars[pos_mask]
            ctr_loss = self.loss_centerness(pos_ctr_outs, pos_ctr_tars) / num_pos_ctr
        else:
            ctr_loss = losses.zero_loss(device=cls_outs.device)

        
        ###### calc reg loss ######
        # first do proper normalization
        pos_reg_tars = reg_tars[pos_mask, :].t()  # [4, m], in ltrb format            
        pos_reg_tars = (pos_reg_tars - self.reg_mean) / self.reg_std
        pos_reg_outs = reg_outs[:, pos_mask]      # [4, m] or [16*4, m], ltrb or distribution
        if self.use_dfl:
            cls_channels = self.loss_dfl.cls_channels
            stride = self.loss_dfl.stride
            pos_reg_tars = pos_reg_tars.contiguous().view(-1) # [4, m ] to [4m], ltrb
            _, left_idx = length2class(pos_reg_tars, cls_channels, stride) # [4m, 16]
            pos_reg_outs = pos_reg_outs.view(cls_channels, -1).t() # [16*4, m] to [16, 4m] to [4m, 16]
            dfl_loss = self.loss_dfl(pos_reg_outs, pos_reg_tars, left_idx,
                                     weight=cls_as_weight.repeat(4).view(-1), avg_factor=4.0) 
        else:
            dfl_loss = losses.zero_loss(device=cls_outs.device)
            
        # calc bbox loss, here it is usually GIoU loss
        if self.loss_bbox is None:
            bbox_loss = losses.zero_loss(device=cls_outs.device)
        else:
            if self.use_dfl:
                stride = self.loss_dfl.stride
                pos_reg_out_ltrb = class2length(pos_reg_outs.softmax(-1), stride).view(4, -1) # [4m] to [4, m]
                pos_reg_tar_ltrb = pos_reg_tars.view(4, -1) # [4m] to [4, m]
                pos_reg_out_simple = simple_ltrb2bbox(pos_reg_out_ltrb, (0.0, 0.0))
                pos_reg_tar_simple = simple_ltrb2bbox(pos_reg_tar_ltrb, (0.0, 0.0))
                bbox_loss = self.loss_bbox(pos_reg_out_simple, pos_reg_tar_simple, weight=cls_as_weight,
                                           avg_factor=1.0)
            elif self.use_qfl:
                pos_reg_outs = simple_ltrb2bbox(pos_reg_outs, (0.0, 0.0))
                pos_reg_tars = simple_ltrb2bbox(pos_reg_tars, (0.0, 0.0))
                bbox_loss = self.loss_bbox(pos_reg_outs, pos_reg_tars, weight=cls_as_weight,
                                           avg_factor=num_pos_cls)
            else:
                # ATSS with no GFL
                assert self.reg_mean <= 0
                pos_reg_outs = simple_ltrb2bbox(pos_reg_outs, (0.0, 0.0))
                pos_reg_tars = simple_ltrb2bbox(pos_reg_tars, (0.0, 0.0))
                bbox_loss = self.loss_bbox(pos_reg_outs, pos_reg_tars, weight=pos_ctr_tars,
                                           avg_factor=num_pos_cls)
            
        all_loss = {'cls_loss': cls_loss, 'ctr_loss': ctr_loss, 'dfl_loss': dfl_loss, 'bbox_loss': bbox_loss}
        return {loss_name:loss_val for loss_name, loss_val in all_loss.items() if loss_val.item() != 0}
        

    def loss(self):
        # 1, find targets
        # 2, calc loss
        pass

    # main interface for detector, it returns fcos head loss as a dict
    # loss: cls_loss, reg_loss, centerness_loss
    def forward_train(self, feats, gt_bboxes, gt_labels, img_metas, train_cfg):
        
        # forward data and calc loss
        cls_outs, reg_outs, ctr_outs = self.forward(feats)
        cls_outs_img = utils.split_by_image(cls_outs)
        reg_outs_img = utils.split_by_image(reg_outs)
        ctr_outs_img = None
        if ctr_outs is not None:
            ctr_outs_img = utils.split_by_image(ctr_outs)

        device = feats[0].device
        grids = [x.shape[-2:] for x in cls_outs_img[0]]

        input_size = utils.input_size(img_metas)
        logging.debug('Input size infered: '.format(input_size))

        if self.use_atss:
            _ = [x.to(device=device) for x in self.anchor_creators]
            lvl_anchors = tuple((
                self.anchor_creators[i](stride, grids[i]).squeeze() \
                for i, stride in enumerate(self.strides)
            ))
            tars = utils.unpack_multi_result(utils.multi_apply(
                self.single_image_targets_atss,
                cls_outs_img,
                reg_outs_img,
                ctr_outs_img,
                lvl_anchors,
                gt_bboxes,
                gt_labels,
                img_metas,
                train_cfg))
        else:
            tars = utils.unpack_multi_result(utils.multi_apply(
                self.single_image_targets,
                cls_outs_img,
                reg_outs_img,
                ctr_outs_img,
                gt_bboxes,
                gt_labels,
                img_metas,
                train_cfg))
        tars = [cls_outs_img, reg_outs_img, ctr_outs_img] + list(tars)
        return self.calc_loss(*tars)

    def predict_single_image(self, cls_outs, reg_outs, ctr_outs, img_meta, test_cfg):
        num_lvl = len(cls_outs)
        use_center = self.use_centerness
        # next calc bbox
        min_size = img_meta['scale_factor'] * test_cfg.min_bbox_size
        img_size = img_meta['img_shape'][:2]
        assert num_lvl == len(self.strides)
        bboxes, scores, centerness = [], [], []
        for i in range(num_lvl):
            if self.use_dfl:
                lvl_reg_out = reg_outs[i] # [16*4, m, n]
                lvl_grid_size = lvl_reg_out.shape[-2:] # [m, n]
                lvl_reg_out = lvl_reg_out.view(self.loss_dfl.cls_channels, -1).t()
                # from [16*4, m, n] to [16, 4*m*n] to [4*m*n, 16]
                lvl_reg_out = lvl_reg_out.softmax(-1)
                lvl_ltrb = class2length(lvl_reg_out, self.loss_bbox.strides[0]) # [4*m*n]
                lvl_ltrb = lvl_ltrb * self.reg_std + self.reg_mean
                lvl_ltrb = lvl_ltrb.view(4, *lvl_grid_size)
                bbox = ltrb2bbox(lvl_ltrb, self.strides[i])
            else:
                reg_outs[i] = reg_outs[i] * self.reg_std + self.reg_mean
                bbox = ltrb2bbox(reg_outs[i], self.strides[i])
            score = cls_outs[i].sigmoid()
            ctr_score = ctr_outs[i].sigmoid() if use_center else None

            bbox = bbox.view(4, -1)
            score = score.view(self.cls_channels, -1)
            ctr_score = ctr_score.view(1, -1) if use_center else None
            
            
            bbox = utils.clamp_bbox(bbox, img_size)
            non_small = (bbox[2]-bbox[0] + 1>min_size) & (bbox[3]-bbox[1]+1>min_size)

            score = score[:, non_small]
            bbox = bbox[:, non_small]
            ctr_score = ctr_score[:, non_small] if use_center else None
            if test_cfg.pre_nms > 0 and test_cfg.pre_nms < score.shape[1]:
                if use_center:
                    max_score, _ = (score * ctr_score).max(0)
                else:
                    max_score, _ = score.max(0)
                _, top_inds = max_score.topk(test_cfg.pre_nms)
                score = score[:, top_inds]
                bbox = bbox[:, top_inds]
                ctr_score = ctr_score[:, top_inds] if use_center else None

            bboxes.append(bbox)
            scores.append(score)
            centerness.append(ctr_score)
        mlvl_score = torch.cat(scores, dim=1)
        mlvl_bbox  = torch.cat(bboxes, dim=1)
        mlvl_ctr = torch.cat(centerness, dim=1).view(-1) if use_center else None
        nms_label_set = list(range(0, self.cls_channels))
        label_adjust = 1

        if 'nms_type' not in test_cfg:
            nms_op = utils.multiclass_nms_mmdet
        elif test_cfg.nms_type == 'official':
            nms_op = utils.multiclass_nms_mmdet
        elif test_cfg.nms_type == 'strict':
            nms_op = utils.multiclass_nms_v2
        else:
            raise ValueError('Unknown nms_type: {}'.format(test_cfg.nms_type))
        keep_bbox, keep_score, keep_label = nms_op(
            mlvl_bbox.t(), mlvl_score.t(), nms_label_set,
            test_cfg.nms_iou, test_cfg.min_score, test_cfg.max_per_img, mlvl_ctr)
        keep_label += label_adjust
        return keep_bbox.t(), keep_score, keep_label
            
                

    # main interface for detector, for testing
    def predict_bboxes(self, feats, img_metas, test_cfg):
        cls_outs, reg_outs, ctr_outs = self.forward(feats)
        cls_outs_img = utils.split_by_image(cls_outs)
        reg_outs_img = utils.split_by_image(reg_outs)
        ctr_outs_img = utils.split_by_image(ctr_outs) if self.use_centerness else None
        
        pred_res =utils.unpack_multi_result(utils.multi_apply(
            self.predict_single_image,
            cls_outs_img,
            reg_outs_img,
            ctr_outs_img,
            img_metas,
            test_cfg))
        return pred_res

    def predict_bboxes_from_output(self):
        pass
