from torch import nn
from ..anchor import AnchorCreator
from ..region import inside_grid_mask, inside_anchor_mask
from ..anchor import anchor_target
from ..utils import class_name
from .. import losses
from .. import utils
import logging, torch

'''
Anchor head, as a typical head, takes in some input(features or rois) and does two predictions, 
one is class of the object and the other is the bbox of the object. 
For anchor head, it uses anchors to do predictions.
'''

class AnchorHead(nn.Module):
    def __init__(self,
                 num_classes,
                 anchor_scales=[8],
                 anchor_ratios=[0.5, 1.0, 2.0],
                 anchor_strides=[4, 8, 16, 32, 64],
                 anchor_center_lt=False,
                 target_means=[0.0, 0.0, 0.0, 0.0],
                 target_stds=[1.0, 1.0, 1.0, 1.0],
                 loss_cls=None,
                 loss_bbox=None):
        super(AnchorHead, self).__init__()
        self.num_classes=num_classes
        self.anchor_scales=anchor_scales
        self.anchor_ratios=anchor_ratios
        self.anchor_strides=anchor_strides
        self.anchor_base_sizes=anchor_strides
        self.anchor_center_lt=anchor_center_lt
        self.anchor_creators=[AnchorCreator(base=base_size, scales=anchor_scales,
                                            aspect_ratios=anchor_ratios, center_lt=anchor_center_lt)
                              for base_size in anchor_strides]
        self.num_anchors=len(anchor_scales)*len(anchor_ratios)
        
        self.target_means=target_means
        self.target_stds=target_stds
        from ..builder import build_module
        self.loss_cls=loss_cls
        if isinstance(loss_cls, dict):
            self.loss_cls=build_module(loss_cls)
        self.loss_bbox=loss_bbox
        if isinstance(loss_bbox, dict):
            self.loss_bbox=build_module(loss_bbox)

        self.use_sigmoid=self.loss_cls.use_sigmoid
        self.cls_channels=num_classes-1 if self.use_sigmoid else num_classes

    def to(self, device):
        super(AnchorHead, self).to(device)
        for ac in self.anchor_creators:
            ac.to(device)

    def init_layers(self):
        raise NotImplementedError('init_layers is not implemented')

    def init_weights(self):
        raise NotImplementedError('init_weights is not implemented')

    def forward(self):
        raise NotImplementedError('forward is not implemented')

    def create_anchors(self, grid_sizes):
        return [actr(self.anchor_strides[i], grid_sizes[i]) for i, actr in enumerate(self.anchor_creators)]

    def single_image_targets(self, level_cls_outs, level_reg_outs, gt_bbox, gt_label,
                             level_anchors, input_size, grid_sizes, img_meta, train_cfg):
        '''
        Args:
            level_cls_outs: class predict on all levels
            [180, 100, 136], [180, 50, 68], [180, 25, 34], [180, 13, 17], [180, 7, 9]
            level_reg_outs: bbox predict on all levels
            [36, 100, 136],  [36, 50, 68],  [36, 25, 34],  [36, 13, 17],  [36, 7, 9]
            level_anchors:
            [4, 9, 100, 136],  [4, 9, 50, 68],  [4, 9, 25, 34],  [4, 9, 13, 17],  [4, 9, 7, 9]
        '''
        logging.debug(' {}: find targets for one image '.format(class_name(self)).center(50, '*'))
        device=level_cls_outs[0].device
        cls_outs = [lvl_cls_out.view(self.cls_channels, -1) for lvl_cls_out in level_cls_outs]
        reg_outs = [lvl_reg_out.view(4, -1) for lvl_reg_out in level_reg_outs]
        cls_out = torch.cat(cls_outs, dim=1)
        reg_out = torch.cat(reg_outs, dim=1)
        logging.debug('cls_out: {}'.format(cls_out.shape))
        logging.debug('reg_out: {}'.format(reg_out.shape))
        
        anchors = [lvl_anchors.view(4, -1) for lvl_anchors in level_anchors]
        anchors = torch.cat(anchors, dim=1)
        logging.debug('anchors:    {}'.format(anchors.shape))
        img_size = img_meta['img_shape'][:2]
        in_img_mask = inside_anchor_mask(anchors, img_size, train_cfg.allowed_border)
        in_grid_masks = [inside_grid_mask(self.num_anchors, img_size, grid_sizes[lvl], stride, device)
                         for lvl, stride in enumerate(self.anchor_strides)]
        in_grid_mask = torch.cat(in_grid_masks)
        logging.debug('in_img_mask: {}'.format(in_img_mask.sum().item()))
        logging.debug('in_grid_mask: {}'.format(in_grid_mask.sum().item()))
        in_mask = in_img_mask & in_grid_mask.bool()
        in_anchors = anchors[:, in_mask]

        logging.debug('in_mask: {}'.format(in_mask.sum().item()))
        logging.debug('in_anchors: {}'.format(in_anchors.shape))
        tar_cls_out, tar_reg_out, tar_labels, tar_anchors, tar_bbox, tar_param \
            = anchor_target(cls_out, reg_out, self.cls_channels,
                               in_anchors, in_mask, gt_bbox, gt_label,
                               train_cfg.assigner, train_cfg.get('sampler', None),
                               self.target_means, self.target_stds)
        logging.debug('after calc targets, pos={}, neg={}'.format((tar_labels==0).sum().item(),
                                                                  (tar_labels> 0).sum().item()))
        return tar_cls_out, tar_reg_out, tar_labels, tar_param

    def calc_loss(self, tar_cls_out, tar_reg_out, tar_labels, tar_param, train_cfg):
        '''
        Loss is usally calculate on one place, i.e. combined targets
        Args:
            tar_cls_out: [20, n], class predict on targets
            tar_reg_out: [4,  n], bbox predict on targets
            tar_labels:  [n], labels on targets
            tar_param: [4, n], delta values on targets
        '''
        logging.debug(' {}: calculate loss of combined images '.format(class_name(self)).center(50, '*'))
        logging.debug('tar_labels: 0:{}, >0:{}'.format(
            (tar_labels==0).sum().item(), (tar_labels>0).sum().item()))
        device = tar_cls_out.device
        cls_loss, reg_loss = losses.zero_loss(device), losses.zero_loss(device)
        sampling = 'sampler' in train_cfg
        pos_tars = (tar_labels>0)
        avg_factor = len(tar_labels) if sampling else pos_tars.sum()
        logging.debug('avg_factor: {}'.format(avg_factor))
        if tar_labels.numel() != 0:
            cls_loss = self.loss_cls(tar_cls_out.t(), tar_labels) / avg_factor
            if pos_tars.sum() == 0:
                logging.warning('{} recieved no positive samples to train'.format(self.__class__.__name__))
            else:
                reg_loss = self.loss_bbox(tar_reg_out[:, pos_tars], tar_param[:, pos_tars]) / avg_factor
        else:
            logging.warning('{} recieved no samples to train, return dummy zero losses')
        return cls_loss, reg_loss
        

    '''
    1, create anchors for all feature levels which can be shared by all images
    2, calculate targets for each image
        2.1, reshape and concate all the outputs
        2.2, reshape and concate all the anchors from different levels
        2.4, find inside anchors and valid anchors
        2.5, assign and sample targets all together
    3, combine targets from all images
    4, calculate loss at once
    '''
    def loss(self, cls_outs, reg_outs, gt_bboxes, gt_labels, img_metas, train_cfg):
        '''
        Args:
            cls_outs: class prediction on all levels, on each level it contains multi-image results
            [2, 180, 100, 152], 2 is number of images, 180 is num_anchors*num_cls, 100*152 is size of feature map
            reg_outs: bbox prediction on all levels, on each level it contains multi-image results
            [2, 36, 100, 152], 2 is number of images, 36 is num_anchors*4, 100*152 is size of feature map
        '''
        logging.info(' {}: find targets and calculate loss '.format(class_name(self)).center(50, '='))
        logging.info('cls_out: {}'.format('\n' + '\n'.join([str(cls_out.shape) for cls_out in cls_outs])))
        logging.info('reg_out: {}'.format('\n' + '\n'.join([str(reg_out.shape) for reg_out in reg_outs])))
        # first move to proper device
        device=cls_outs[0].device
        self.to(device)
        
        num_imgs = len(img_metas)
        num_levels = len(cls_outs)
        input_size = utils.input_size(img_metas)
        logging.info('Get input size: {}'.format(input_size))
        grid_sizes = [cls_out.shape[-2:] for cls_out in cls_outs]
        logging.info('Grid sizes: {}'.format(grid_sizes))
        level_anchors = self.create_anchors(grid_sizes)
        logging.info('level_anchors: {}'.format('\n' + '\n'.join([str(ac.shape) for ac in level_anchors])))

        img_tar_cls_out, img_tar_reg_out, img_tar_label, img_tar_param = [], [], [], []
        for i, gt_info in enumerate(zip(gt_bboxes, gt_labels, img_metas)):
            gt_bbox, gt_label, img_meta = gt_info
            level_cls_outs = [cls_out[i] for cls_out in cls_outs]
            level_reg_outs = [reg_out[i] for reg_out in reg_outs]
            tar_cls_out, tar_reg_out, tar_label, tar_param \
                = self.single_image_targets(level_cls_outs, level_reg_outs, gt_bbox, gt_label,
                                            level_anchors, input_size, grid_sizes, img_meta, train_cfg)
            img_tar_cls_out.append(tar_cls_out)
            img_tar_reg_out.append(tar_reg_out)
            img_tar_label.append(tar_label)
            img_tar_param.append(tar_param)

        img_tar_cls_out = torch.cat(img_tar_cls_out, dim=1)
        img_tar_reg_out = torch.cat(img_tar_reg_out, dim=1)
        img_tar_label   = torch.cat(img_tar_label)
        img_tar_param   = torch.cat(img_tar_param, dim=1)

        logging.info('tar_cls_out of all images: {}'.format(img_tar_cls_out.shape))
        logging.info('tar_labels  of all images: {}'.format(img_tar_label.shape))
        logging.info('tar_labels: 0:{}, >0:{}'.format(
            (img_tar_label==0).sum().item(), (img_tar_label>0).sum().item()))

        return self.calc_loss(img_tar_cls_out, img_tar_reg_out, img_tar_label, img_tar_param, train_cfg)
        

    def forward_train(self, feats, gt_bboxes, gt_labels, img_metas, train_cfg):
        cls_outs, reg_outs = self.forward(feats)
        return self.loss(cls_outs, reg_outs, gt_bboxes, gt_labels, img_metas, train_cfg)
    
    # this method combines proposals created in all levels and then does one nms on the result
    def predict_single_image(self, level_cls_outs, level_reg_outs, level_anchors, img_meta, test_cfg):
        '''
        Args:
            level_cls_outs: [180, 152, 100], [180, 76, 50], ...
            level_reg_outs: [36, 152, 100], [26, 76, 50], ...
            level_anchors: [4, 9, 152, 100], [4, 9, 76, 50], ...
        '''
        logging.info(' {}: predict one image '.format(class_name(self)).center(50, '*'))
        cls_outs = [lvl_cls_out.view(self.cls_channels, -1) for lvl_cls_out in level_cls_outs]
        reg_outs = [lvl_reg_out.view(4, -1) for lvl_reg_out in level_reg_outs]
        anchors = [anchor.view(4, -1) for anchor in level_anchors]
        num_levels = len(level_cls_outs)
        device = level_cls_outs[0].device
        img_size = img_meta['img_shape'][:2]
        H, W = img_size
        min_size = img_meta['scale_factor'] * test_cfg.min_bbox_size
        cls_scores, pred_bboxes = [], []
        for i in range(num_levels):
            cls_out, reg_out = cls_outs[i], reg_outs[i]
            anchor = anchors[i]
            
            if self.use_sigmoid:
                cls_score = cls_out.sigmoid()
            else:
                cls_score = cls_out.softmax(dim=0)
            if test_cfg.pre_nms > 0 and test_cfg.pre_nms < cls_score.shape[1]:
                if self.use_sigmoid:
                    max_score, _ = cls_score.max(0)
                else:
                    max_score, _ = cls_score[1:, :].max(0)
                _, topk_inds = max_score.topk(test_cfg.pre_nms)
                cls_score = cls_score[:, topk_inds]
                reg_out = reg_out[:, topk_inds]
                anchor = anchor[:, topk_inds]
            pred_bbox = utils.param2bbox(anchor, reg_out, self.target_means, self.target_stds, img_size)
            if min_size > 0:
                non_small = (pred_bbox[2]-pred_bbox[0] + 1 >= min_size) \
                            & (pred_bbox[3]-pred_bbox[1] + 1 >= min_size)
                cls_score = cls_score[:, non_small]
                pred_bbox = pred_bbox[:, non_small]
            cls_scores.append(cls_score)
            pred_bboxes.append(pred_bbox)
        mlvl_cls_score = torch.cat(cls_scores, dim=1)
        mlvl_pred_bbox = torch.cat(pred_bboxes, dim=1)
        if self.use_sigmoid:
            nms_label_set = list(range(0, self.num_classes-1))
            label_adjust = 1
        else:
            nms_label_set = list(range(1, self.num_classes))
            label_adjust = 0
        nms_mode = test_cfg.get('nms_type', 'official')
        keep_bbox, keep_score, keep_label = utils.multiclass_nms(
            mlvl_pred_bbox.t(), mlvl_cls_score.t(), nms_label_set,
            test_cfg.nms_iou, test_cfg.min_score, test_cfg.max_per_img, mode=nms_mode)
        keep_label += label_adjust
        return keep_bbox.t(), keep_score, keep_label

    def predict_bboxes(self, feats, img_metas, test_cfg):
        cls_outs, reg_outs = self.forward(feats)
        return self.predict_bboxes_from_output(cls_outs, reg_outs, img_metas, test_cfg)
            
    def predict_bboxes_from_output(self, cls_outs, reg_outs, img_metas, test_cfg):
        '''
        Args:
            cls_outs: list(Tensor) [2, 180, 100, 152], ...
            reg_outs: list(Tensor) [2, 36, 100, 152], ...
        '''
        num_levels = len(cls_outs)
        num_imgs = len(img_metas)
        device=cls_outs[0].device
        _ = [ac.to(device) for ac in self.anchor_creators]
        input_size = utils.input_size(img_metas)
        grid_sizes = [cls_out.shape[-2:] for cls_out in cls_outs]
        level_anchors = self.create_anchors(grid_sizes)
        preds = []
        
        for i, img_meta in enumerate(img_metas):
            level_cls_outs = [cls_out[i] for cls_out in cls_outs]
            level_reg_outs = [reg_out[i] for reg_out in reg_outs]
            bbox, score, label = self.predict_single_image(
                level_cls_outs, level_reg_outs, level_anchors, img_meta, test_cfg)
            preds.append([bbox, score, label])
        return utils.unpack_multi_result(preds)
