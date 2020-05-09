from torch import nn
from .. import utils
from ..utils import class_name
import logging, torch


class CascadeRCNN(nn.Module):
    def __init__(self,
                 num_stages=3,
                 backbone=None,
                 neck=None,
                 rpn_head=None,
                 roi_extractor=None,
                 shared_head=None,
                 rcnn_head=None,
                 train_cfg=None,
                 test_cfg=None):
        super(CascadeRCNN, self).__init__()
        from ..builder import build_module
        assert num_stages > 0
        self.num_stages=num_stages

        self.backbone = build_module(backbone)
        if neck is not None:
            if isinstance(neck, list):
                self.neck = nn.Sequential(*[build_module(neck_cfg) for neck_cfg in neck])
            else:
                self.neck = build_module(neck)
            self.with_neck=True
        else:
            self.with_neck=False

        self.rpn_head = build_module(rpn_head)

        if isinstance(roi_extractor, list):
            assert len(roi_extractor) >= self.num_stages
            self.roi_extractors = nn.ModuleList([build_module(roi_extractor[i]) for i in range(self.num_stages)])
        else:
            self.roi_extractors = nn.ModuleList([build_module(roi_extractor) for _ in range(self.num_stages)])

        if shared_head is not None:
            self.shared_head = build_module(shared_head)
            self.with_shared_head = True
        else:
            self.with_shared_head = False

        if isinstance(rcnn_head, list):
            assert len(rcnn_head) >= self.num_stages
            rcnn_head = [build_module(rcnn_head[i]) for i in range(self.num_stages)]
        else:
            assert self.num_stages==1, 'rcnn_head must be consistent with num_stages'
            rcnn_head = [build_module(rcnn_head)]
        self.rcnn_head = nn.ModuleList(rcnn_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights()
        logging.info('Constructed CascadeRCNN')
        logging.info('Number of stages: {}'.format(self.num_stages))
        print('neck')
        print(self.neck)
        print('rpn')
        print(self.rpn_head)
        print('rcnn')
        print(self.rcnn_head)

    def init_weights(self):
        self.rpn_head.init_weights()
        # init neck weights
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                _ = [nk.init_weights() for nk in self.neck]
            else:
                self.neck.init_weights()
        for i, bbox_head in enumerate(self.rcnn_head):
            bbox_head.init_weights()
            logging.info('(weights of bbox_head={})'.format(i))
        if self.with_shared_head:
            self.shared_head.init_weights()
        logging.info('Initialized weights for CascadeRCNN')

    def extract_feat(self, x):
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        return x
                                               
    def forward_train(self, img_data, gt_bboxes, gt_labels, img_metas):
        losses = {}
        feats = self.extract_feat(img_data)
        train_cfg = self.train_cfg

        logging.debug('{}: forward train'.format(class_name(self)))

        rpn_gt_labels = [torch.full_like(gt_label, 1) for gt_label in gt_labels]
        if class_name(self.rpn_head) in ['GARPNHead']:
            forward_res = self.rpn_head(feats)
            rpn_loss = self.rpn_head.loss(*(forward_res+(gt_bboxes, gt_labels, img_metas, train_cfg.rpn)))
            losses.update(rpn_loss)
            rpn_props = self.rpn_head.predict_bboxes_from_output(*(forward_res+(img_metas, train_cfg.rpn_proposal,)))
        else:
            rpn_cls_outs, rpn_reg_outs = self.rpn_head(feats)
            rpn_cls_loss, rpn_reg_loss = self.rpn_head.loss(
                rpn_cls_outs, rpn_reg_outs, gt_bboxes, rpn_gt_labels, img_metas, train_cfg.rpn)
            losses['rpn_cls_loss'] = rpn_cls_loss
            losses['rpn_reg_loss'] = rpn_reg_loss
            rpn_props = self.rpn_head.predict_bboxes_from_output(
                rpn_cls_outs, rpn_reg_outs, img_metas, train_cfg.rpn_proposal)
            rpn_props = rpn_props[0]
        
        logging.debug('{}:  proposals from rpn: {}'.format(
            class_name(self), '\n' + '\n'.join([str(pr.shape) for pr in rpn_props])))
        
        props = rpn_props
        for i in range(self.num_stages):
            logging.info('{}: in stage {}'.format(class_name(self), i).center(70, '*').format(i))
            cur_rcnn_head = self.rcnn_head[i]
            cur_roi_extractor = self.roi_extractors[i]
            cur_rcnn_train_cfg = train_cfg.rcnn[i]

            tar_props, tar_bboxes, tar_labels, tar_params, tar_is_gts \
                = cur_rcnn_head.bbox_targets(props, gt_bboxes, gt_labels, cur_rcnn_train_cfg)
            
            logging.info('{}: target props: {}'.format(
                class_name(self), ', '.join([str(tp.shape) for tp in tar_props])))
            
            roi_outs = cur_roi_extractor(feats, tar_props)
            logging.debug('{}: rois from extractor: {}'.format(
                class_name(self), '\n'+'\n'.join([str(roi.shape) for roi in roi_outs])))
            
            if self.with_shared_head:
                raise NotImplementedError('multi-image shared head is not implemented for CascadeRCNN')
            
            cls_outs, reg_outs = cur_rcnn_head(roi_outs)
            #logging.debug('cls_outs by current head: {}'.format([co.shape for co in cls_outs]))
            #logging.debug('reg_outs by current head: {}'.format([ro.shape for ro in reg_outs]))
            cur_cls_loss, cur_reg_loss = cur_rcnn_head.calc_loss(
                cls_outs, reg_outs, tar_labels, tar_params, cur_rcnn_train_cfg)
            losses['rcnn_{}_cls_loss'.format(i)] = cur_cls_loss * train_cfg.stage_loss_weight[i]
            losses['rcnn_{}_reg_loss'.format(i)] = cur_reg_loss * train_cfg.stage_loss_weight[i]
            
            if i < self.num_stages - 1:
                with torch.no_grad():
                    refined_props = cur_rcnn_head.refine_bboxes(tar_props, tar_labels, reg_outs, tar_is_gts, img_metas)
                    logging.debug('{}: refinded props: {}'.format(
                        class_name(self), ', '.join([str(rps.shape) for rps in refined_props])))
                    props = refined_props
        return losses


    def forward_test(self, img_data, img_metas):
        logging.info('start to predict for detector')
        logging.debug('img_data: {}'.format(img_data.shape))
        logging.debug('img_metas: {}'.format(img_metas))
        test_cfg = self.test_cfg
        feats = self.extract_feat(img_data)
        logging.debug('Feature size: {}'.format([feat.shape for feat in feats]))

        rpn_props = self.rpn_head.predict_bboxes(feats, img_metas, test_cfg.rpn)
        
        props = rpn_props[0]
        logging.debug('{}: proposals from rpn: {}'.format(class_name(self), [pr.shape for pr in props]))

        img_sizes = [img_meta['img_shape'][:2] for img_meta in img_metas]

        ms_cls_outs = []
        for i in range(self.num_stages):
            logging.info('test in stage: {}'.format(i).center(80, '+'))

            cur_rcnn_head = self.rcnn_head[i]
            cur_roi_extractor = self.roi_extractors[i]

            roi_outs = cur_roi_extractor(feats, props)

            cls_outs, reg_outs = cur_rcnn_head(roi_outs)

            ms_cls_outs.append(cls_outs)
            if i < self.num_stages - 1:
                bbox_labels = [cls_out.argmax(1) for cls_out in cls_outs] # TODO: what about use_sigmoid?
                if cur_rcnn_head.use_sigmoid:
                    bbox_labels += 1
                props = cur_rcnn_head.refine_bboxes(props, bbox_labels, reg_outs, None, img_metas)

        mi_cls_outs = utils.unpack_multi_result(ms_cls_outs)
        mi_cls_outs = [sum(img_cls_out)/self.num_stages for img_cls_out in mi_cls_outs]

        test_res = utils.unpack_multi_result(
            utils.multi_apply(self.rcnn_head[-1].predict_bboxes_single_image,
                              props,
                              mi_cls_outs,
                              reg_outs,
                              img_sizes,
                              test_cfg.rcnn))

        return test_res
        
        

