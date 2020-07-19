from torch import nn
import logging

class FCOS(nn.Module):
    def __init__(self,
                 backbone=None,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None):
        super(FCOS, self).__init__()
        from ..builder import build_module
        self.backbone = build_module(backbone)
        if neck is not None:
            self.neck = build_module(neck)
            self.with_neck = True
        else:
            self.with_neck = False

        self.bbox_head = build_module(bbox_head)
        self.train_cfg, self.test_cfg = train_cfg, test_cfg

        self.init_weights()
        logging.info('Constructed FCOS detector')
        print('neck:')
        print(self.neck)
        print('bbox_head:')
        print(self.bbox_head)

    def init_weights(self):
        self.backbone.init_weights()
        if self.with_neck:
            self.neck.init_weights()
        self.bbox_head.init_weights()
        logging.info('Initialzied weight for FCOS detector')

    def extract_feat(self, x):
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, img_data, gt_bboxes, gt_labels, img_metas):
        # return loss dict
        logging.info('Start to forward detector in train mode')
        logging.info('Input image tensor: {}'.format(img_data.shape))
        feats = self.extract_feat(img_data)
        train_cfg = self.train_cfg
        fcoshead_loss = self.bbox_head.forward_train(
            feats, gt_bboxes, gt_labels, img_metas, train_cfg)
        return fcoshead_loss

    def forward_test(self, img_data, img_metas):
        # return specific format of prediction results
        logging.info('Start to predict for detector')
        test_cfg =self.test_cfg
        feats = self.extract_feat(img_data)
        return self.bbox_head.predict_bboxes(feats, img_metas, test_cfg)
