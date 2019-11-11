import sys, os
import torch
import os.path as osp
import math

cur_dir = osp.dirname(osp.realpath(__file__))
sys.path.append(osp.join(cur_dir, '..'))

from lib import data, modules, utils, region
import time

TEST_IMG = osp.join(cur_dir, 'dog.jpg')
TEST_IMG_DIR \
    = '/home/server2/4T/liyiqing/dataset/PASCAL_VOC_07/voc2007_trainval/VOC2007/JPEGImages'
TEST_COCO_JSON \
    = '/home/server2/4T/liyiqing/dataset/PASCAL_VOC_07/voc2007_trainval/voc2007_trainval.json'


def dict2str(d):
    return ', '.join(['{}:{}'.format(k,v) for k, v in d.items()])


import random
random.seed(2019)

def test_proposal_creator():
    img_data = utils.image2tensor(TEST_IMG)
    vgg = modules.VGG()
    feat = vgg(img_data)
    img_size = tuple(img_data.shape[-2:])
    feat_size = tuple(feat.shape[-2:])
    print('Image shape:', img_data.shape)
    print('Feature shape:', feat.shape)

    anchor_scales = [128, 256, 512]
    anchor_ars = [1.0, 0.5, 2.0]
    anchor_gen = region.AnchorGenerator(anchor_scales, anchor_ars)
    rpn = modules.RPN(num_classes=20, num_anchors=len(anchor_scales)*len(anchor_ars))
    cls, reg = rpn(feat)

    print('Classifier output shape:', cls.shape)
    print('Regressor output shape:', reg.shape)

    proposal_gen = region.ProposalCreator(anchor_gen)
    proposals = proposal_gen.proposals(cls, reg, img_size, feat_size)
    print('Number of proposals:', len(proposals))
    for prop in proposals[:10]:
        print('---new proposal---')
        for k, v in prop.items():
            print(k, ':', v)

    prop_filtered = proposal_gen.proposals_filtered(
        cls, reg, img_size, feat_size,
        max_by_score=12000, max_after_nms=2000, nms_iou=0.7)
    print('len of prop_filtered:', len(prop_filtered))
    
    print('Test NMS'.center(90, '*'))
    rois = region.apply_nms(proposals, score_map=lambda x: x['obj_score'],
                            bbox_map=lambda x: x['bbox'], iou_thr=0.7)
    print('ROIs after NMS:', len(rois))
    mut_ious = []
    num_rois = len(rois)
    for i, roi in enumerate(rois):
        for j in range(i+1, num_rois):
            mut_ious.append(region.calc_iou(roi['bbox'], rois[j]['bbox']))
    mut_ious.sort()
    print(mut_ious[-100:])
    print('len of mut_ious:', len(mut_ious))


if __name__ == '__main__':
    #test_iou()
    #debug_one_case()
    test_proposal_creator()
