import sys, os
import torch
import os.path as osp
import math

cur_dir = osp.dirname(osp.realpath(__file__))
sys.path.append(osp.join(cur_dir, '..'))

from lib import data, modules, utils, region, loss
import time

TEST_IMG = osp.join(cur_dir, 'dog.jpg')
TEST_IMG_DIR \
    = '/home/server2/4T/liyiqing/dataset/PASCAL_VOC_07/voc2007_trainval/VOC2007/JPEGImages'
TEST_COCO_JSON \
    = '/home/server2/4T/liyiqing/dataset/PASCAL_VOC_07/voc2007_trainval/voc2007_trainval.json'
TEST_IMG_DIR \
    = '/home/lee/datasets/VOCdevkit/VOC2007/JPEGImages'
TEST_COCO_JSON \
    = '../data/voc2007_trainval.json'
               

def dict2str(d):
    return ', '.join(['{}:{}'.format(k,v) for k, v in d.items()])


import random
random.seed(2019)
torch.manual_seed(2019)

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

def test_proposal_target_creator():
    dataset = data.CocoDetDataset(TEST_IMG_DIR, TEST_COCO_JSON,
                                  transform=data.faster_transform(1000, 600))
    dataloader = data.torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False)
    scales = [128, 256, 512]
    aspect_ratios = [1, 0.5, 2]
    anchor_gen = region.AnchorGenerator(scales, aspect_ratios)
    target_gen = region.AnchorTargetCreator(anchor_gen)
    num_imgs = 1
    for train_data in dataloader:
        print('\nA new image:', num_imgs)
        img_data, bboxes_data = train_data
        print('Image shape:', img_data.shape, 'BBoxes data shape:', bboxes_data.shape)
        gt = region.GroundTruth(bboxes_data)
        print('BBox data:', bboxes_data)
        vgg = modules.VGG()
        feat = vgg(img_data)
        print('Feature map size:', feat.shape)
        img_size, grid = img_data.shape[-2:], feat.shape[-2:]
        targets = target_gen.targets(img_size, grid, gt)
        print('Number of targets:', len(targets))
        for tar in targets:
            if tar['gt_label'] == 0:
                continue
            print('gt_label:', tar['gt_label'], 'anchor:', tar['anchor']['bbox'],
                  'gt_bbox:',  tar['gt_bbox'], 'id:', tar['anchor']['id'],
                  'iou:', tar['iou'])
            if tar['gt_label'] == 1:
                print('IOU:', region.calc_iou(tar['anchor']['bbox'], tar['gt_bbox']))
        print('Apply rpn...')
        rpn = modules.RPN(num_classes=20, num_anchors=len(scales)*len(aspect_ratios))
        cls_out, reg_out = rpn(feat)
        print('cls_out shape:', cls_out.shape)
        print('reg_out shape:', reg_out.shape)
        rpn_loss = loss.rpn_loss(cls_out, reg_out, anchor_gen, targets, lamb=10)
        print('rpn loss:', rpn_loss)

        props_gen = region.ProposalCreator(anchor_gen)
        props = props_gen.proposals_filtered(cls_out, reg_out, img_size, grid, 12000, 2000, 0.7)
        print('generated filtered proposals')
        print('Test ProposalTargetCreator'.center(90, '*'))
        prop_target_gen = region.ProposalTargetCreator()
        prop_targets = prop_target_gen.targets(props, gt)
        
        num_imgs = num_imgs - 1
        if num_imgs == 0:
            break
    
    pass


if __name__ == '__main__':
    #test_iou()
    #debug_one_case()
    #test_proposal_creator()
    test_proposal_target_creator()
