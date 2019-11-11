import sys, os
import torch
import os.path as osp
import math

cur_dir = osp.dirname(osp.realpath(__file__))
sys.path.append(osp.join(cur_dir, '..'))

from lib import data, modules, utils, region, loss
import time

TEST_IMG = osp.join(cur_dir, 'dog.jpg')
TEST_DIR = '/home/server2/4T/liyiqing/dataset/PASCAL_VOC_07/voc2007_trainval'
TEST_IMG_DIR = osp.join(TEST_DIR, 'VOC2007/JPEGImages')
TEST_COCO_JSON = osp.join(TEST_DIR, 'voc2007_trainval.json')

def dict2str(d):
    return ', '.join(['{}:{}'.format(k,v) for k, v in d.items()])

import random
random.seed(2019)


def parameter_xywh(bbox, anchor):
    print('test one conversion'.center(60, '-'))
    print('Received bbox {}, anchor {}'.format(bbox, anchor))
    params = region.xywh2param(bbox, anchor)
    print('Calculate parameters using xywh2param:', params)
    xywh = region.param2xywh(params, anchor)
    print('Calculate back bbox:', xywh)

def test_param_xywh():
    anchor_bbox = region.BBox(xywh=(3,4,5,6))
    bboxes = [
        [3,4,5,6],
        [4,4,5,6],
        [4,5,6,7],
        [2,3,4,5],
        [2,3,7,8]
    ]
    for bbox in bboxes:
        parameter_xywh(bbox, anchor_bbox)

def test_rpn_loss():
    print('Test rpn loss'.center(90, '*'))
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
    num_imgs = 50
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
        num_imgs = num_imgs - 1
        if num_imgs == 0:
            break
            

if __name__ == '__main__':
    test_param_xywh()
    test_rpn_loss()
