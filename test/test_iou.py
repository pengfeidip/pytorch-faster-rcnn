import sys, os
import torch
import os.path as osp
import math

cur_dir = osp.dirname(osp.realpath(__file__))
sys.path.append(osp.join(cur_dir, '..'))

from lib import data, modules, utils, region
import time

TEST_IMG = osp.join(cur_dir, 'dog.jpg')
TEST_IMG_DIR = '/home/lee/datasets/VOCdevkit/VOC2007/JPEGImages'
TEST_COCO_JSON = '../data/voc2007_trainval.json'


def dict2str(d):
    return ', '.join(['{}:{}'.format(k,v) for k, v in d.items()])

import random
random.seed(2019)

def test_iou():
    print('Test IoU calculation'.center(90, '*'))
    a = region.BBox(xywh=(0,0,10,20))
    b = region.BBox(xywh=(5,10,5,10))
    print('IoU of {} and {} is {}'.format(a, b, region.calc_iou_v2(a,b)))
    print('IoU of {} and {} is {}'.format(a, b, region.calc_iou(a,b)))
    a = region.BBox(xywh=(0,0,10,20))
    b = region.BBox(xywh=(5,10,10,20))
    print('IoU of {} and {} is {}'.format(a, b, region.calc_iou_v2(a,b)))
    print('IoU of {} and {} is {}'.format(a, b, region.calc_iou(a,b)))
    a = region.BBox(xywh=(0,0,10,20))
    b = region.BBox(xywh=(0,0,10,20))
    print('IoU of {} and {} is {}'.format(a, b, region.calc_iou_v2(a,b)))
    print('IoU of {} and {} is {}'.format(a, b, region.calc_iou(a,b)))
    a = region.BBox(xywh=(0,0,10,20))
    b = region.BBox(xywh=(10,20,5,10))
    print('IoU of {} and {} is {}'.format(a, b, region.calc_iou_v2(a,b)))
    print('IoU of {} and {} is {}'.format(a, b, region.calc_iou(a,b)))
    
    a = region.BBox(xywh=(10,15,10,10))
    b = region.BBox(xywh=(15,18,10,5))
    print('IoU of {} and {} is {}'.format(a, b, region.calc_iou_v2(a,b)))
    print('IoU of {} and {} is {}'.format(a, b, region.calc_iou(a,b)))

if __name__ == '__main__':
    test_iou()
