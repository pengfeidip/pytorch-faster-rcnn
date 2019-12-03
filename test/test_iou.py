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

def rand_bbox(x_bound, y_bound, w_bound, h_bound):
    return region.BBox(xywh=(
        random.randint(0, x_bound),
        random.randint(0, y_bound),
        random.randint(1, w_bound),
        random.randint(1, h_bound)
    ))
    
def points_of_bbox(bbox):
    x,y,w,h = bbox.get_xywh()
    pts = set()
    for i in range(x, x+w):
        for j in range(y, y+h):
            pts.add((i, j))
    return pts

def calc_iou_brute(a, b):
    a_pts = points_of_bbox(a)
    b_pts = points_of_bbox(b)
    overlap = a_pts & b_pts
    return len(overlap) / (len(a_pts) + len(b_pts) - len(overlap))
    

def test_iou_2():
    for i in range(1000):
        a = rand_bbox(200,200,100,300)
        b = rand_bbox(200,200,300,200)
        print(a,b)
        iou1 = region.calc_iou(a,b)
        iou2 = calc_iou_brute(a,b)
        print(iou1, iou2)
        diff = abs(iou1-iou2)
        print(diff)
        if diff > 0.1:
            print(a, b)
    print('No problem')

if __name__ == '__main__':
    test_iou_2()
