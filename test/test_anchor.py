import sys, os
import torch
import os.path as osp
import math

cur_dir = osp.dirname(osp.realpath(__file__))
sys.path.append(osp.join(cur_dir, '..'))

from lib import data, modules, utils, region
import time

TEST_IMG = osp.join(cur_dir, 'dog.jpg')

def anchor_generator():
    print('Test anchor generator complex'.center(90, '*'))
    gen = region.AnchorGenerator(scales=[128, 256, 512],
                                 aspect_ratios=[1, 0.5, 2])
    tot_anchors = 0
    for res in gen.generate_anchors((1000, 600), (62, 37), allow_cross=False):
        print('-'*10)
        print('Center:', res['center'])
        print('Anchors:')
        for a in res['bboxes']:
            print(a)
        tot_anchors += len(res['bboxes'])
    print('Total anchors:', tot_anchors)

    print('Test anchor generator simple'.center(90, '*'))
    gen = region.AnchorGenerator(scales=[1],
                                 aspect_ratios=[1])
    tot_anchors = 0
    for res in gen.generate_anchors((3, 4), (3, 4), allow_cross=True):
        print('-'*10)
        print('Center:', res['center'])
        print('Anchors:')
        for a in res['bboxes']:
            print(a)
        tot_anchors += len(res['bboxes'])
    print('Total anchors:', tot_anchors)
    
    

if __name__ == '__main__':
    anchor_generator()
