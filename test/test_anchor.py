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
    print('Test anchor generator'.center(90, '*'))
    gen = region.AnchorGenerator(scales=[128, 256, 512],
                                 aspect_ratios=[1, 0.5, 2])
    tot_anchors = 0
    for res in gen.generate_anchors((1000, 600), (60, 40)):
        print('-'*10)
        print('Center: x:', res['center'].x, 'y:', res['center'].y)
        print('Anchors:')
        for a in res['anchors']:
            print(a)
        tot_anchors += len(res['anchors'])
    print('Total anchors:', tot_anchors)
    

if __name__ == '__main__':
    anchor_generator()
