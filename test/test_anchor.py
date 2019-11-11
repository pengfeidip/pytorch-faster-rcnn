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
    print('Test anchor generator simple'.center(90, '*'))
    gen = region.AnchorGenerator(scales=[1, 2],
                                 aspect_ratios=[1, 0.5])
    tot_anchors = 0
    for anchor in gen.anchors((30, 50), (3, 5)):
        print('\n'+'-'*10)
        print('Anchor members:', list(anchor.keys()))
        for k, v in anchor.items():
            print(k, ':', v, end='|')
        tot_anchors += 1
    print('Total anchors:', tot_anchors)

def debug_anchor_id():
    gen = region.AnchorGenerator(scales=[128, 256, 512],
                                 aspect_ratios=[1, 0.5, 2])
    all_anchors = gen.anchors_list((1000,600), (62,37))
    all_anchors = [x['id'] for x in all_anchors]
    print('Number of all_anchors:', len(all_anchors))
    print('Number of unique all_anchors:', len(set(all_anchors)))

if __name__ == '__main__':
    anchor_generator()
    debug_anchor_id()
