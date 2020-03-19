import sys, os, time
import os.path as osp
cur_dir = osp.dirname(osp.realpath(__file__))
sys.path.append(osp.join(cur_dir, '..'))
from lib.registry import build_module
import torch
import random

from copy import deepcopy

random.seed(2020)


backbone_cfg = dict(
    type='ResNet50',
    out_layers=(1,2,3,4),
    frozen_stages=1
)
fpn_cfg = dict(
    type='FPN',
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
    num_outs=5,
    start_level=0,
    end_level=-1,
    extra_use_convs=False,
    extra_convs_on_inputs=True,
    relu_before_extra_convs=False
)

def test():
    img_data = torch.rand(1, 3, 1344, 800)
    
    backbone = build_module(backbone_cfg)
    feats = backbone(img_data)
    print('Features:')
    for x in feats:
        print(x.shape)
        
    print()

    print('config:', fpn_cfg)
    neck = build_module(fpn_cfg)
    print(neck)
    neck_out = neck(feats)
    for x in neck_out:
        print(x.shape)

    print('extra_use_convs=True')
    cfg = deepcopy(fpn_cfg)
    cfg['extra_use_convs']=True
    print('config:', cfg)
    neck = build_module(cfg)
    print(neck)
    neck_out = neck(feats)
    for x in neck_out:
        print(x.shape)

    print('RetinaNet neck')
    cfg = deepcopy(fpn_cfg)
    cfg['extra_use_convs']=True
    cfg['start_level'] = 1
    print('config:', cfg)
    neck = build_module(cfg)
    print(neck)
    neck_out = neck(feats)
    for x in neck_out:
        print(x.shape)

    print('')
    cfg = deepcopy(fpn_cfg)
    cfg['extra_use_convs']=True
    cfg['start_level'] = 1
    cfg['extra_convs_on_inputs']=False
    print('config:', cfg)
    neck = build_module(cfg)
    print(neck)
    neck_out = neck(feats)
    for x in neck_out:
        print(x.shape)
    


if __name__ == '__main__':
    test()
