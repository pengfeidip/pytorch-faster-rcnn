import sys, os, time
import os.path as osp
cur_dir = osp.dirname(osp.realpath(__file__))
sys.path.append(osp.join(cur_dir, '..'))
from lib import data
from lib import config
from lib import utils
from lib import region
import torch

def print_old(a):
    for k,v in a.items():
        print(k,':',v)


def test_anchor():
    img_size = (600,800)
    grid=(37,50)
    print('Test AnchorCreator'.center(50, '-'))
    anchor_creator = region.AnchorCreator(device=torch.device('cuda:0'))
    print('start to create anchors using new method:')
    anchors = anchor_creator(img_size=(600, 800), grid=(37, 50))
    print(anchors.shape)
    print('first 10 anchors')
    print(anchors.view(4,9,-1)[:,:,0])
    
    ws = anchors[2] - anchors[0]
    hs = anchors[3] - anchors[1]
    
    old_gen = region.AnchorGenerator(scales=[128, 256, 512],
                                     aspect_ratios=[0.5, 1.0, 2.0], allow_cross=True)
    print('start to create anchors using old method')
    old_anchors = list(old_gen.anchors(img_size=img_size, grid=grid))
    print('number of old anchors:', len(old_anchors))
    print('anchor 10:\n', old_anchors[0])
    for i in range(10):
        print('old anchor', i)
        print_old(old_anchors[i])

    print('Next test performance')
    idx = 0
    start = time.time()
    for h in range(500, 700):
        for w in range(800, 1000):
            grid = (int(h/16), int(w/16))
            idx += 1
            anchors = anchor_creator((h,w), grid)
    secs = time.time() - start
    print('create {} set of anchors, used {} seconds or {} mins'\
          .format(idx, secs, secs/60))

    print('Next test performance(old)')
    idx = 0
    start = time.time()
    for h in range(500, 700):
        for w in range(800, 1000):
            grid = (int(h/16), int(w/16))
            idx += 1
            anchors = old_gen.anchors((h, w), grid)
            anchors = list(anchors)
    secs = time.time() - start
    print('create {} set of anchors, used {} seconds or {} mins'\
          .format(idx, secs, secs/60))
    
if __name__ == '__main__':
    test_anchor()
    
