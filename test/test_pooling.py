import sys, os
import torch
import os.path as osp
import math

cur_dir = osp.dirname(osp.realpath(__file__))
sys.path.append(osp.join(cur_dir, '..'))

from lib import data, modules, utils, region
import time

TEST_IMG = osp.join(cur_dir, 'dog.jpg')

def pooling_formula():
    n = 7
    w = 20
    for w in range(1, 21):
        print('*'*40)
        print('w is:', w)
        for i in range(1, n+1):
            s = int((i-1)/n*w)
            e = min(int((i)/n*w)+1, w-1)
            print(s, e, e-s+1)
            
# only tested the bin bounds
def roi_pool_op():
    print('Test roi pooling operation'.center(90, '*'))
    shape = torch.tensor([1, 512, 8, 5])
    t = torch.arange(shape.prod().item()).view(*(shape.tolist()))
    region.roi_pool(t, (5,7))

def roi_pooling():
    print('Test ROIPooling using adaptive max pool'.center(90, '*'))
    roi_pool = region.ROIPooling()
    in_data = [
        torch.rand(1,3,10,20),
        torch.rand(1,3,21, 5),
        torch.rand(1,3, 1, 1),
        torch.rand(1,3, 4, 6),
        torch.rand(1,3, 5,19),
        torch.rand(1,3,30,50),
        torch.rand(4,3,10,10)
    ]
    print('Input shapes:')
    for i in in_data:
        print(i.shape)
    out = roi_pool(in_data)
    print('Output shape:', out.shape)
    print('The corner case input:')
    print(in_data[2])
    print('The corner case output:')
    print(out[2])
            
if __name__ == '__main__':
    pooling_formula()
    roi_pool_op()
    roi_pooling()
    
