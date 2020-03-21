import sys, os, time
import os.path as osp
cur_dir = osp.dirname(osp.realpath(__file__))
sys.path.append(osp.join(cur_dir, '..'))
from lib import utils
from lib import region
import torch


def test():
    for i in range(1000):
        a = torch.rand(4, 1000, device='cuda:0')
        b = torch.rand(4, 1000, device='cuda:0')
        ele = utils.elem_iou(a, b)
        tab = utils.calc_iou(a, b)
        if (ele!=tab.diag()).any():
            print('WARNING, found diff')
            print(ele.tolist())
            print(tab.tolist())

if __name__ == '__main__':
    test()
