import sys, os
import torch
import os.path as osp
cur_dir = osp.dirname(osp.realpath(__file__))
sys.path.append(osp.join(cur_dir, '..'))
from lib import data, modules
import time


def test_rpn():
    print('Test RPN'.center(90, '*'))
    

if __name__ == '__main__':
    test_rpn()
