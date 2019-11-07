import sys, os
import torch
import os.path as osp

cur_dir = osp.dirname(osp.realpath(__file__))
sys.path.append(osp.join(cur_dir, '..'))

from lib import data, modules, utils
import time

TEST_IMG = osp.join(cur_dir, 'dog.jpg')

def test_pooling():
    n = 7
    w = 20
    for w in range(1, 21):
        print('*'*40)
        print('w is:', w)
        for i in range(1, n+1):
            s = int((i-1)/n*w)
            e = min(int((i)/n*w)+1, w-1)
            print(s, e, e-s+1)
    

if __name__ == '__main__':
    test_pooling()
