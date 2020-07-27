import sys, os, json, time, torch
import os.path as osp
cur_dir = osp.dirname(osp.realpath(__file__))
sys.path.append(osp.join(cur_dir, '..'))
from lib.losses import QualityFocalLoss
from lib.losses import DistributedFocalLoss as DFL
import numpy as np

def test_qfl():
    pred = torch.rand(3, 4)
    quality = torch.tensor([0.8, 0.0, 0.3])
    label = torch.tensor([1,0,2])
    qfl = QualityFocalLoss()
    loss = qfl(pred, quality, label)
    print('loss:', loss)
    pass

def dfl_paper(y, y1, y2, s1, s2):
    loss = (y2-y)*np.log(s1) + (y-y1)*np.log(s2)
    return -loss

def test_dfl():
    pred = torch.tensor([-10, -10, -10, 6, 2, -10]).float()
    pred = torch.rand(6)
    pred_soft = pred.softmax(0)
    print('pred:', pred)
    print('pred.softmax:', pred.softmax(0))
    loss = DFL(6, [4])
    print('loss by impl:', loss(pred.view(1, -1), 13, torch.tensor([3]), 4))
    print('loss by paper:', dfl_paper(13, 12, 16, pred_soft[3], pred_soft[4]))
    pass

if __name__ == '__main__':
    test_qfl()
    test_dfl()
