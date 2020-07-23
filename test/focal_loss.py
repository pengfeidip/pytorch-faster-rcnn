import sys, os, json, time, torch
import os.path as osp
cur_dir = osp.dirname(osp.realpath(__file__))
sys.path.append(osp.join(cur_dir, '..'))
from lib.losses import QualityFocalLoss

def test_qfl():
    pred = torch.rand(3, 4)
    quality = torch.tensor([0.8, 0.0, 0.3])
    label = torch.tensor([1,0,2])
    qfl = QualityFocalLoss()
    loss = qfl(pred, quality, label)
    print('loss:', loss)
    pass

if __name__ == '__main__':
    test_qfl()
