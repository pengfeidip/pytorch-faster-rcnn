import sys, os, json, time, torch
import os.path as osp
cur_dir = osp.dirname(osp.realpath(__file__))
sys.path.append(osp.join(cur_dir, '..'))
from lib.losses import BoundedIoULoss

def smooth_l1_loss():
    a = torch.rand(3,4)
    b = torch.rand(3,4)
    print(loss.smooth_l1_loss(a,b, 3.0))
    
    a = torch.tensor(3.0)
    b = torch.tensor(3.0)
    print(loss.smooth_l1_loss(a,b, 3.0))

    print(loss.smooth_l1_loss(torch.tensor(0.0),
                              torch.tensor(1/9.0), 3.0))
    print(loss.smooth_l1_loss(torch.tensor(0.0),
                              torch.tensor(1/9.0-0.000001), 3.0))

def test_bounded_iou():
    loss = BoundedIoULoss(0.2)
    x = torch.rand(3,4)
    print(loss(x,x+1))
    

if __name__ == '__main__':
    test_bounded_iou()
