import sys, os, json, time, torch
import os.path as osp
cur_dir = osp.dirname(osp.realpath(__file__))
sys.path.append(osp.join(cur_dir, '..'))
from lib import data, data_, loss
from lib import config

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

if __name__ == '__main__':
    smooth_l1_loss()
