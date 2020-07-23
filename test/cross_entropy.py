import os.path as osp
import sys
cur_dir = osp.dirname(osp.realpath(__file__))
sys.path.insert(0, osp.join(cur_dir, '..'))

from lib.heads.fcos_head import length2class, class2length
from lib.losses import distributed_focal_loss as dfl
from lib.losses import multilabel_softmax_cross_entropy_with_logits as ml
import torch

if __name__ == '__main__':
    pred = torch.rand(3, 4)
    tar = pred.softmax(1)
    print('pred:', pred)
    print('target:', tar)
    print('loss:', ml(pred, tar))

    pred = torch.tensor([-1.0, -1.0, 1.0]).view(1,3)
    print('softmax:', pred.softmax(1))
    tar  = torch.tensor([0.0, 0.0, 1.0]).view(1,3)
    print('loss:', ml(pred, tar))
    ce = torch.nn.functional.cross_entropy(pred, torch.tensor([2]))
    print('ce from torch:', ce)
