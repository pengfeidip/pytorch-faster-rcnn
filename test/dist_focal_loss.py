import os.path as osp
import sys
cur_dir = osp.dirname(osp.realpath(__file__))
sys.path.insert(0, osp.join(cur_dir, '..'))

from lib.heads.fcos_head import length2class, class2length
from lib.losses import distributed_focal_loss as dfl
import torch

if __name__ == '__main__':
    lens = torch.tensor([0,17,8,15,7])
    print('lens:', lens)
    cls_res, left_idx = length2class(lens, 6, 4)
    print('left idx:', left_idx)
    print('len2class:', cls_res)
    print('transfer back to lens:', class2length(cls_res, 4))
    print('dfl of lens and cls:', dfl(cls_res, lens, left_idx, stride=4))
    pass
