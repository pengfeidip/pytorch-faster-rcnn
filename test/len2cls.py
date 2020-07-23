import os.path as osp
import sys
cur_dir = osp.dirname(osp.realpath(__file__))
sys.path.insert(0, osp.join(cur_dir, '..'))

from lib.heads.fcos_head import length2class, class2length
import torch

if __name__ == '__main__':
    lens = torch.tensor([0,17,8,15,7])
    print('lens:', lens)
    cls_res = length2class(lens, 6, 4)
    print('len2class:', cls_res)
    print('transfer back to lens:', class2length(cls_res, 4))
    pass
