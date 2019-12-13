import sys, os
import os.path as osp
cur_dir = osp.dirname(osp.realpath(__file__))
sys.path.append(osp.join(cur_dir, '..'))
from lib import data
from lib import config
from lib import utils
import torch



def test_bbox_2_param():
    n = 100
    bbox = torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
    bbox = bbox.to(torch.float32)
    base = bbox + 1


    para2 = utils.bbox2param(base, bbox)
    print('para2.shape:', para2.shape)
    print('para2.sum():', para2.sum())
    bbox2 = utils.param2bbox(base, para2)
    print('bbox2.sum():', bbox2.sum())
    print((bbox2-bbox).sum())

def test_iou():
    bbox = torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], dtype=torch.float32)
    a = torch.tensor([[1],[4],[7],[10]], dtype=torch.float32)
    b = torch.tensor([[2],[5],[8],[11]], dtype=torch.float32)
    print(utils.calc_iou(bbox, bbox))
    
if __name__ == '__main__':
    test_bbox_2_param()
    test_iou()
    
