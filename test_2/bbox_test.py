import sys, os, time
import os.path as osp
cur_dir = osp.dirname(osp.realpath(__file__))
sys.path.append(osp.join(cur_dir, '..'))
from lib import data
from lib import config
from lib import utils
from lib import region
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

def test_iou_performance():
    # "for VOC 5000 images, we only need to do calc_iou 5000 * 16 * 2"
    anchor_creator = region.AnchorCreator(device=torch.device('cuda:0'))
    anchors = anchor_creator((600, 800), (37, 50)).view(4, -1)
    bbox = anchors[:, :5]
    print(anchors.shape)
    print(bbox.shape)
    start = time.time()
    for i in range(80000*2):
        iou_tab = utils.calc_iou(anchors, bbox)
    print(time.time() - start)
    
    
if __name__ == '__main__':
    #test_bbox_2_param()
    #test_iou()
    test_iou_performance()
    
