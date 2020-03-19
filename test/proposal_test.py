import sys, os, time
import os.path as osp
cur_dir = osp.dirname(osp.realpath(__file__))
sys.path.append(osp.join(cur_dir, '..'))
from lib import data
from lib import config
from lib import utils
from lib import region
import torch

img_size = (600,800)
grid=(37,50)
anchor_creator = region.AnchorCreator(device=torch.device('cuda:0'))
anchors = anchor_creator(img_size=(600, 800), grid=(37, 50))
print(anchors.shape)
anchors = anchors.view(4,-1)
device = torch.device('cuda:0')
    
def test_proposal_creator_data():
    rpn_cls_out = torch.rand(1,18,37,50, device=device)
    rpn_reg_out = torch.rand(1,36,37,50, device=device)

    props_creator = region.ProposalCreator(6000, 50, 0.7, 16)
    rois, scores = props_creator(rpn_cls_out, rpn_reg_out, anchors, img_size)
    print(rois.shape)
    print(rois)
    print(scores)
    
    
if __name__ == '__main__':
    #test_find_inside_index()
    test_proposal_creator_data()
    
