import sys, os, time
import os.path as osp
cur_dir = osp.dirname(osp.realpath(__file__))
sys.path.append(osp.join(cur_dir, '..'))
from lib import data
from lib import config
from lib import utils
from lib import region
import torch
import random

random.seed(2019)
torch.manual_seed(2019)
device=torch.device('cuda:1')


def test_proposal_target():
    bbox = torch.tensor([
        [115, 131, 613, 513],
        [230,  53, 460, 506],
        [643, 133, 697, 272],
        [706, 158, 800, 257],
        [300, 400, 500, 600]
    ], device=device)
    bbox = bbox.t()
    img_size = (600, 800)
    feat_size = (37, 50)
    anchor_creator = region.AnchorCreator(device=torch.device('cuda:1'))
    anchors = anchor_creator(img_size, feat_size).view(4, -1)
    in_index = region.find_inside_index(anchors, img_size)

    rpn_cls_out = torch.rand(1,18,37,50, device=device)
    rpn_reg_out = torch.rand(1,36,37,50, device=device)
    
    props_creator = region.ProposalCreator(6000, 2000, 0.7, 16)
    rois, scores = props_creator(rpn_cls_out, rpn_reg_out, anchors, img_size)
    print('number of rois:', rois.shape)
    print('scores', scores.shape)
    gt_label = torch.tensor([2,2,5,6,8], device=device)
    props_tar_creator = region.ProposalTargetCreator()
    roi_bbox, roi_label, roi_param = props_tar_creator(rois, bbox, gt_label)
    print('roi_bbox.shape', roi_bbox.shape)
    print('roi_label.shape', roi_label.shape)
    print('roi_param.shape', roi_param.shape)
    print('roi_label==1:', (roi_label==1).sum())
    print('roi_label==0:', (roi_label==0).sum())
    start = time.time()
    for i in range(5000):
        roi_bbox, roi_label, roi_param = props_tar_creator(rois, bbox, gt_label)
    end = time.time()
    print('time used: {} secs'.format(end-start))
        
    
if __name__ == '__main__':
    test_proposal_target()
    
