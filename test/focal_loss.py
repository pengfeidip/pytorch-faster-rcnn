import sys, os, json, time, torch
import os.path as osp
cur_dir = osp.dirname(osp.realpath(__file__))
sys.path.append(osp.join(cur_dir, '..'))
from lib.losses import QualityFocalLoss
from lib.losses import DistributionFocalLoss as DFL
import numpy as np
import torch.nn.functional as F

def test_qfl():
    pred = torch.rand(3, 4)
    quality = torch.tensor([0.8, 0.0, 0.3])
    label = torch.tensor([1,0,2])
    qfl = QualityFocalLoss()
    loss = qfl(pred, quality, label)
    print('loss:', loss)
    pass

def dfl_paper(y, y1, y2, s1, s2):
    loss = (y2-y)*np.log(s1) + (y-y1)*np.log(s2)
    return -loss

def test_dfl():
    pred = torch.tensor([-10, -10, -10, 6, 2, -10]).float()
    pred = torch.rand(6)
    pred_soft = pred.softmax(0)
    print('pred:', pred)
    print('pred.softmax:', pred.softmax(0))
    loss = DFL(6, [4])
    print('loss by impl:', loss(pred.view(1, -1), 13, torch.tensor([3]), 4))
    print('loss by paper:', dfl_paper(13, 12, 16, pred_soft[3], pred_soft[4]))

def quality_focal_loss(pred, target, beta=2.0):
    # label denotes the category id, score denotes the quality score
    label, score = target
    
    # negatives are supervised by 0 quality score
    pred_sigmoid = pred.sigmoid()
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = F.binary_cross_entropy_with_logits(
        pred, zerolabel, reduction='none') * scale_factor.pow(beta)
    
    # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    bg_class_ind = pred.size(1)
    pos = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)
    pos_label = label[pos].long()
    # positives are supervised by bbox quality (IoU) score
    scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
    loss[pos, pos_label] = F.binary_cross_entropy_with_logits(
        pred[pos, pos_label], score[pos],
        reduction='none') * scale_factor.abs().pow(beta)
    
    loss = loss.sum(dim=1, keepdim=False)
    return loss



def test_qfl():
    qfl = QualityFocalLoss()
    pred = torch.rand(5, 3)
    label  = torch.tensor( [0,  2,  3,  3,  1])
    label_ = torch.tensor( [1,  3,  0,  0,  2])
    quality = torch.tensor([0.7,0.8,0.2,0.0,0.3])
    qfl_res = qfl(pred, quality, label_)
    mmdet_res = quality_focal_loss(pred, (label, quality))
    print('this qfl:', qfl_res)
    print('mmdet qfl:', mmdet_res.sum())
    

if __name__ == '__main__':
    test_qfl()
    #test_dfl()
