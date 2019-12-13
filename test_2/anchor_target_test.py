import sys, os, time
import os.path as osp
cur_dir = osp.dirname(osp.realpath(__file__))
sys.path.append(osp.join(cur_dir, '..'))
from lib import data
from lib import config
from lib import utils
from lib import region
import torch

def test_find_inside_index():
    img_size = (600,800)
    grid=(37,50)
    anchor_creator = region.AnchorCreator(device=torch.device('cuda:0'))
    anchors = anchor_creator(img_size=(600, 800), grid=(37, 50))
    anchors = anchors.view(4, -1)
    inside_idx = region.find_inside_index(anchors, img_size)
    labels = torch.full((16650,), -1) 
    print('inside_idx type:', type(inside_idx))
    print('inside_idx:\n', inside_idx)
    print('inside_idx number:', len(inside_idx[0]))
    labels[inside_idx] = 1
    labels[8000:8050] = 0
    print('1 labels:', (labels==1).sum())

    print('test random sample')
    region.random_sample_label(labels, 128, 256)
    print('1 labels:', (labels==1).sum())
    print('0 labels:', (labels==0).sum())
    print('-1 labels:', (labels==-1).sum())

def test_anchor_target_creator():
    bbox = torch.tensor([
        [115, 131, 613, 513],
        [230,  53, 460, 506],
        [643, 133, 697, 272],
        [706, 158, 800, 257]
    ], device=torch.device('cuda:0'))
    bbox = bbox.t()
    img_size = (600, 800)
    feat_size = (37, 50)
    anchor_creator = region.AnchorCreator(device=torch.device('cuda:0'))
    anchors = anchor_creator(img_size, feat_size).view(4, -1)
    in_index = region.find_inside_index(anchors, img_size)
    in_anchors = anchors[:, in_index[0]]
    anchor_tar_creator = region.AnchorTargetCreator()
    print('in_anchors.shape', in_anchors.shape)
    labels, param, bbox_labels = anchor_tar_creator(img_size, feat_size, in_anchors, bbox)
    print('1 labels:', (labels==1).sum())
    print('0 labels:', (labels==0).sum())
    print('-1 labels:', (labels==-1).sum())
    pos_index = utils.index_of(labels==1)
    pos_anchors = in_anchors[:, pos_index[0]]
    pos_bbox_labels = bbox_labels[:, pos_index[0]]
    print('pos_anchors:', pos_anchors, pos_anchors.shape)
    print('pos_bbox_labels:', pos_bbox_labels, pos_bbox_labels.shape)
    ious = utils.calc_iou(pos_anchors, pos_bbox_labels)
    print('check ious:', ious.tolist())
    print('check ious diag:', ious.diag())
    
    print('Next test performance')
    img_size2 = (605, 805)
    start = time.time()
    for i in range(5000):
        if i % 2 == 0:
            labels, param, bbox_labels = anchor_tar_creator(img_size, feat_size, in_anchors, bbox)
        else:
            labels, param, bbox_labels = anchor_tar_creator(img_size2, feat_size, in_anchors, bbox)
    used = time.time() - start
    print('used {} secs'.format(used))
    
    
    
    
if __name__ == '__main__':
    #test_find_inside_index()
    test_anchor_target_creator()
    
