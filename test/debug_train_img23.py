import sys, os
import torch
import os.path as osp
import math, statistics

cur_dir = osp.dirname(osp.realpath(__file__))
sys.path.append(osp.join(cur_dir, '..'))

from lib import data, modules, utils, region, loss, faster_rcnn
import time

TEST_DIR = '/home/server2/4T/liyiqing/dataset/PASCAL_VOC_07/voc2007_trainval/'
TEST_IMG = osp.join(cur_dir, 'dog.jpg')
TEST_IMG_DIR = osp.join(TEST_DIR, 'VOC2007/JPEGImages')
TEST_COCO_JSON =osp.join(TEST_DIR, 'voc2007_trainval.json')
#TEST_IMG_DIR \
#    = '/home/lee/datasets/VOCdevkit/VOC2007/JPEGImages'
#TEST_COCO_JSON \
#    = '../data/voc2007_trainval.json'
               
def dict2str(d):
    ret = ''
    if not isinstance(d, dict):
        return str(d)
    else:
        return '{ ' + ', '.join(['{}:{}'.format(k, dict2str(v)) for k,v in d.items()]) + ' }'

import random
random.seed(2019)
torch.manual_seed(2019)

def mean(nums):
    if len(nums) == 0:
        return None
    return statistics.mean(nums)
def max_(nums):
    if len(nums) == 0:
        return None
    return max(nums)
def min_(nums):
    if len(nums) == 0:
        return None
    return min(nums)
def stats(nums):
    sts = [x(nums) for x  in [len, max_, min_, mean]]
    return 'len, max, min, mean: ' + str(sts)

def train_one(i, dataset, faster, rpn_loss, rcnn_loss, optimizer, device):
    optimizer.zero_grad()
    print('\n')
    print(('A new image: {}'.format(i)).center(90, '*'))
    img_data, bboxes_data, img_info = dataset[i]
    #img_data, bboxes_data = dataset[0]  # only train one image and track loss
    img_data = img_data.unsqueeze(0)
    bboxes_data = bboxes_data.unsqueeze(0)
    img_data = img_data.to(device)
    bboxes_data = bboxes_data.to(device)
    print('Image info:', img_info)
    print('Image shape:', img_data.shape, 'BBoxes data shape:', bboxes_data.shape)
    gt = region.GroundTruth(bboxes_data)
    print('BBox data:', bboxes_data)
    
    rpn_cls_out, rpn_reg_out, rcnn_cls_out, rcnn_reg_out, \
        anchor_targets, props, props_targets = faster(img_data, gt)
    faster_out = [
        rpn_cls_out, rpn_reg_out, rcnn_cls_out, rcnn_reg_out, \
        anchor_targets, props, props_targets]
    print('rpn_cls_out.shape', rpn_cls_out.shape)
    print('rpn_reg_out.shape', rpn_reg_out.shape)
    print('rcnn_cls_out.shape', rcnn_cls_out.shape)
    print('rcnn_reg_out.shape', rcnn_reg_out.shape)
    
    print('anchor_targets'.center(50, '-'))
    if anchor_targets is not None:
        print('Anchor_targets[0]:', dict2str(anchor_targets[0]))
        print('Number of anchor_targets:', len(anchor_targets))
        print('Positive anchor_targets:',
              sum([1 for ii in anchor_targets if ii['gt_label']==1]))
        print('Stats of positive IOU:',
              stats([ii['iou'] for ii in anchor_targets if ii['gt_label']==1]))
        print('Stats of negative IOU:',
              stats([ii['iou'] for ii in anchor_targets if ii['gt_label']==0]))
    else:
        print('anchor_gargets: None')
            
    print('proposals'.center(50, '-'))
    print('Number of props:', len(props))
    print('props[0]', dict2str(props[0]))
    print('Stats of props area:', stats([ii['adj_bbox'].area() for ii in props]))

        
    print('props_targets'.center(50, '-'))
    if props_targets is not None:
        print('Number of props_targets:', len(props_targets))
        print('props_targets[0]:', dict2str(props_targets[0]))
        print('Stats of all IOU:', stats([ii['iou'] for ii in props_targets]))
        print('Stats of positive IOU:',
              stats([ii['iou'] for ii in props_targets if ii['gt_label']==1]))
        print('Stats of negative IOU:',
              stats([ii['iou'] for ii in props_targets if ii['gt_label']==0]))
        print('Stats of props_targets area:', stats([ii['adj_bbox'].area() for ii in props]))
        print('Stats of posirive props_targets area:',
              stats([ii['adj_bbox'].area() for ii in props_targets if ii['gt_label']==1]))
        print('Stats of negative props_targets area:',
              stats([ii['adj_bbox'].area() for ii in props_targets if ii['gt_label']==0]))
    else:
        print('props_targets: None')
        
    print('LOSS'.center(50, '-'))
    rpnloss = rpn_loss(rpn_cls_out, rpn_reg_out, anchor_targets)
    rcnnloss = rcnn_loss(rcnn_cls_out, rcnn_reg_out, props_targets)
    print('RPN loss:', rpnloss)
    print('RCNN loss:', rcnnloss)
    comb_loss = rpnloss + rcnnloss
    print('Combined loss:', comb_loss)

    comb_loss.backward()
    optimizer.step()
    return faster.state_dict()

def main():
    import traceback
    device = torch.device('cuda:0')
    saved_model = 'saved_model_cuda.pth'
    print('Test train fasterRCNN'.center(90, '*'))
    dataset = data.CocoDetDataset(TEST_IMG_DIR, TEST_COCO_JSON,
                                  transform=data.faster_transform(1000, 600))
    dataloader = data.torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0,
                                                  shuffle=False)

    faster = faster_rcnn.FasterRCNNModule()
    faster.to(device)
    faster.train()
    print('Init FasterRCNNModule:')
    print(faster)
    rpn_loss = loss.RPNLoss(faster.anchor_gen, 10)
    rcnn_loss = loss.RCNNLoss(10)
    optim_sgd = torch.optim.SGD(faster.parameters(), lr=0.0025, momentum=0.9, weight_decay=0.0001)
    optim_adam = torch.optim.Adam(faster.parameters(), lr=0.0001)
    optimizer = optim_adam
    print('Optimizer:', optimizer)
    dataset_size = len(dataset)
    faster_state_dict = None

    try:
        for i in range(dataset_size):
            faster_state_dict = train_one(i, dataset, faster, rpn_loss,
                                          rcnn_loss, optimizer, device)
    except Exception as e:
        print('Traceback:')
        print(traceback.format_exc())
        if faster_state_dict is not None:
            torch.save(faster_state_dict, saved_model)
        print('Encounter an error at image {}, previous state_dict is stored at {}'\
              .format(i, saved_model))
        
def reproduce():
    net = faster_rcnn.FasterRCNNModule()
    state = torch.load('saved_model.pth')
    net.load_state_dict(state)
    dataset = data.CocoDetDataset(TEST_IMG_DIR, TEST_COCO_JSON)
    img, bboxes, img_info = dataset[23]
    img = img.unsqueeze(0)
    bboxes = bboxes.unsqueeze(0)
    print('Image:', img.shape)
    print('Bboxes:', bboxes)

    gt = region.GroundTruth(bboxes)

    out = net(img, gt)

if __name__ == '__main__':
    #pass_data()
    main()
    #reproduce()
