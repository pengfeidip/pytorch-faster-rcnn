import sys, os
import torch
import os.path as osp
import math

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
        return '{' + ','.join(['{}:{}'.format(k, dict2str(v)) for k,v in d.items()]) + '}'


import random
random.seed(2019)
torch.manual_seed(2019)


def pass_data():
    print('Test train fasterRCNN'.center(90, '*'))
    dataset = data.CocoDetDataset(TEST_IMG_DIR, TEST_COCO_JSON,
                                  transform=data.faster_transform(1000, 600))
    dataloader = data.torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0,
                                                  shuffle=False)

    faster = faster_rcnn.FasterRCNNModule()
    #faster.to('cuda:0')
    faster.train()
    print('Init FasterRCNNModule:')
    print(faster)
    rpn_loss = loss.RPNLoss(faster.anchor_gen, 10)
    rcnn_loss = loss.RCNNLoss(10)
    num_imgs = 50
    img_id = 0
    optimizer = torch.optim.SGD(faster.parameters(), lr=0.0025, momentum=0.9, weight_decay=0.0001)
    for train_data in dataloader:
        optimizer.zero_grad()
        print('\nA new image:', img_id)
        img_data, bboxes_data = train_data
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
        print('anchor_targets.__len__', len(anchor_targets) if anchor_targets is not None else None)
        print('props', len(props))
        print('props_targets', len(props_targets) if props_targets is not None else None)

        rpnloss = rpn_loss(rpn_cls_out, rpn_reg_out, anchor_targets)
        rcnnloss = rcnn_loss(rcnn_cls_out, rcnn_reg_out, props_targets)
        print('RPN loss:', rpnloss)
        print('RCNN loss:', rcnnloss)
        comb_loss = rpnloss + rcnnloss
        print('Combined loss:', comb_loss)

        comb_loss.backward()
        optimizer.step()
        img_id += 1
        if img_id >= num_imgs:
            break


if __name__ == '__main__':
    pass_data()
    
