import sys, os
import torch
import os.path as osp
import math

cur_dir = osp.dirname(osp.realpath(__file__))
sys.path.append(osp.join(cur_dir, '..'))

from lib import data, modules, utils, region, loss
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
        return ','.join(['{{{}:{}}}'.format(k, dict2str(v)) for k,v in d.items()])


import random
random.seed(2019)
torch.manual_seed(2019)


def pass_data():
    dataset = data.CocoDetDataset(TEST_IMG_DIR, TEST_COCO_JSON,
                                  transform=data.faster_transform(1000, 600))
    dataloader = data.torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0,
                                                  shuffle=False)
    num_imgs = 1
    img_id = 0
    scales = [128, 256, 512]
    aspect_ratios = [1, 0.5, 2]
    roi_pooling_size = (7, 7)
    print('Scales:', scales, 'Aspect ratios:', aspect_ratios,
          'Roi pooling size:', roi_pooling_size)

    anchor_gen = region.AnchorGenerator(scales, aspect_ratios)
    anchor_target_gen = region.AnchorTargetCreator(anchor_gen)
    props_gen = region.ProposalCreator(anchor_gen)
    props_target_gen = region.ProposalTargetCreator()
    roi_crop = region.ROICropping()
    roi_pool = region.ROIPooling(output_size=roi_pooling_size)
    vgg = modules.VGG()
    rpn = modules.RPN(num_classes=20, num_anchors=len(scales)*len(aspect_ratios))
    vgg16 = vgg.vgg16
    head = modules.Head(num_classes=20,
                        fc1_state_dict=vgg16.classifier[0].state_dict(),
                        fc2_state_dict=vgg16.classifier[3].state_dict())
    print('Initialized head with VGG fully connected layers.')
    print('Number of images to test:', num_imgs)
    for train_data in dataloader:
        print('\nA new image:', img_id)
        img_data, bboxes_data = train_data
        print('Image shape:', img_data.shape, 'BBoxes data shape:', bboxes_data.shape)
        gt = region.GroundTruth(bboxes_data)
        print('BBox data:', bboxes_data)
        feat = vgg(img_data)
        print('Feature map size:', feat.shape)
        img_size, grid = img_data.shape[-2:], feat.shape[-2:]
        anchor_targets = anchor_target_gen.targets(img_size, grid, gt)
        print('Number of anchor targets:', len(anchor_targets))
        print('{} of anchor targets:'.format(5))
        print(dict2str(anchor_targets[5]))
            
        print('Test RPN'.center(90, '*'))
        cls_out, reg_out = rpn(feat)
        print('cls_out shape:', cls_out.shape)
        print('reg_out shape:', reg_out.shape)
        print('cls_out:', cls_out.flatten()[:20], '...')
        print('reg_out:', reg_out.flatten()[:20], '...')

        rpn_loss = loss.RPNLoss(anchor_gen, lamb=10)
        rloss = rpn_loss(cls_out, reg_out, anchor_targets)
        #rpn_loss = loss.rpn_loss(cls_out, reg_out, anchor_gen, anchor_targets, lamb=10)
        print('rpn loss:', rloss)
        props = props_gen.proposals_filtered(cls_out, reg_out, img_size, grid,
                                             12000, 2000, 0.7)
        print('Filtered proposals generated:', len(props))
        print('Filtered proposals[10]', dict2str(props[10]))
        print('Test ProposalTargetCreator'.center(90, '*'))
        prop_targets = props_target_gen.targets(props, gt)
        print('number of proposal targets for training head:', len(prop_targets))
        print('Proposal targets[20]:', dict2str(prop_targets[20]))
        print('Test ROICropping'.center(90, '*'))
        crops, adj_bboxes, gt_bboxes, cates = roi_crop.crop(img_size, feat, prop_targets)
        print('len crops:', len(crops),
              'len categories:', len(cates),
              'len gts:', len(gt_bboxes),
              'len adj_bboxes:', len(adj_bboxes))
        print('gt_bboxes:', ', '.join([str(ii) for ii in gt_bboxes[:10]]))
        print('adj_bboxes[:5]:', ', '.join([str(ii) for ii in adj_bboxes[:5]]))
        roi_pool_res = roi_pool(crops)
        print('roi_pool_res.shape:', roi_pool_res.shape)
        print('roi_pool_res:', roi_pool_res.flatten()[:30], '...')

        cls_out, reg_out = head(roi_pool_res)
        print('Head cls_out.shape:', cls_out.shape)
        print('Head reg_out.shape:', reg_out.shape)
        print('cls_out:', cls_out.flatten()[:20], '...')
        print('reg_out:', reg_out.flatten()[:40], '...')

        head_loss = loss.HeadLoss(lamb=10)
        hloss = head_loss(cls_out, reg_out, adj_bboxes, gt_bboxes, cates)
        #head_loss = loss.head_loss(cls_out, reg_out, adj_bboxes, gt_bboxes, cates, lamb=10)
        print('head_loss:', hloss)
        
        img_id += 1
        if img_id >= num_imgs:
            break


if __name__ == '__main__':
    pass_data()
    
