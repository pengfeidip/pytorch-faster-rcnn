import sys, os
import torch
import os.path as osp
import math

cur_dir = osp.dirname(osp.realpath(__file__))
sys.path.append(osp.join(cur_dir, '..'))

from lib import data, modules, utils, region
import time

TEST_IMG = osp.join(cur_dir, 'dog.jpg')
TEST_IMG_DIR \
    = '/home/server2/4T/liyiqing/dataset/PASCAL_VOC_07/voc2007_trainval/VOC2007/JPEGImages'
TEST_COCO_JSON \
    = '/home/server2/4T/liyiqing/dataset/PASCAL_VOC_07/voc2007_trainval/voc2007_trainval.json'


def dict2str(d):
    return ', '.join(['{}:{}'.format(k,v) for k, v in d.items()])


import random
random.seed(2019)
def test_iou():
    print('Test IoU calculation'.center(90, '*'))
    a = region.BBox(xywh=(0,0,10,20))
    b = region.BBox(xywh=(5,10,5,10))
    print('IoU of {} and {} is {}'.format(a, b, region.calc_iou(a,b)))
    a = region.BBox(xywh=(0,0,10,20))
    b = region.BBox(xywh=(5,10,10,20))
    print('IoU of {} and {} is {}'.format(a, b, region.calc_iou(a,b)))
    a = region.BBox(xywh=(0,0,10,20))
    b = region.BBox(xywh=(0,0,10,20))
    print('IoU of {} and {} is {}'.format(a, b, region.calc_iou(a,b)))
    a = region.BBox(xywh=(0,0,10,20))
    b = region.BBox(xywh=(10,20,5,10))
    print('IoU of {} and {} is {}'.format(a, b, region.calc_iou(a,b)))

def test_anchor_target_creator():
    NUM_WORKERS = 0
    print('Test AnchorTargetCreator'.center(90, '*'))
    dataset = data.CocoDetDataset(TEST_IMG_DIR, TEST_COCO_JSON,
                                  transform=data.faster_transform(1000, 600))
    dataloader = data.torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=NUM_WORKERS,
        shuffle=False)
    start = time.time()
    scales = [128, 256, 512]
    aspect_ratios = [1, 0.5, 2]
    anchor_gen = region.AnchorGenerator(scales, aspect_ratios)
    target_gen = region.AnchorTargetCreator(anchor_gen)
    num_imgs = 5
    print('Number of images to test:', num_imgs)
    for train_data in dataloader:
        print('\nA new image')
        img_data, bboxes_data = train_data
        print('Image shape:', img_data.shape, 'BBoxes data shape:', bboxes_data.shape)
        gt = region.GroundTruth(bboxes_data)
        print('BBox data:', bboxes_data)
        vgg = modules.VGG()
        out = vgg(img_data)
        print('Feature map size:', out.shape)
        img_size, grid = img_data.shape[-2:], out.shape[-2:]
        targets = target_gen.targets(img_size, grid, gt)
        print('Number of targets:', len(targets))
        for tar in targets:
            if tar['gt_label'] == 0:
                continue
            print('gt_label:', tar['gt_label'], 'anchor:', tar['anchor']['bbox'],
                  'gt_bbox:',  tar['gt_bbox'], 'id:', tar['anchor']['id'],
                  'iou:', tar['iou'])
            if tar['gt_label'] == 1:
                print('IOU:', region.calc_iou(tar['anchor']['bbox'], tar['gt_bbox']))
        num_imgs = num_imgs - 1
        if num_imgs == 0:
            break
    secs_used = time.time() - start
    print('Finished loading one batch using {} workers, time used: {} mins'\
          .format(NUM_WORKERS, secs_used/60))
    print('End of test AnchorTargetCreator'.center(90, '*'))


def debug_one_case():
    img_size, feat_size = (600, 800), (37, 50)
    scales, aspect_ratios = (128, 256, 512), (1, 0.5, 2)
    gt_bbox = torch.tensor([6, 389, 101, 210, 0]).view(1, 1, 5)
    gt = region.GroundTruth(gt_bbox)
    anchor_gen = region.AnchorGenerator(scales, aspect_ratios)
    target_gen = region.AnchorTargetCreator(img_size, feat_size,
                                            anchor_gen, ground_truth=gt)
    targets = target_gen.targets()
    print('Number of targets:', len(targets))
    for tar in targets[:2]:
        print('anchor:', dict2str(tar['anchor']))
        print('gt_bbox:', tar['gt_bbox'], 'gt_label:', tar['gt_label'], 'category:', tar['category'], 'iou:', tar['iou'])

if __name__ == '__main__':
    #test_iou()
    test_anchor_target_creator()
    #debug_one_case()
