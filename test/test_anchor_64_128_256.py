import sys, os
import torch
import os.path as osp
import math, statistics
import logging

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

def test_train():
    dataset = data.CocoDetDataset(TEST_IMG_DIR, TEST_COCO_JSON,
                                  transform=data.faster_transform(1000, 600))
    dataloader = data.torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=2,
                                                  shuffle=True)
    
    faster_configs = dict(anchor_scales=[64, 128, 256])
    trainer = faster_rcnn.FasterRCNNTrain(faster_configs,
                                          dataloader,
                                          '../work_dirs/anchor_scales_64_128_256/',
                                          12,
                                          log_file='train.log',
                                          log_level=logging.DEBUG,
                                          device=torch.device('cuda:2'),
                                          optim_kwargs=dict(lr=0.001, momentum=0.9, weight_decay=0.0005),
                                          seed=2019)
    trainer.init_module()
    #trainer.resume_from(12)
    trainer.train()

if __name__ == '__main__':
    test_train()
