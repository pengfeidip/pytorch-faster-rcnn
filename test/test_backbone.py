import sys, os
import torch
import os.path as osp
cur_dir = osp.dirname(osp.realpath(__file__))
sys.path.append(osp.join(cur_dir, '..'))
from lib import data, modules, utils
import time

TEST_DATA_DIR = '/home/server2/4T/liyiqing/dataset/PASCAL_VOC_07/'
TEST_IMG_DIR = osp.join(TEST_DATA_DIR, 'voc2007_trainval/VOC2007/JPEGImages')
TEST_COCO_JSON = osp.join(TEST_DATA_DIR, 'voc2007_trainval/voc2007_trainval.json')
TEST_IMG = osp.join(TEST_IMG_DIR, '009894.jpg')

def vgg_backbone():
    vgg_bb = modules.VGG()
    print('Successfully loaded pretrained VGG16 backbone.')
    print('Print the network:')
    print(vgg_bb.nn_module)
    print('Print the device:', vgg_bb.device)
    print('Print pretrained:', vgg_bb.pretrained)
    img_data = utils.image2tensor(TEST_IMG)
    print('Input image shape:', img_data.shape)
    out = vgg_bb(img_data)
    print('After pass an image:', out.shape)

if __name__ == '__main__':
    vgg_backbone()
