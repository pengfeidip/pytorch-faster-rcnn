import sys, os, time
import os.path as osp
cur_dir = osp.dirname(osp.realpath(__file__))
sys.path.append(osp.join(cur_dir, '..'))
from lib import data, config, utils, region, modules, faster_rcnn, loss
import torch, random

random.seed(2019)
torch.manual_seed(2019)
device=torch.device('cuda:0')
TEST_IMG_DIR = '/home/server2/4T/liyiqing/dataset/PASCAL_VOC_07/voc2007_trainval/VOC2007/JPEGImages'
TEST_COCO_JSON = '/home/server2/4T/liyiqing/dataset/PASCAL_VOC_07/voc2007_trainval/voc2007_trainval.json'

def test_backbone():
    back, cls = modules.make_vgg16_backbone()
    print(back)

    
if __name__ == '__main__':
    test_backbone()
    
