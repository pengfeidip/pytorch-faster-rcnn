import sys, os
import torch
import os.path as osp

cur_dir = osp.dirname(osp.realpath(__file__))
sys.path.append(osp.join(cur_dir, '..'))

from lib import data, modules, utils
import time

TEST_IMG = osp.join(cur_dir, 'dog.jpg')

def test_rpn():
    print('Test RPN'.center(90, '*'))
    img_data = utils.image2tensor(TEST_IMG)
    print('Image shape:', img_data.shape)
    vgg = modules.VGG()
    feature = vgg(img_data)
    print('Feature shape:', feature.shape)

    rpn = modules.RPN(num_classes=20, num_anchors=9)
    cls, reg = rpn(feature)
    print('Classifier out shape:', cls.shape)
    print('Regressor out shape:', reg.shape)
    
    print('Create fake image of size 1000 x 600')
    img_fake = torch.rand(1, 3, 1000, 600)
    print('Fake image shape:', img_fake.shape)
    feat = vgg(img_fake)
    print('Output feature size:', feat.shape)
    cls, reg = rpn(feat)
    print('Classifier out shape:', cls.shape)
    print('Regressor out shape:', reg.shape)
    
    

if __name__ == '__main__':
    test_rpn()
