import sys, os, time
import os.path as osp
cur_dir = osp.dirname(osp.realpath(__file__))
sys.path.append(osp.join(cur_dir, '..'))
from lib import data, config, utils, region, modules, faster_rcnn, loss
import torch, random

random.seed(2019)
torch.manual_seed(2019)
device=torch.device('cuda:1')
TEST_IMG_DIR = '/home/server2/4T/liyiqing/dataset/PASCAL_VOC_07/voc2007_trainval/VOC2007/JPEGImages'
TEST_COCO_JSON = '/home/server2/4T/liyiqing/dataset/PASCAL_VOC_07/voc2007_trainval/voc2007_trainval.json'



def test_module():
    dataset = data.CocoDetDataset(TEST_IMG_DIR, TEST_COCO_JSON,
                                  transform=data.faster_transform(
                                      1000, 600,
                                      config.VOC2007_MEAN,
                                      config.VOC2007_STD
                                  ))
    dataloader = data.torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=2,
        shuffle=False
    )
    
    device=torch.device('cuda:0')
    faster_module = faster_rcnn.FasterRCNNModule()
    faster_module.to(device)
    rpn_loss = loss.RPNLoss()
    rcnn_loss = loss.RCNNLoss()
    for img, bboxes, labels, info in dataloader:
        scale = info['scale'].item()
        print(img.shape)
        print(bboxes.shape)
        print(labels.shape)
        print(info)
        print(scale)
        bboxes = bboxes.squeeze(0).t()
        labels = labels.squeeze(0)
        img = img.to(device)
        labels = labels.to(device)
        bboxes = bboxes.to(device)
        faster_res = faster_module(img, bboxes, labels, scale)
        for i in faster_res:
            print(i.shape)
        rpn_tar_cls, rpn_tar_reg, rpn_tar_label, rpn_tar_param, \
            rcnn_cls, rcnn_reg, rcnn_tar_label, rcnn_tar_param = faster_res
        rpnloss = rpn_loss(rpn_tar_cls, rpn_tar_reg, rpn_tar_label, rpn_tar_param)
        rcnnloss = rcnn_loss(rcnn_cls, rcnn_reg, rcnn_tar_label, rcnn_tar_param)
        print('rpnloss:', rpnloss)
        print('rcnnloss:', rcnnloss)
        exit()
    
if __name__ == '__main__':
    test_module()
    
