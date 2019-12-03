import mmcv, torch
import os.path as osp
import mmcv, random, torch, json, logging
from lib import data, faster_rcnn
import sys


HOME = '/home/server2/4T/liyiqing/projects/pytorch-faster-rcnn-test-data/work_dirs/faster_rcnn_paper.py'
CKPT = osp.join(HOME, 'epoch_7.pth')
CONFIG = osp.join(HOME, '../../../pytorch-faster-rcnn/configs/faster_rcnn_paper.py')
IMG_DIR = '/home/server2/4T/liyiqing/dataset/PASCAL_VOC_07/voc2007_trainval/VOC2007/JPEGImages'
GPU = 'cuda:2'
OUT = 'rpn_res.json'



if __name__ == '__main__':
    try:
        IMG_DIR, CONFIG, CKPT, OUT, GPU = sys.argv[1:]
        GPU = 'cuda:{}'.format(GPU)
    except:
        print('\nUsage:', __file__, 'IMG_DIR CONFIG CKPT OUT GPU\n')
        exit()
        
    config = mmcv.Config.fromfile(CONFIG)
    test_data_cfg, train_data_cfg = config.test_data_cfg, config.train_data_cfg
    dataset = data.ImageDataset(IMG_DIR,
                                transform=data.faster_transform(*train_data_cfg.img_size,
                                                                **train_data_cfg.img_norm))
    dataloader = torch.utils.data.DataLoader(dataset, **test_data_cfg.loader_cfg)
    device = torch.device(GPU)
    tester = faster_rcnn.RPNTest(config.model, device=device)
    tester.load_ckpt(CKPT)
    infer_res = tester.inference(dataloader)
    json.dump(infer_res, open(OUT, 'w'))
