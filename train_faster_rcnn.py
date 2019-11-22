import argparse
parser = argparse.ArgumentParser('Train Faster RCNN')
parser.add_argument('--work-dir', required=True, metavar='REQUIRED',
                    help='Where to save ckpts, log and etc.')
parser.add_argument('--config', required=True, metavar='REQUIRED',
                    help='Model configs, train configs and test configs.')
parser.add_argument('--resume-from', 
                    help='Epoch to resume from, so will start at epoch+1.')
parser.add_argument('--gpu', 
                    help='GPU cardinal, only support single GPU at now.')
parser.add_argument('--seed',
                    help='Random seed.')

args = parser.parse_args()

import os, sys, glob, random, logging
import os.path as osp
import mmcv, torch
from lib import data, faster_rcnn


def check_args():
    assert osp.exists(args.work_dir) and osp.isdir(args.work_dir), '--work-dir does not exists'
    args.work_dir = osp.realpath(args.work_dir)
    args.config_file = osp.realpath(args.config)
    args.config = mmcv.Config.fromfile(args.config)
    if args.resume_from is not None:
        args.resume_from = int(args.resume_from)

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

def main():
    check_args()
    if args.seed is not None:
        set_seed(args.seed)
    config = args.config
    train_data_cfg = config.train_data_cfg
    dataset = data.CocoDetDataset(train_data_cfg.img_dir,
                                  train_data_cfg.json,
                                  transform=data.faster_transform(
                                      *train_data_cfg.img_size,
                                      **train_data_cfg.img_norm))
    dataloader = torch.utils.data.DataLoader(dataset, **train_data_cfg.loader_cfg)

    train_cfg = config['train_cfg']
    train_cfg['faster_configs'] = config['model']
    train_cfg['dataloader'] = dataloader
    train_cfg['work_dir'] = args.work_dir
    device = torch.device('cpu')
    if args.gpu is not None:
        device = torch.device('cuda:{}'.format(args.gpu))
    train_cfg['device'] = device
    
    trainer = faster_rcnn.FasterRCNNTrain(**train_cfg)
    # do not start to log until logging.basicConfig is set
    logging.info('Work dir: {}'.format(args.work_dir))
    logging.info('Config: {}'.format(args.config_file))
    logging.info('Seed: {}'.format(args.seed))
    logging.info('GPU: {}'.format(args.gpu))
    logging.info('Image size: {}'.format(train_data_cfg.img_size))
    logging.info('Image norm: {}'.format(train_data_cfg.img_norm))
    logging.info('Configuration details: {}'.format(config))

    trainer.init_module()
    if args.resume_from is not None:
        trainer.resume_from(args.resume_from)
    
    trainer.train()
    
    

if __name__ == '__main__':
    main()
    pass
