import argparse
parser = argparse.ArgumentParser('Train Cascade RCNN')
parser.add_argument('config', help='Model configs, train configs and test configs.')
parser.add_argument('work_dir', help='Where to save ckpts, log and etc.')
parser.add_argument('--gpu', help='GPU cardinal, only support single GPU at now.')
parser.add_argument('--seed', help='Random seed.')

args = parser.parse_args()

import os, sys, glob, random, logging
import os.path as osp
import mmcv, torch
from lib import data, cascade_rcnn, data_
import torch


def check_args():
    assert osp.exists(args.work_dir) and osp.isdir(args.work_dir), 'work-dir does not exists'
    args.work_dir = osp.realpath(args.work_dir)
    args.config_file = osp.realpath(args.config)
    args.config = mmcv.Config.fromfile(args.config)

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

def main():
    check_args()
    if args.seed is not None:
        set_seed(args.seed)
    config = args.config
    data_opt = {'voc_data_dir': config.data.train.voc_data_dir,
                'min_size': config.data.train.min_size,
                'max_size': config.data.train.max_size}
    dataset = data_.Dataset(data_opt)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        **config.data.train.loader)
    
    train_cfg = config['train_cfg']
    train_cfg['dataloader'] = dataloader
    train_cfg['work_dir'] = args.work_dir
    device = torch.device('cpu')
    if args.gpu is not None:
        device = torch.device('cuda:{}'.format(args.gpu))
    config.device = device
    
    trainer = cascade_rcnn.CascadeRCNNTrain(
        cascade_cfg=config.model,
        dataloader=dataloader,
        work_dir=args.work_dir,
        total_epochs=config.train_cfg.total_epochs,
        optimizer=config.train_cfg.optimizer,
        log_file=config.train_cfg.log_file,
        lr_decay=config.train_cfg.lr_decay,
        save_interval=config.train_cfg.save_interval,
        device=device,
        train_cfg=config.train_cfg,
        test_cfg=config.test_cfg)

    # do not start to log until logging.basicConfig is set
    logging.info('Work dir: {}'.format(args.work_dir))
    logging.info('Config: {}'.format(args.config_file))
    logging.info('Seed: {}'.format(args.seed))
    logging.info('GPU: {}'.format(args.gpu))
    logging.info('Configuration details: {}'.format(config))
    
    trainer.init_detector()    
    trainer.train()
                                                                                                                                        
if __name__ == '__main__':
    main()
