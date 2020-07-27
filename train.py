import argparse
parser = argparse.ArgumentParser('Train RetinaNet')
parser.add_argument('config', help='Model configs, train configs and test configs.')
parser.add_argument('work_dir', help='Where to save ckpts, log and etc.')
parser.add_argument('--gpu', help='GPU cardinal, only support single GPU at now.')
parser.add_argument('--seed', help='Random seed.')

args = parser.parse_args()

import os, sys, glob, random, logging, shutil
import os.path as osp
import mmcv, torch, numpy as np
from lib import datasets
from lib.trainer import BasicTrainer
import torch

LOG_LEVEL = {'DEBUG': logging.DEBUG, 'INFO': logging.INFO, 'WARNING': logging.WARNING}

def check_args():
    assert osp.exists(args.work_dir) and osp.isdir(args.work_dir), 'work-dir does not exists'
    args.work_dir = osp.realpath(args.work_dir)
    args.config_file = osp.realpath(args.config)
    args.config = mmcv.Config.fromfile(args.config)
    if args.seed is not None:
        args.seed = int(args.seed)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def set_logging(log_file=None, level='DEBUG'):
    assert level in LOG_LEVEL
    log_cfg = {
        'format':'%(asctime)s: %(message)s\t[%(levelname)s]',
        'datefmt':'%y%m%d_%H%M%S_%a',
        'level': LOG_LEVEL[level]
    }
    if log_file is not None:
        log_cfg['filename']=log_file
    logging.basicConfig(**log_cfg)


def main():
    check_args()
    if args.seed is not None:
        set_seed(args.seed)
    config = args.config

    device = torch.device('cpu')  # set default device
    if args.gpu is not None:
        device = torch.device('cuda:{}'.format(args.gpu))
    config.device = device

    if args.seed is not None:
        set_seed(args.seed)

    shutil.copyfile(args.config_file, osp.join(args.work_dir, osp.basename(args.config_file)))

    dataset = datasets.VOCDataset(
        ann_file=config.data.train.ann_file,
        img_prefix=config.data.train.img_prefix,
        pipeline=config.data.train.pipeline
    )
    dataloader = datasets.build_dataloader(dataset, config.data.train.imgs_per_gpu, config.data.train.loader.num_workers,
                                           1, dist=False,
                                           shuffle=config.data.train.loader.shuffle)
    
    train_cfg = config.train_cfg
    train_cfg.dataloader = dataloader
    train_cfg.work_dir = args.work_dir
    test_cfg = config.test_cfg

    log_file = train_cfg.log_file
    log_level = train_cfg.log_level
    if log_file is not None:
        log_file = osp.join(args.work_dir, log_file)
    set_logging(log_file, log_level)

    from lib.builder import build_module
    model = build_module(config.model, train_cfg=train_cfg, test_cfg=test_cfg)

    trainer = BasicTrainer(
        dataloader,
        args.work_dir,
        model,
        train_cfg,
        config.optimizer,
        config.optimizer_config,
        config.lr_config,
        config.ckpt_config,
        None,
        device=device
    )
    
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
