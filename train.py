import argparse
parser = argparse.ArgumentParser('Train RetinaNet')
parser.add_argument('config', help='Model configs, train configs and test configs.')
parser.add_argument('--work-dir', help='Where to save ckpts, log and etc.')
parser.add_argument('--gpu', help='GPU cardinal, only support single GPU at moment.')
parser.add_argument('--debug', action='store_true',
                    help='Output debug log into a file in work_dir.')
parser.add_argument('--seed', help='Random seed.')

args = parser.parse_args()

import os, sys, glob, random, logging, shutil
import os.path as osp
import mmcv, torch, numpy as np
from lib import datasets
from lib.trainer import BasicTrainer
import torch

LOG_LEVEL = {'DEBUG': logging.DEBUG, 'INFO': logging.INFO, 'WARNING': logging.WARNING}

# basic checking file/dir existence etc.
def check_args():
    if args.work_dir is not None:
        assert osp.exists(args.work_dir) and osp.isdir(args.work_dir), 'work-dir does not exists'
        args.work_dir = osp.realpath(args.work_dir)
    args.config_file = osp.realpath(args.config)
    assert osp.exists(args.config_file), 'Config file not exits.'
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

def disable_logging():
    logger = logging.getLogger()
    logger.disabled=True

def safe_mkdir(d):
    try:
        os.mkdir(d)
    except:
        pass
    
def main():
    check_args()
    config = args.config

    # set seed
    if args.seed is not None:
        set_seed(args.seed)

    # set device
    device = torch.device('cpu')  
    if args.gpu is not None:
        device = torch.device('cuda:{}'.format(args.gpu))
    config.device = device
    
    # set work_dir
    if args.work_dir is not None:
        pass
    elif 'work_dir' in config and config.work_dir is not None:
        args.work_dir = config.work_dir
    else:
        config_file_name = osp.basename(args.config_file)
        work_dir = osp.join(osp.dirname(osp.realpath(args.config_file)),
                            '..',
                            'work_dirs',
                            config_file_name)
        args.work_dir = work_dir
        safe_mkdir(work_dir)

    # copy config file
    shutil.copyfile(args.config_file, osp.join(args.work_dir, osp.basename(args.config_file)))

    # set debug log, either output logging content to debug_file or disable logging entirely.
    if args.debug:
        MAX_NAME_ITER = 10000
        for i in range(MAX_NAME_ITER):
            debug_log_file = osp.join(args.work_dir, 'debug_{}.log'.format(i))
            if not osp.exists(debug_log_file):
                set_logging(debug_log_file, 'DEBUG')
                config.debug_log = debug_log_file
                break
        if i == MAX_NAME_ITER-1:
            raise RuntimeError('Too many existing debug.log files')
    else:
        disable_logging()

    # initiate dataset and dataloader
    dataset = datasets.VOCDataset(
        ann_file=config.data.train.ann_file,
        img_prefix=config.data.train.img_prefix,
        pipeline=config.data.train.pipeline
    )
    dataloader = datasets.build_dataloader(dataset, config.data.train.imgs_per_gpu,
                                           config.data.train.loader.num_workers,
                                           1, dist=False,
                                           shuffle=config.data.train.loader.shuffle)
    
    train_cfg = config.train_cfg
    train_cfg.dataloader = dataloader
    train_cfg.work_dir = args.work_dir
    test_cfg = config.test_cfg

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
        config.report_config,
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
