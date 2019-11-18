import argparse

parser = argparse.ArgumentParser('Test or inference of a Faster RCNN detector.')
parser.add_argument('--config', required=True, metavar='REQUIRED',
                    help='Configuration file.')
parser.add_argument('--ckpt', required=True, metavar='REQUIRED', 
                    help="Checkpoint saved model, usually in '.pth' format.")
parser.add_argument('--img-dir', required=True,
                    help='Test image directory.')
parser.add_argument('--out', required=True, metavar='REQUIRED', 
                    help='Output json file in coco format.')
parser.add_argument('--gpu',
                    help='GPU to use.')
parser.add_argument('--seed',
                    help='Seed for rng.')
args = parser.parse_args()

import os.path as osp
import mmcv, random, torch, json, logging
from lib import data, faster_rcnn

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

def check_args():
    args.img_dir = osp.realpath(args.img_dir)
    assert osp.exists(args.img_dir) and osp.isdir(args.img_dir), 'Can not find img-dir: {}'\
        .format(args.img_dir)
    args.config_file = osp.realpath(args.config)
    args.out = osp.realpath(args.out)
    assert not osp.isdir(args.out), 'Output should not be a directory: {}'.format(args.out)
    assert osp.exists(args.ckpt), 'Can not find ckpt file: {}'.format(args.ckpt)
    args.config = mmcv.Config.fromfile(args.config)

    out_dir = osp.dirname(args.out)
    assert osp.exists(out_dir) and osp.isdir(out_dir), \
        'Output directory does not exist: {}'.format(out_dir)

        
def main():
    check_args()
    if args.seed is not None:
        set_seed(args.seed)
    logging.basicConfig(format='%(asctime)s: %(message)s\t[%(levelname)s]',
                        datefmt='%y%m%d_%H%M%S_%a',
                        level=logging.DEBUG)
    config = args.config
    test_data_cfg, train_data_cfg = config.test_data_cfg, config.train_data_cfg
    dataset = data.ImageDataset(args.img_dir,
                                transform=data.faster_transform(*train_data_cfg.img_size,
                                                                **train_data_cfg.img_norm))
    dataloader = torch.utils.data.DataLoader(dataset, **test_data_cfg.loader_cfg)
    device = torch.device('cpu')
    if args.gpu is not None:
        device = torch.device('cuda:{}'.format(args.gpu))
    
    tester = faster_rcnn.FasterRCNNTest(config.model, device = device)
    tester.load_ckpt(args.ckpt)
    infer_res = tester.inference(dataloader, min_score=config.test_cfg.min_score)
    json.dump(infer_res, open(args.out, 'w'))
    

if __name__ == '__main__':
    main()
    

