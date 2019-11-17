import argparse

parser = argparse.ArgumentParser('Test or inference of a Faster RCNN detector.')
parser.add_argument('--config', required=True, metavar='REQUIRED',
                    help='Configuration file.')
parser.add_argument('--ckpt', required=True, metavar='REQUIRED', 
                    help="Checkpoint saved model, usually in '.pth' format.")
parser.add_argument('--out', required=True, metavar='REQUIRED', 
                    help='Output json file in coco format.')
args = parser.parse_args()

import os.path as osp
import mmcv

def check_args():
    args.config_file = osp.realpath(args.config)
    args.out = osp.realpath(args.out)
    assert not osp.isdir(args.out), 'Output should not be a directory: {}'.format(args.out)
    args.config = mmcv.Config.fromfile(args.config)

    out_dir = osp.dirname(args.out)
    assert osp.exists(out_dir) and osp.isdir(out_dir), 'Output directory does not exist: {}'.format(out_dir)
    

def main():
    check_args()
    print(args)
    

if __name__ == '__main__':
    main()
    

