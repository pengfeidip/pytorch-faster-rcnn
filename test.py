import argparse
parser = argparse.ArgumentParser('Inference RetinaNet')
parser.add_argument('config', help='Model configs, train configs and test configs.')
parser.add_argument('ckpt', help='Model ckpt file ending with .pth')
parser.add_argument('--gpu', help='GPU cardinal, only support single GPU at now.')
parser.add_argument('--out', required=True, help='Output result in json format.')

args = parser.parse_args()

import os, sys, glob, random, logging, json
import os.path as osp
import mmcv, torch
from lib import datasets
from lib.tester import BasicTester
import torch, time


def check_args():
    args.config_file = osp.realpath(args.config)
    args.config = mmcv.Config.fromfile(args.config)
    out = osp.realpath(args.out)
    out_dir = osp.dirname(out)
    assert osp.exists(out_dir), 'Output directory does not exit: {}'.format(out_dir)
    args.out = out

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

def main():
    check_args()
    logging.basicConfig(format='%(asctime)s: %(message)s\t[%(levelname)s]',
                        datefmt='%y%m%d_%H%M%S_%a',
                        level=logging.DEBUG)

    config = args.config
    dataset = datasets.VOCDataset(
        ann_file=config.data.test.ann_file,
        img_prefix=config.data.test.img_prefix,
        pipeline=config.data.test.pipeline
    )
    dataloader = datasets.build_dataloader(dataset, config.data.test.imgs_per_gpu, config.data.test.loader.num_workers,
                                           1, dist=False, shuffle=config.data.test.loader.shuffle)
    device = torch.device('cpu')
    if args.gpu is not None:
        device = torch.device('cuda:{}'.format(args.gpu))

    from lib.registry import build_module
    model = build_module(config.model, train_cfg=config.train_cfg, test_cfg=config.test_cfg)
    model.to(device)

    tester = BasicTester(
        model,
        config.train_cfg,
        config.test_cfg,
        device)

    tester.load_ckpt(args.ckpt)
    start = time.time()
    infer_res = tester.inference(dataloader)

    anno_idx, out_json = 0, []
    for pred in infer_res:
        iid, bbox_xywh, score, category \
            = pred['image_id'], pred['bbox'], pred['score'], pred['category']
        for i, cur_bbox in enumerate(bbox_xywh):
            cur_pred = {
                'id': anno_idx,
                'image_id': iid,
                'bbox': [round(x.item(), 2) for x in cur_bbox],
                'score': round(score[i].item(), 3),
                'category_id': category[i].item()
            }
            out_json.append(cur_pred)
            anno_idx += 1
    json.dump(out_json, open(args.out, 'w'))
    time_used = time.time() - start
    print('Finished inference, time used: {} secs or {} mins'.format(time_used, time_used/60))
    
if __name__ == '__main__':
    main()
