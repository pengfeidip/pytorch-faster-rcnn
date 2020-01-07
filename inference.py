import argparse

parser = argparse.ArgumentParser('Test or inference of a Faster RCNN detector.')
parser.add_argument('--config', required=True, metavar='REQUIRED',
                    help='Configuration file.')
parser.add_argument('--ckpt', required=True, metavar='REQUIRED', 
                    help="Checkpoint saved model, usually in '.pth' format.")
parser.add_argument('--out', required=True, metavar='REQUIRED', 
                    help='Output json file in coco format.')
parser.add_argument('--gpu',
                    help='GPU to use.')
args = parser.parse_args()

import os.path as osp
import mmcv, random, torch, json, logging, json
from lib import data, faster_rcnn, data_

def load_image_info(JSON):
    cont = json.load(open(JSON))
    return {img['file_name']:img['id'] for img in cont['images']}

def check_args():
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
    logging.basicConfig(format='%(asctime)s: %(message)s\t[%(levelname)s]',
                        datefmt='%y%m%d_%H%M%S_%a',
                        level=logging.DEBUG)
    config = args.config
    data_opt = {'voc_data_dir':config.test_data_cfg.voc_data_dir}
    dataset = data_.TestDataset(data_opt)

    dataloader = torch.utils.data.DataLoader(dataset, **config.test_data_cfg.loader_cfg)
    device = torch.device('cuda:0')
    if args.gpu is not None:
        device = torch.device('cuda:{}'.format(args.gpu))
    
    tester = faster_rcnn.FasterRCNNTest(config.model, device = device)
    tester.load_ckpt(args.ckpt)
    infer_res = tester.inference(dataloader, config.test_cfg.min_score)
    
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
                'category_id': category[i]
            }
            out_json.append(cur_pred)
            anno_idx += 1
    json.dump(out_json, open(args.out, 'w'))

if __name__ == '__main__':
    main()
    

