import mmcv, torch
import os.path as osp
import mmcv, random, torch, json, logging
from lib import data, faster_rcnn
import sys, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help='Config file')
parser.add_argument('--ckpt', required=True, help='Checkpoint to load')
parser.add_argument('--img-dir', required=True, help='Image dir')
parser.add_argument('--out', required=True, help='Output json')
parser.add_argument('--gpu', help='Which GPU to use')
parser.add_argument('--gt', help='Will add image ids to the result')
parser.add_argument('--min-score', help='Only keep result with min score')
parser.add_argument('--top', help='Only keep top results')

args = parser.parse_args()

def process_result(json_res):
    print('Result before process:', len(json_res))
    if args.min_score is not None:
        min_score = float(args.min_score)
        json_res = [anno for anno in json_res if anno['score'] >= min_score]
    if args.gt is not None:
        gt = json.load(open(args.gt))
        name2id = {img['file_name']:img['id'] for img in gt['images']}
        new_res = []
        for anno in json_res:
            if anno['file_name'] in name2id:
                anno['id'] = name2id[anno['file_name']]
            new_res.append(anno)
        json_res = new_res
    if args.top is not None:
        top = int(args.top)
        grouped = {anno['image_id']:[] for anno in json_res}
        new_res = []
        for anno in json_res:
            grouped[anno['image_id']].append(anno)
        for k, v in grouped.items():
            v.sort(key=lambda x:x['score'], reverse=True)
            new_res += v[:top]
        json_res = new_res
    return json_res

def main():
    logging.basicConfig(format='%(asctime)s: %(message)s\t[%(levelname)s]',
                        datefmt='%y%m%d_%H%M%S_%a',
                        level=logging.DEBUG)

    config = mmcv.Config.fromfile(args.config)
    test_data_cfg, train_data_cfg = config.test_data_cfg, config.train_data_cfg
    dataset = data.ImageDataset(args.img_dir,
                                transform=data.faster_transform(*train_data_cfg.img_size,
                                                                **train_data_cfg.img_norm))
    dataloader = torch.utils.data.DataLoader(dataset, **test_data_cfg.loader_cfg)
    device = torch.device('cpu')
    if args.gpu is not None:
        device = torch.device('cuda:{}'.format(args.gpu))
    tester = faster_rcnn.RPNTest(config.model, device=device)
    logging.info('Loading checkpoint: {}'.format(args.ckpt))
    tester.load_ckpt(args.ckpt)
    # 
    json_res = tester.inference(dataloader)
    json_res = process_result(json_res)
    json.dump(json_res, open(args.out, 'w'))
    logging.info('Output result to {}'.format(args.out))

if __name__ == '__main__':
    main()
