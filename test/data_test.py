import sys, os
import os.path as osp
cur_dir = osp.dirname(osp.realpath(__file__))
sys.path.append(osp.join(cur_dir, '..'))
from lib import data
from lib import config
TEST_IMG_DIR = '/home/liyiqing/dataset/voc2007_trainval/VOC2007/JPEGImages/'
TEST_COCO_JSON = '/home/liyiqing/dataset/voc2007_trainval/voc2007_trainval.json'

def test_dataloader():
    dataset = data.CocoDetDataset(TEST_IMG_DIR, TEST_COCO_JSON,
                                  transform=data.faster_transform(
                                      1000, 600,
                                      config.VOC2007_MEAN,
                                      config.VOC2007_STD
                                  ))
    dataloader = data.torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=2,
        shuffle=False
    )
    for train_data in dataloader:
        img, bboxes, labels, info = train_data
        print('image shape:', img.shape)
        print('bboxes.shape:', bboxes.shape)
        print('bboxes:')
        print(bboxes)
        print('labels.shape:', labels.shape)
        print('info:', info)
        break

if __name__ == '__main__':
    test_dataloader()

