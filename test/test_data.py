import sys, os
import os.path as osp
cur_dir = osp.dirname(osp.realpath(__file__))
sys.path.append(osp.join(cur_dir, '..'))
from lib import data
import time
from PIL import Image

TEST_DATA_DIR = '/home/server2/4T/liyiqing/dataset/PASCAL_VOC_07/'
TEST_IMG_DIR = osp.join(TEST_DATA_DIR, 'voc2007_trainval/VOC2007/JPEGImages')
TEST_COCO_JSON = osp.join(TEST_DATA_DIR, 'voc2007_trainval/voc2007_trainval.json')
TEST_IMG = osp.join(TEST_IMG_DIR, '009894.jpg')

# increase this will save loading time
NUM_WORKERS = 3

def test_resize():
    print('Test resizing'.center(90, '*'))
    resize = data.ResizeFasterInput(1000, 600)
    img = Image.open(TEST_IMG)
    print('Image before resize:', img.size)
    img = resize(img)
    print('Image after resize:', img.size)


def test_transform():
    print('Test BASIC_TRANSFORM'.center(90, '*'))
    print('Tested image:', TEST_IMG)
    img = Image.open(TEST_IMG)
    img_ = data.BASIC_TRANSFORM(img)
    print('Shape after BASIC_TRANSFORM:', img_.shape)
    print('Test faster RCNN transform'.center(90, '*'))
    trans = data.faster_transform(1000, 600)
    img_ = trans(img)
    print('Shape after faster RCNN transform:', img_.shape)

def test_dataset():
    print('Test CocoDetDataset'.center(90, '*'))
    dataset = data.CocoDetDataset(TEST_IMG_DIR, TEST_COCO_JSON)
    print('Dataset size:', len(dataset))
    print('The 0th data:', dataset[0])
    print('Data of image:', dataset[0][0].shape, dataset[0][0].dtype)
    print('Data of bboxes:', dataset[0][1].shape, dataset[0][1].dtype)


def test_dataloader():
    print('Test DataLoader'.center(90, '*'))
    dataset = data.CocoDetDataset(TEST_IMG_DIR, TEST_COCO_JSON)
    dataloader = data.torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=NUM_WORKERS,
        shuffle=False)
    start = time.time()
    for train_data in dataloader:
        img_data = train_data[0]
        bboxes_data = train_data[1]
    secs_used = time.time() - start
    print('Finished loading one batch using {} workers, time used: {} seconds.'\
          .format(NUM_WORKERS, secs_used))


if __name__ == '__main__':
    test_resize()
    test_transform()
    test_dataset()
    test_dataloader()
