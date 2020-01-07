import sys, os, json, time
import os.path as osp
cur_dir = osp.dirname(osp.realpath(__file__))
sys.path.append(osp.join(cur_dir, '..'))
from lib import data, data_
from lib import config
TEST_IMG_DIR = '/home/liyiqing/dataset/voc2007_trainval/VOC2007/JPEGImages/'
TEST_COCO_JSON = '/home/liyiqing/dataset/voc2007_trainval/voc2007_trainval.json'

def test_dataloader():
    opt = {'voc_data_dir':'/home/server2/4T/liyiqing/dataset/PASCAL_VOC_07/voc2007_all/VOC2007'}
    dataset = data_.TestDataset(opt)
    res_json = []
    sz = len(dataset)
    print('number of test images:', len(dataset))
    start = time.time()
    for i in range(sz):
        img, img_size, bbox, label, difficult, iid = dataset[i]
        iid = int(iid)
        for j, cur_bbox in enumerate(bbox):
            y_min,x_min,y_max,x_max = cur_bbox.tolist()
            res_json.append({
                'image_id': iid,
                'bbox': [x_min, y_min, x_max-x_min, y_max-y_min],
                'category_id': int(label[j])
            })
        if i % 500 == 0:
            print('finished ', i)
    json.dump(res_json, open('voc2007_test.json', 'w'))

if __name__ == '__main__':
    test_dataloader()

