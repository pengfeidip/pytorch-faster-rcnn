from mmdet.datasets import CocoDataset 
from mmdet.datasets import build_dataloader

VOC_CLASSES=(
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor'
)



class VOCDataset(CocoDataset):
    CLASSES=VOC_CLASSES

