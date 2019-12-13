from . import config
from PIL import Image

import torchvision as tv
import torch

# transform from PIL image to normalized tensor with shape [3 x H x W]
IMG_PIL2TENSOR = tv.transforms.Compose(
    [tv.transforms.ToTensor(),
     tv.transforms.Normalize(mean=config.IMGNET_MEAN, std=config.IMGNET_STD)])

# from image file to batched input to a network
def imread(file_path):
    img = Image.open(file_path)
    return IMG_PIL2TENSOR(img).unsqueeze(0)

def image2tensor(file_path):
    return imread(file_path)

def dict2str(d):
    return '{ '+', '.join(['{}:{}'.format(k, dict2str(v)) for k,v in d.items()])+' }' \
        if isinstance(d, dict) else str(d)

def index_of(bool_tsr):
    return tuple(torch.nonzero(bool_tsr).t())

def wh_from_xyxy(bbox):
    w,h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    return w,h

def bbox2param(base, bbox):
    """
    It calculates the relative distance of a bbox to a base bbox.

    Args:
        base (tensor): the base bbox, of size (4, n) where n is number of base bboxes
              and 4 is (x_min, y_min, x_max, y_max) or (left, top, right, bottom)
        bbox (tensor): the bbox 
    Returns:
        parameters representing the distance of bbox to base bbox (tx, ty, tw, th)
    """
    assert base.shape == bbox.shape
    base_w, base_h = wh_from_xyxy(base)
    bbox_w, bbox_h = wh_from_xyxy(bbox)
    tx, ty = (bbox[0]-base[0])/base_w, (bbox[1]-base[1])/base_h
    tw, th = torch.log(bbox_w/base_w), torch.log(bbox_h/base_h)
    return torch.stack([tx, ty, tw, th])

def param2bbox(base, param):
    """
    It calculates the bbox coordinates according to base bbox and parameters

    Args:
        base (tensor): the base bbox, of size (n, 4) where n is number of base bboxes
              and 4 is (x_min, y_min, x_max, y_max) or (left, top, right, bottom)
        param (tensor): parameters/distances (tx, ty, tw, th)
    Returns:
        the original bbox
    """
    assert base.shape == param.shape
    base_w, base_h = wh_from_xyxy(base)
    tx, ty, tw, th = [param[i] for i in range(4)]
    x, y = tx*base_w + base[0], ty*base_h + base[1]
    w, h = torch.exp(tw)*base_w, torch.exp(th)*base_h
    x2, y2 = x+w, y+h
    return torch.stack([x,y,x2,y2])
    
def calc_iou(a, b):
    """
    It calculates iou of a_i and b_j and put it in a table of size (N, K). N is number of bbox
    in a and K is number of bbox in b.

    Args:
        a (tensor): an array of bboxes, it's size is (4, N), 
                    4 coordinates: (x_min, y_min, x_max, y_max)
        b (tensor): another array of bboxes, it's size is (4, K)

    Returns:
        tensor: a table if ious

    """
    assert a.shape[0] == 4 and b.shape[0] == 4
    tl = torch.max(a[:2].view(2, -1, 1), b[:2].view(2, 1, -1))
    br = torch.min(a[2:].view(2, -1, 1), b[2:].view(2, 1, -1))
    area_i = torch.prod(br - tl, dim=0)
    area_i = area_i * (tl < br).all(dim=0).float()
    area_a = torch.prod(a[2:]-a[:2], dim=0)
    area_b = torch.prod(b[2:]-b[:2], dim=0)
    return area_i / (area_a.view(-1, 1) + area_b.view(1, -1) - area_i)

