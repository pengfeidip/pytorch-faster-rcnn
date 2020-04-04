from PIL import Image

import torchvision as tv
import torch

IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD  = [0.229, 0.224, 0.225]

# transform from PIL image to normalized tensor with shape [3 x H x W]
IMG_PIL2TENSOR = tv.transforms.Compose(
    [tv.transforms.ToTensor(),
     tv.transforms.Normalize(mean=IMGNET_MEAN, std=IMGNET_STD)])

# from image file to batched input to a network
def imread(file_path):
    img = Image.open(file_path)
    return IMG_PIL2TENSOR(img).unsqueeze(0)

def image2tensor(file_path):
    return imread(file_path)

def dict2str(d):
    return '{ '+', '.join(['{}:{}'.format(k, dict2str(v)) for k,v in d.items()])+' }' \
        if isinstance(d, dict) else str(d)

def image2feature(bbox, img_size, feat_size):                                                                             
    h_rat, w_rat = [feat_size[i]/img_size[i] for i in range(2)]
    return bbox * torch.tensor([[w_rat], [h_rat], [w_rat], [h_rat]],
                               device=bbox.device, dtype=torch.float32)

def index_of(bool_tsr):
    return tuple(torch.nonzero(bool_tsr).t())

def wh_from_xyxy(bbox):
    w,h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    return w,h

def center_of(bbox):
    return (bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2

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
    base_center = center_of(base)
    bbox_center = center_of(bbox)
    tx, ty = (bbox_center[0]-base_center[0])/base_w, (bbox_center[1]-base_center[1])/base_h
    #tx, ty = (bbox[0]-base[0])/base_w, (bbox[1]-base[1])/base_h
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
    base_center_x, base_center_y = center_of(base)
    tx, ty, tw, th = [param[i] for i in range(4)]
    center_x, center_y = tx*base_w + base_center_x, ty*base_h + base_center_y
    w, h = torch.exp(tw)*base_w, torch.exp(th)*base_h
    return torch.stack([center_x-w/2,
                        center_y-h/2,
                        center_x+w/2,
                        center_y+h/2])

def xyxy2xywh(xyxy):
    return torch.stack([xyxy[0], xyxy[1], xyxy[2]-xyxy[0], xyxy[3]-xyxy[1]])
def xywh2xyxy(xywh):
    return torch.stack([xywh[0], xywh[1], xywh[0]+xywh[2], xywh[1]+xywh[3]])
    
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
    area_i = torch.prod(br - tl + 1, dim=0)
    area_i = area_i * (tl < br).all(dim=0).float()
    area_a = torch.prod(a[2:]-a[:2] + 1, dim=0)
    area_b = torch.prod(b[2:]-b[:2] + 1, dim=0)
    return area_i / (area_a.view(-1, 1) + area_b.view(1, -1) - area_i)

def elem_iou(a, b):
    assert a.shape[0]==4 and b.shape[0]==4 and a.shape == b.shape
    tl = torch.max(a[:2], b[:2])
    br = torch.min(a[2:], b[2:])
    area_i = torch.prod(br-tl, dim=0)
    area_i = area_i * (tl<br).all(0).float()
    area_a = torch.prod(a[2:]-a[:2], dim=0)
    area_b = torch.prod(b[2:]-b[:2], dim=0)
    return area_i / (area_a + area_b - area_i)
    

def init_module_normal(m, mean=0.0, std=1.0):
    for name, param in m.named_parameters():
        if 'weight' in name:
            param.data.normal_(mean, std)
        if 'bias' in name:
            param.data.zero_()

# label has -1:ignore, 0:negative, >0:specific positive index
# turn positive values into uniformly 1
def simplify_label(label):
    label_ = label.clone().detach()
    label_[label>0]=1
    return label_

def to_pair(val):
    if isinstance(val, int):
        pair=tuple([val, val])
    else:
        val = list(val)
        assert len(val) == 2
        pair = tuple(val)
    return pair

def multiclass_nms(bbox, score, label, label_set, nms_iou, min_score):
    label_set = list(label_set)

    nms_bbox, nms_score, nms_label = [], [], []
    for cur_label in label_set:
        chosen = (label==cur_label)
        if chosen.sum()==0:
            continue
        cur_bbox = bbox[:, chosen]
        cur_score = score[chosen]
        non_small = cur_score > min_score
        cur_score = cur_score[non_small]
        cur_bbox = cur_bbox[:, non_small]
        keep = tv.ops.nms(cur_bbox.t(), cur_score, nms_iou)
        nms_score.append(cur_score[keep])
        nms_bbox.append(cur_bbox[:, keep])
        nms_label.append(
            label.new_full((len(keep), ), cur_label)
        )

    if len(nms_bbox) != 0:
        return torch.cat(nms_bbox, 1), torch.cat(nms_score), torch.cat(nms_label)
    return nms_bbox, nms_score, nms_label


def one_hot_embedding(label, n_cls):
    n = len(label)
    one_hot = label.new_full((n, n_cls), 0)
    one_hot[torch.arange(n), label] = 1
    return one_hot

def multi_apply(*args, func):
    list_args = [arg for arg in args if isinstance(arg, list)]
    if len(list_args) == 0:
        return func(*args)
    mult_arg_len = len(list_args[0])
    for arg in list_args:
        if len(arg) != mult_arg_len:
            raise ValueError('Arg: {} does not have the same length as others'.format(arg))
    result = []
    for i in range(mult_arg_len):
        cur_args = []
        for arg in args:
            if isinstance(arg, list):
                cur_args.append(arg[i])
            else:
                cur_args.append(arg)
        result.append(func(*cur_args))
    return result
    
