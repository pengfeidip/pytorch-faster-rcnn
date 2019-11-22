import torch
import sys, os, warnings, json, glob
import os.path as osp
import torchvision as tv

from . import config
from PIL import Image


class ResizeFasterInput(object):
    r"""
    Resize input image to the size that is acceptable by a detector.
    After resize:
        the longer side is <= longer, 
        the shorter side is <= shorter,
        at least one equation is satisfied.
    """
    def __init__(self, longer=1000, shorter=600):
        assert longer >= shorter
        self.longer = longer
        self.shorter = shorter

    # in_data is an PIL.Image object
    def __call__(self, image):
        h, w = image.height, image.width
        for h_i, w_i in self._resize_all_(h, w):
            if self._is_satisfied_(h_i, w_i):
                h, w = h_i, w_i
        action = tv.transforms.Resize((h, w))
        return action(image)

    def _resize_all_(self, h, w):
        sizes = [self.longer, self.shorter]
        ret_val = []
        for sz in sizes:
            ret_val.append(self._fix_ratio_resize_(h, w, sz/h))
            ret_val.append(self._fix_ratio_resize_(h, w, sz/w))
        return ret_val

    def _fix_ratio_resize_(self, h, w, ratio):
        return [round(h*ratio), round(w*ratio)]

    def _is_satisfied_(self, h, w):
        longer = max(h, w)
        shorter = min(h, w)
        return \
            longer <= self.longer and \
            shorter <= self.shorter and \
            (longer == self.longer or shorter == self.shorter)


# ToTensor and Normalize only
BASIC_TRANSFORM = tv.transforms.Compose(
    [tv.transforms.ToTensor(),
     tv.transforms.Normalize(mean=config.IMGNET_MEAN, std=config.IMGNET_STD)])

# Resize input image to Faster RCNN acceptable sizes
def faster_transform(longer, shorter, mean=config.IMGNET_MEAN, std=config.IMGNET_STD):
    return tv.transforms.Compose([
        ResizeFasterInput(longer, shorter),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=mean, std=std)
    ])

class ImageDataset(torch.utils.data.Dataset):
    r"""
    This is for testing images
    """
    def __init__(self, img_dir, transform=BASIC_TRANSFORM):
        super(ImageDataset, self).__init__()
        self.img_dir = osp.realpath(img_dir)
        self.transform = transform
        self.imgs = glob.glob(osp.join(self.img_dir, '*'))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img_loc = self.imgs[i]
        img_name = osp.basename(img_loc)
        img = Image.open(img_loc)
        img_w, img_h = img.width, img.height
        img_resized = self.transform(img)
        h2, w2 = img_resized.shape[-2:]
        w_amp, h_amp = w2/img_w, h2/img_h
        return img_resized, w_amp, img_name, img_w, img_h

class CocoDetDataset(torch.utils.data.Dataset):
    r"""
    Only loads detection related data from a coco formatted data.
    It returns an image and bboxes in that image
    """
    def __init__(self, img_dir, anno_json, transform=BASIC_TRANSFORM):
        super(CocoDetDataset, self).__init__()
        self.img_dir = img_dir
        self.anno_json = anno_json
        self.anno = json.load(open(anno_json))
        self.transform = transform

        img_files = set([item for item in os.listdir(self.img_dir) \
                         if not osp.isdir(osp.join(self.img_dir, item))])
        # a image_id to file_name mapping
        anno_imgs = {i['id']:i['file_name'] for i in self.anno['images']}
        not_found = [fname for iid, fname in anno_imgs.items() \
                     if fname not in img_files]
        if len(not_found) != 0:
            raise ValueError('Can not find {} images that are in json '
                             'but not in the img_dir: {}'\
                             .format(len(not_found), self.img_dir))
        # a image_id to bboxes mapping
        anno_bboxes = {}
        for annotation in self.anno['annotations']:
            iid = annotation['image_id']
            bbox = annotation['bbox']
            cid = annotation['category_id']
            bbox.append(cid)
            if iid in anno_bboxes:
                anno_bboxes[iid].append(bbox)
            else:
                anno_bboxes[iid] = [bbox]

        not_found = [iid for iid in anno_bboxes if iid not in anno_imgs]
        if len(not_found) != 0:
            raise ValueError('Can not find {} images that are in annotations '
                             'but not in images, '
                             'this may cause training interuption'.format(len(not_found)))
                
        self.anno_bboxes = anno_bboxes
        self.anno_imgs = anno_imgs
        self.img_ids = list(anno_imgs.keys())
        
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, i):
        # for debug: train one image
        # i = 1234
        iid = self.img_ids[i]
        fname = self.anno_imgs[iid]
        img_loc = osp.join(self.img_dir, fname)
        img = Image.open(img_loc)
        img_w, img_h = img.width, img.height
        img_resized = self.transform(img)
        h2, w2 = img_resized.shape[-2:]
        w_amp = w2 / img_w
        h_amp = h2 / img_h
        bboxes_data = self.anno_bboxes[iid]
        trans_bboxes = []
        for bbox in bboxes_data:
            x,y,w,h = bbox[:4]
            trans_bboxes.append([round(x*w_amp), round(y*h_amp), round(w*w_amp),
                                 round(h*h_amp), bbox[-1]])
        bboxes_data = torch.tensor(trans_bboxes)
        return img_resized, bboxes_data, {'iid':iid,
                                       'file_name':fname,
                                       'original_tensor_wh':(img_w,img_h),
                                       'img_loc': img_loc}
        
