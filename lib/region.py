import torch
import torch.nn as nn
import copy, random
from . import config



class BBox(object):
    r"""
    Represent a bounding box, it accepts various input and convert to xywh inside.
    It also provides conversions to various formats, the conversion is simply applying 
    conversion formula and does not do round up, so it is up to the users to do round up 
    before or after the conversion if users want integer results. 
    x, y here is upper left corner of the box.
    x, y in center_xywh is coordinates of the center
    """
    def __init__(self, xywh=None, xyxy=None, xxyy=None, center_xywh=None):
        self.center_xywh = center_xywh
        if xywh is not None:
            assert len(xywh) == 4
            self.xywh = xywh
        elif xyxy is not None:
            assert len(xyxy) == 4
            x1,y1,x2,y2 = xyxy
            self.xywh = (x1, y1, x2-x1, y2-y1)
        elif xxyy is not None:
            assert len(xxyy) == 4
            x1,x2,y1,y2 = xxyy
            self.xywh = (x1, y1, x2-x1, y2-y1)
        elif center_xywh is not None:
            assert len(center_xywh) == 4
            cx,cy,w,h = center_xywh
            self.xywh = (cx - w/2, cy - h/2, w, h)
            self.center_xywh = center_xywh
        else:
            raise ValueError('No coordinates provided to __init__ of bbox.')

    def xywh2xyxy(self, xywh):
        x,y,w,h = xywh
        return (x,y,x+w,y+h)
    def xywh2xxyy(self, xywh):
        x,y,w,h = xywh
        return (x,x+w,y,y+h)
    def get_xywh(self):
        return self.xywh
    def get_xyxy(self):
        return self.xywh2xyxy(self.xywh)
    def get_xxyy(self):
        return self.xywh2xxyy(self.xywh)
    def get_center_xywh(self):
        if self.center_xywh is not None:
            return self.center_xywh
        x,y,w,h = self.xywh
        return (x+w/2, y+h/w, w, h)
    def is_valid(self):
        x,y,w,h = self.xywh
        if x>=0 and y>=0 and w>=0 and h>=0:
            return True
        return False
    def round_xywh(self):
        return tuple((round(x) for x in self.get_xywh()))
    def round_center_xywh(self):
        return tuple((round(x) for x in self.get_center_xywh()))
    def contain_point(self, x, y):
        xx, yy, ww, hh = self.get_xywh()
        return x>=xx and x<=xx+ww and y>=yy and y<=yy+hh
    def get_area(self):
        x, y, w, h = self.get_xywh()
        return w * h
        
    def __str__(self):
        return 'xywh:'+str(self.get_xywh())

# Provide an abstraction of a point 2D space so that users do not get confused whether to (w,h) or (x,y) notation.
class Point(object):
    def __init__(self, x=None, y=None, w=None, h=None):
        if x is not None and y is not None:
            self.x, self.y, self.w, self.h = x, y, x, y
        elif w is not None and h is not None:
            self.x, self.y, self.w, self.h = w, h, w, h
        else:
            raise ValueError('Provide valid values for Point class.')
    def __str__(self):
        return 'xy:{}'.format((self.x, self.y))
    
# a and b are BBox objects
def calc_iou(a, b):
    pts = []
    ax1, ay1, ax2, ay2 = a.get_xyxy()
    if b.contain_point(x=ax1, y=ay1):
        pts.append((ax1, ay1))
    if b.contain_point(x=ax2, y=ay2):
        pts.append((ax2, ay2))
    bx1, by1, bx2, by2 = b.get_xyxy()
    if a.contain_point(x=bx1, y=by1):
        pts.append((bx1, by1))
    if a.contain_point(x=bx2, y=by2):
        pts.append((bx2, by2))
    if len(pts)<2:
        return 0
    p1, p2 = pts[:2]
    overlap_area = abs(p1[0]-p2[0]) * abs(p1[1]-p2[1])
    return overlap_area / (a.get_area() + b.get_area() - overlap_area)

class GroundTruth(object):
    r"""
    Input is a tensor of (1, n, 5) that contains n ground truth bboxes in an image 
    """
    def __init__(self, bboxes, iid=None):
        bboxes = bboxes.squeeze(0)
        self.iid = iid
        self.bboxes = []
        self.categories = []
        for bbox in bboxes:
            cate = bbox[-1]
            self.categories.append(cate)
            self.bboxes.append(BBox(xywh=tuple([x.item() for x in bbox[:4]])))
    
class AnchorGenerator(object):
    r"""
    Generate a list of BBox objects that represents anchors of a certain setting.
    """
    def __init__(self, scales, aspect_ratios):
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.aspect_ratios_sqrt = [x**0.5 for x in aspect_ratios]
        
    def anchors(self, img_size, grid, allow_cross=False):
        ret_list = []
        assert len(img_size)==2 and len(grid)==2
        assert img_size[0]>=0 and img_size[1]>=0 and grid[0]>=0 and grid[1]>=0
        h_img,  w_img  = img_size
        h_grid, w_grid = grid
        grid_dist_h, grid_dist_w = h_img/h_grid, w_img/w_grid
        idx = 0
        for i in range(h_grid):
            for j in range(w_grid):
                i_center = grid_dist_h/2 + grid_dist_h*i
                j_center = grid_dist_w/2 + grid_dist_w*j
                for scale in self.scales:
                    for ar_sqrt in self.aspect_ratios:
                        anchor = {
                            'center': Point(y=i_center, x=j_center),
                            'feat_loc': Point(y=i, x=j),
                        }
                        anchor_h = scale / ar_sqrt
                        anchor_w = scale * ar_sqrt
                        bbox = BBox(center_xywh=(j_center,
                                                 i_center,
                                                 anchor_w,
                                                 anchor_h))
                        x1,y1,x2,y2 = bbox.get_xyxy()
                        # get rid of cross boundary anchors
                        if allow_cross or (x1>=0 and x2>=0 and y1>=0 and y2>=0 and x1<w_img and x2<w_img and y1<h_img and y2<h_img):
                            anchor['bbox'] = bbox
                            anchor['id'] = idx
                            yield anchor
                            idx += 1
    def anchors_list(self, img_size, grid, allow_cross=False):
        return list(self.anchors(img_size, grid, allow_cross))
                            
class AnchorTargetCreator(object):
    r"""
    Given ground truth bboxes and a set of anchors, find 256 training targets for RPN network.
    Anchor has following members: center, feat_loc, bbox and id.
    """
    def __init__(self, img_size, grid, anchor_generator, ground_truth, pos_iou=0.7,
                 neg_iou=0.3, max_pos=128, max_targets=256, allow_cross=False):
        self.img_size = img_size
        self.grid = grid
        self.anchor_generator = anchor_generator
        self.ground_truth = ground_truth
        self.pos_iou = pos_iou
        self.neg_iou = neg_iou
        self.max_pos = max_pos
        self.max_targets = max_targets
        self.allow_cross = allow_cross

    def targets(self):
        anchors = [x for x in self.anchor_generator.anchors(self.img_size, self.grid, self.allow_cross)]
        all_anchor_ids = [x['id'] for x in anchors]
        # find positive targets for training
        pos_targets = []
        for gt_bbox, category in zip(self.ground_truth.bboxes, self.ground_truth.categories):
            max_iou = -1
            max_anchor = None
            large_iou_anchors = []
            for anchor in anchors:
                iou = calc_iou(gt_bbox, anchor['bbox'])
                xx, yy, ww, hh = anchor['bbox'].get_xywh()
                if iou > max_iou:
                    max_iou = iou
                    max_anchor = anchor
                if iou >= self.pos_iou:
                    large_iou_anchors.append([iou, anchor])
            # first add the anchor of max iou to the positive target list
            pos_targets.append({
                'anchor': max_anchor,
                'gt_bbox': gt_bbox,
                'gt_label': 1,
                'category': category,
                'iou': max_iou
            })
            # sort the anchors of iou larger than pos_iou by iou
            large_iou_anchors = sorted(large_iou_anchors, key=lambda x: x[0], reverse=True)
            # look down the list and add anchor that is not the max anchor to the positive target list
            for iou, anchor in large_iou_anchors:
                if anchor['id'] != max_anchor['id']:
                    pos_targets.append({
                        'anchor': anchor,
                        'gt_bbox': gt_bbox,
                        'gt_label': 1,
                        'category': category,
                        'iou': iou})
                    break
        # limit the positive targets to max_pos
        pos_targets = pos_targets[:self.max_pos]
        pos_anchor_ids = set((x['anchor']['id'] for x in pos_targets))
        # to find negative targets for training
        max_neg = self.max_targets - len(pos_targets)
        neg_targets = []
        # shuffle the anchors so that selection of negative targets are random
        random.shuffle(anchors)
        for anchor in anchors:
            if len(neg_targets) >= max_neg:
                break
            small_iou = True
            for gt_bbox in self.ground_truth.bboxes:
                iou = calc_iou(gt_bbox, anchor['bbox'])
                if iou > self.neg_iou:
                    small_iou = False
                    break
            if small_iou and anchor['id'] not in pos_anchor_ids:
                neg_targets.append({
                    'anchor': anchor,
                    'gt_bbox': None,
                    'gt_label': 0,
                    'category': None,
                    'iou': None
                })
        return pos_targets + neg_targets
            
                            
class ROIPooling(nn.Module):
    r"""
    Accepts a list of crops from the feature map of various spatial size(same channels)
    and output a batched feature map of the same spatial size(e.g. 7 x 7).
    It uses nn.AdaptiveMaxPool2d as the ROI pooling operator.
    """
    def __init__(self, output_size):
        super(ROIPooling, self).__init__()
        self.output_size = output_size
        self.adaptive_pool \
            = nn.AdaptiveMaxPool2d(output_size)
    # rois is a list of roi, which has shape like [1, 512, 26, 32]
    def forward(self, rois):
        batch_size = len(rois)
        outs = [self.adaptive_pool(roi) for roi in rois]
        return torch.cat(outs)
                                                                        

#####################################################################################
# Apparently ROI pooling operator is already implemented in Pytorch, it is          #
# the Module: torch.nn.AdaptiveMaxPool2d(output_size, return_indices=False).        #
# It defines the bounds of i-th bin slightly different than in the SPPnet paper.    #
# However, the adaptive max pool's way seems more intuitive.                        #
#                                                                                   #
# NOTE: the following ROI pooling utilities are not implemented                     #
#####################################################################################
class RoiPoolOp(torch.autograd.Function):
    r"""
    Defines ROI pooling operator as an autograd.Function.
    Roi is a tensor of at least 2 dimensions, the pooling takes place in the
    last two dimensions.
    It pools the last two dimensions into a fixed sized tensor.
    The bounds of i-th bin follows the following formula:
        [floor((i-1)/n*h), ceiling(i/n*h)] where h is input size and n is
        output size(num of bins)
    """
    def __init__(self, output_size):
        super(RoiPoolOp, self).__init_()
        self.output_size = output_size
        
    @staticmethod
    def forward(ctx, roi):
        # unfinished
        pass

    @staticmethod
    def backward(ctx, grad_output):
        # unfinished
        pass

# not finished
def roi_pool(roi, out_size):
    assert len(out_size) == 2
    h, w = roi.shape[-2:]
    h_out, w_out = out_size
    # the SPPnet paper uses ceiling at right end, but using floor seems more intuitive
    # in that it seperates different bins better.
    h_bounds = [[int((i-1)/h_out*h), min(int(i/h_out*h), h-1)] for i in range(1, h_out+1)]
    w_bounds = [[int((i-1)/w_out*w), min(int(i/w_out*w), w-1)] for i in range(1, w_out+1)]
    print('h_bounds:')
    for x in h_bounds:
        print(x, x[1]-x[0]+1)
    print('w_bounds:')
    for x in w_bounds:
        print(x, x[1]-x[0]+1)

####################################################################################
####################################################################################
####################################################################################

