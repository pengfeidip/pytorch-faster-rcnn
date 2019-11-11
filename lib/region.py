import torch
import torch.nn as nn
import copy, random
import numpy as np
from . import config, utils



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
    def area(self):
        x, y, w, h = self.get_xywh()
        return w * h
        
    def __str__(self):
        return 'BBox:xywh'+'({})'.format(', '.join([str(round(x, 2)) for x in self.get_xywh()]))

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
        return 'xy:{}'.format((round(self.x, 2), round(self.y, 2)))
    
# previous wrong version
def calc_iou_v2(a, b):
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
    return overlap_area / (a.area() + b.area() - overlap_area)

# a and b are BBox objects
def calc_iou(a, b):
    bboxes = [a, b]
    x_coor, y_coor = [], []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox.get_xyxy()
        x_coor += [[x1, bbox], [x2, bbox]]
        y_coor += [[y1, bbox], [y2, bbox]]
    x_coor.sort(key=lambda x: x[0])
    y_coor.sort(key=lambda x: x[0])
    if x_coor[0][1] == x_coor[1][1] or y_coor[0][1] == y_coor[1][1]:
        return 0
    overlap = abs(x_coor[1][0] - x_coor[2][0]) * abs(y_coor[1][0] - y_coor[2][0])
    union = a.area() + b.area() - overlap
    if union == 0:
        print('WARNING: Encounter bboxes with zero union area. {}, {}'.format(a, b))
        return 0
    return overlap / union
        

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
            self.categories.append(cate.item())
            self.bboxes.append(BBox(xywh=tuple([x.item() for x in bbox[:4]])))
    
class AnchorGenerator(object):
    r"""
    Generate a list of BBox objects that represents anchors of a certain setting.
    """
    def __init__(self, scales, aspect_ratios, allow_cross=False):
        self.allow_cross = allow_cross
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.aspect_ratios_sqrt = [x**0.5 for x in aspect_ratios]
        
    def anchors(self, img_size, grid):
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
                for scale_idx, scale in enumerate(self.scales):
                    for ar_idx, ar in enumerate(self.aspect_ratios):
                        anchor = {
                            'center': Point(y=i_center, x=j_center),
                            'feat_loc': Point(y=i, x=j),
                            'scale_idx': scale_idx,
                            'ar_idx': ar_idx
                        }
                        anchor_h = scale / ar
                        anchor_w = scale * ar
                        bbox = BBox(center_xywh=(j_center,
                                                 i_center,
                                                 anchor_w,
                                                 anchor_h))
                        x1,y1,x2,y2 = bbox.get_xyxy()
                        # get rid of cross boundary anchors
                        if self.allow_cross or \
                           (x1>=0 and x2>=0 and y1>=0 and y2>=0 and \
                            x1<w_img and x2<w_img and y1<h_img and y2<h_img):
                            anchor['bbox'] = bbox
                            anchor['id'] = idx
                            yield anchor
                            idx += 1
    def anchors_list(self, img_size, grid):
        return list(self.anchors(img_size, grid))
                            
class AnchorTargetCreator(object):
    r"""
    Given ground truth bboxes and a set of anchors, find 256 training targets 
    for RPN network.
    Anchor has following members: center, feat_loc, bbox and id.
    """
    def __init__(self, anchor_generator, pos_iou=0.7,
                 neg_iou=0.3, max_pos=128, max_targets=256):
        self.anchor_generator = anchor_generator
        self.pos_iou = pos_iou
        self.neg_iou = neg_iou
        self.max_pos = max_pos
        self.max_targets = max_targets

    def targets(self, img_size, grid, ground_truth):
        anchors = [x for x in self.anchor_generator.anchors(img_size, grid)]
        all_anchor_ids = [x['id'] for x in anchors]
        # find positive targets for training
        pos_targets = []
        
        for gt_bbox, category in zip(ground_truth.bboxes, ground_truth.categories):
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
            # look down the list and add anchor that is not the max anchor to
            # the positive target list
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
        # record positive anchors so that they do not get to be choosen as negative later.
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
            for gt_bbox in ground_truth.bboxes:
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


# the following provides convertion btw parameters and bbox coordinates
# it turns out neither is involved in backpropagation, so they can be
# implemented in numpy
def xywh2param(xywh, anchor_bbox):
    x, y, w, h = [float(p) for p in xywh]
    x_a, y_a, w_a, h_a = anchor_bbox.get_xywh()
    return [(x-x_a)/w_a, (y-y_a)/h_a, np.log(w/w_a), np.log(h/h_a)]
def param2xywh(param, anchor_bbox):
    x_p, y_p, w_p, h_p = [float(p) for p in param]
    x_a, y_a, w_a, h_a = anchor_bbox.get_xywh()
    return [x_p*w_a+x_a, y_p*h_a+y_a, np.exp(w_p)*w_a, np.exp(h_p)*h_a]


# TODO: improve performance
def apply_nms(anchors, score_map, bbox_map, iou_thr):
    num_anchors = len(anchors)
    deleted = [0 for i in range(num_anchors)]
    anchors.sort(key = score_map, reverse=True)
    for i, anchor in enumerate(anchors):
        if not deleted[i]:
            cur_bbox = bbox_map(anchor)
            for j in range(i+1, num_anchors):
                if not deleted[j] and \
                   calc_iou(cur_bbox, bbox_map(anchors[j]))>=iou_thr:
                    deleted[j] = 1
    return [anchors[i] for i, de in enumerate(deleted) if not de]

class ProposalCreator(object):
    r"""
    Given a RPN classifier output, a RPN regressor output and a anchor generator,
    1, calculate anchor objectness
    2, adjust anchor position with regressed parameters
    3, choose top N(12000) anchors
    4, apply NMS and choose top M(2000)

    After this, anchor will add members: 
        'obj_score': int,
        'adj_bbox': BBox,
        'objectness': torch.tensor # for backprop
        'adjustment': torch.tensor # for backprop
    """
    def __init__(self, anchor_generator):
        self.anchor_generator = anchor_generator

    def proposals(self, rpn_cls_res, rpn_reg_res, img_size, grid):
        grid = tuple(grid)
        cls_res_size = tuple([rpn_cls_res.shape[-2], rpn_cls_res.shape[-1]])
        reg_res_size = tuple([rpn_reg_res.shape[-2], rpn_reg_res.shape[-1]])
        assert grid == cls_res_size and grid == reg_res_size
        anchors = list(self.anchor_generator.anchors(img_size, grid))
        # put score and adjusted bbox to anchors
        # it is convenient that anchors are defined as a dictionaries
        num_scales = len(self.anchor_generator.scales)
        num_ars = len(self.anchor_generator.aspect_ratios)
        for anchor in anchors:
            feat_loc = anchor['feat_loc']
            anchor_bbox = anchor['bbox']
            feat_loc_i, feat_loc_j = feat_loc.y, feat_loc.x
            scale_idx, ar_idx = anchor['scale_idx'], anchor['ar_idx']
            anchor_idx = scale_idx * num_ars + ar_idx
            # get objectness score
            objectness = rpn_cls_res[0, anchor_idx*2:anchor_idx*2 + 2,
                                     feat_loc_i, feat_loc_j]
            adjustment = rpn_reg_res[0, anchor_idx*4:anchor_idx*4 + 4,
                                   feat_loc_i, feat_loc_j]
            # need to apply softmax to get the real score
            obj_score = torch.softmax(objectness, dim=0)[1].item()
            adj_bbox  = param2xywh(adjustment, anchor_bbox)

            # next put extra information to anchor
            anchor['obj_score'] = obj_score
            anchor['adj_bbox']  = BBox(xywh=adj_bbox)
            anchor['objectness'] = objectness
            anchor['adjustment'] = adjustment
        return anchors

    def proposals_filtered(self, rpn_cls_res, rpn_reg_res, img_size, grid,
                           max_by_score, max_after_nms, nms_iou):
        assert max_by_score > 0 and max_after_nms > 0
        props = self.proposals(rpn_cls_res, rpn_reg_res, img_size, grid)
        props.sort(key=lambda x: x['obj_score'], reverse=True)
        props = props[:max_by_score]
        props_nms = apply_nms(
            props,
            score_map=lambda x: x['obj_score'],
            bbox_map=lambda x: x['bbox'],
            iou_thr=nms_iou)
        return props_nms[:max_after_nms]
    
class ProposalTargetCreator(object):
    r"""
    From selected ROIs(around 2000, by ProposalCreator), choose 128 samples for training Head.
    """
    def __init__(self, max_pos=32, max_targets=128, pos_iou=0.5, neg_iou=0.1):
        self.max_pos = max_pos
        self.max_targets = max_targets
        self.pos_iou = pos_iou
        self.neg_iou = neg_iou

    # proposal keys: 'bbox', 'center', 'feat_loc', 'scale_idx', 'ar_idx', 'id', 'obj_score',
    # 'adj_bbox', 'objectness', 'adjustment'
    def targets(self, proposals, gt):
        print('Number of proposals:', len(proposals))
        print(proposals[0])
        pos_targets = []
        neg_targets = []
        for gt_bbox, category in zip(gt.bboxes, gt.categories):
            for prop in proposals:
                adj_bbox = prop['adj_bbox']
                iou = calc_iou(adj_bbox, gt_bbox)
                if iou >= self.pos_iou:
                    prop['gt_bbox'] = gt_bbox
                    prop['gt_label'] = 1
                    prop['category'] = category
                    prop['iou'] = iou
                    pos_targets.append(prop)
                elif iou <= self.neg_iou:
                    prop['gt_bbox'] = None
                    prop['gt_label'] = 0
                    prop['category'] = 0
                    prop['iou'] = iou
                    neg_targets.append(prop)
        print('num pos_targets:', len(pos_targets))
        print('num neg_targets:', len(neg_targets))

        pos_targets = []
        neg_targets = []
        for prop in proposals:
            max_iou = -1
            for gt_bbox, category in zip(gt.bboxes, gt.categories):
                adj_bbox = prop['adj_bbox']
                iou = calc_iou(adj_bbox, gt_bbox)
                if iou >= self.pos_iou:
                    prop['gt_bbox'] = gt_bbox
                    prop['gt_label'] = 1
                    prop['category'] = category
                    prop['iou'] = iou
                    pos_targets.append(prop)
                max_iou = max(max_iou, iou)
            if max_iou <= self.neg_iou:
                prop['gt_bbox'] = None
                prop['gt_label'] = 0
                prop['category'] = 0
                prop['iou'] = iou
                neg_targets.append(prop)
            
        print('num pos_targets:', len(pos_targets))
        print('num neg_targets:', len(neg_targets))
        pass
    

        
                            
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

