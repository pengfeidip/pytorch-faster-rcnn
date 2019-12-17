import torch, torchvision
import torch.nn as nn
import copy, random, logging
import numpy as np
import time, sys, os
import os.path as osp
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
        return 'BBox:xywh'+'({})'.format(', '.join([str(round(x, 2)) \
                                                    for x in self.get_xywh()]))

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
    An anchor is a dict that has members: 
        'center', 'feat_loc', 'scale_idx', 'ar_idx', 'bbox', 'id'
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
                    for ar_idx, ar in enumerate(self.aspect_ratios_sqrt):
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
        if len(anchors) == 0:
            logging.warning('Encounter 0 anchors for image (img_size: {})'\
                            .format(img_size))
            return None
        num_anchors, num_gts = len(anchors), len(ground_truth.bboxes)
        iou_tab = np.zeros((num_anchors, num_gts), dtype=np.float16)
        for anchor_idx, anchor in enumerate(anchors):
            for gt_idx, gt_bbox in enumerate(ground_truth.bboxes):
                iou_tab[anchor_idx][gt_idx] = calc_iou(anchor['bbox'], gt_bbox)
        max_anchor = iou_tab.argmax(0)
        max_gt = iou_tab.argmax(1)

        chosen_pos = set(max_anchor)
        chosen_neg = set()

        for anchor_idx, gt_idx in enumerate(max_gt):
            max_iou = iou_tab[anchor_idx][gt_idx]
            if max_iou >= self.pos_iou:
                chosen_pos.add(anchor_idx)
            elif max_iou < self.neg_iou:
                chosen_neg.add(anchor_idx)
            else:
                pass

        chosen_pos = list(chosen_pos)
        chosen_neg = list(chosen_neg)
        random.shuffle(chosen_pos)
        random.shuffle(chosen_neg)
        chosen_pos = chosen_pos[:self.max_pos]
        chosen_neg = chosen_neg[:self.max_targets - len(chosen_pos)]
        pos_targets = []
        neg_targets = []
        for anchor_idx in chosen_pos:
            max_gt_idx = max_gt[anchor_idx]
            max_iou = iou_tab[anchor_idx][max_gt_idx]
            pos_targets.append({
                'anchor': anchors[anchor_idx],
                'gt_bbox': ground_truth.bboxes[max_gt_idx],
                'gt_label': 1,
                'category': ground_truth.categories[max_gt_idx],
                'iou': max_iou
            })
        for anchor_idx in chosen_neg:
            max_gt_idx = max_gt[anchor_idx]
            max_iou = iou_tab[anchor_idx][max_gt_idx]
            neg_targets.append({
                'anchor': anchors[anchor_idx],
                'gt_bbox': None,
                'gt_label': 0,
                'category': None,
                'iou': max_iou
            })
            
        logging.info('AnchorTargetCreator selected {} postive anchors '
                     'and {} negative anchors to train RPN'.
                     format(len(pos_targets), len(neg_targets)))
        pos_ious = [tar['iou'] for tar in pos_targets]
        neg_ious = [tar['iou'] for tar in neg_targets]
        logging.info('Max and min iou of positive targets: {}, {}'\
                     .format(max(pos_ious) if len(pos_ious)>0 else None,
                             min(pos_ious) if len(pos_ious)>0 else None))
        logging.info('Max and min iou of negative targets: {}, {}'\
                     .format(max(neg_ious) if len(neg_ious)>0 else None,
                             min(neg_ious) if len(neg_ious)>0 else None))

        all_targets = pos_targets + neg_targets
        all_anchor_ids = [tar['anchor']['id'] for tar in all_targets]
        if len(all_anchor_ids) != len(set(all_anchor_ids)):
            logging.warning('Found multiply chosen anchors!')
        return all_targets
            


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
# argument anchors is a list of anchors, score_map is a function that
# maps items in anchors to score and bbox_map maps to bbox which is of BBox class.
def apply_nms_(anchors, score_map, bbox_map, iou_thr):
    start = time.time()
    num_anchors = len(anchors)
    print('number of anchors:', num_anchors)
    # third flag is meaning 'deleted'
    anchors = [[score_map(anchor), bbox_map(anchor), 0, anchor] for anchor in anchors]
    anchors.sort(key = lambda x:x[0], reverse=True)
    for i, anchor in enumerate(anchors):
        if not anchor[2]:
            cur_bbox = anchor[1]
            for j in range(i+1, num_anchors):
                if not anchors[j][2] and \
                   calc_iou(cur_bbox, anchors[j][1])>=iou_thr:
                    anchors[j][2] = 1
    ret_val = [anchor[3] for i, anchor in enumerate(anchors) if not anchor[2]]
    return ret_val

def apply_nms(anchors, score_map, bbox_map, iou_thr):
    start = time.time()
    bboxes = [bbox_map(anchor).get_xyxy() for anchor in anchors]
    scores = [score_map(anchor) for anchor in anchors]
    bboxes = torch.tensor(bboxes)
    scores = torch.tensor(scores)
    keep = torchvision.ops.nms(bboxes, scores, iou_thr)
    ret_val = [anchors[i] for i in keep.numpy()]
    return ret_val

# constrain a proposal to be within the image size
def constrain_proposal(props, img_size):
    x1,y1,x2,y2 = props.get_xyxy()
    img_h, img_w = img_size
    res = BBox(xyxy=(max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)))
    return res


class ProposalCreator(object):
    r"""
    Given a RPN classifier output, a RPN regressor output and a anchor generator,
    1, calculate anchor objectness
    2, adjust anchor position with regressed parameters
    3, choose top N(12000) anchors
    4, apply NMS and choose top M(2000)

    After this, anchor will add members: 
        'obj_score': float,
        'adj_bbox': BBox,
    """
    def __init__(self, anchor_generator, max_by_score, max_after_nms, nms_iou):
        self.anchor_generator = anchor_generator
        self.max_by_score = max_by_score
        self.max_after_nms = max_after_nms
        self.nms_iou = nms_iou

    def proposals(self, rpn_cls_res, rpn_reg_res, img_size, grid):
        grid = tuple(grid)
        cls_res_size = (rpn_cls_res.shape[-2], rpn_cls_res.shape[-1])
        reg_res_size = (rpn_reg_res.shape[-2], rpn_reg_res.shape[-1])
        assert grid == cls_res_size and grid == reg_res_size
        anchors = list(self.anchor_generator.anchors(img_size, grid))
        # put score and adjusted bbox to anchors
        # it is convenient that anchors are defined as a dictionaries
        num_scales = len(self.anchor_generator.scales)
        num_ars = len(self.anchor_generator.aspect_ratios)
        processed = []
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
            obj_soft = torch.softmax(objectness, dim=0)
            obj_score = obj_soft[1].item()
            adj_bbox  = param2xywh(adjustment, anchor_bbox)
            is_finite = True
            for n in adj_bbox:
                if not np.isfinite(n):
                    is_finite = False
                    break
            if not is_finite:
                break

            # next put extra information to anchor
            anchor['obj_score'] = obj_score
            anchor['adj_bbox'] = BBox(xywh=adj_bbox)
            #anchor['adj_bbox'] = constrain_proposal(BBox(xywh=adj_bbox), img_size)
            processed.append(anchor)
        logging.info('ProposalCreator selected {} anchors with coordinated '
                     'adjusted from {} anchors'.format(len(processed), len(anchors)))
        return processed
    # the filter process does:
    #   1, filter adj_bboxes having 0 IOU with image size
    #   2, constrain adj_bboxes within image
    #   3, filter by score
    #   4, apply NMS followed by a filter by number
    def proposals_filtered(self, rpn_cls_res, rpn_reg_res, img_size, grid):
        assert self.max_by_score > 0 and self.max_after_nms > 0
        props = self.proposals(rpn_cls_res, rpn_reg_res, img_size, grid)
        props_filt = []
        img_bbox = BBox(xywh=(0.0, 0.0, img_size[1], img_size[0]))
        for prop in props:
            if calc_iou(prop['adj_bbox'], img_bbox) == 0:
                continue
            prop['adj_bbox'] = constrain_proposal(prop['adj_bbox'], img_size)
            props_filt.append(prop)
        logging.info('ProposalCreator filtered {} adj_bboxes having IOU 0 with image'.
                     format(len(props)-len(props_filt)))
            
        props_filt.sort(key=lambda x: x['obj_score'], reverse=True)
        props = props_filt[:self.max_by_score]
        props_nms = apply_nms(
            props,
            score_map=lambda x: x['obj_score'],
            bbox_map=lambda x: x['adj_bbox'],
            iou_thr=self.nms_iou)
        props_nms = props_nms[:self.max_after_nms] 
        logging.info('ProposalCreator selected {} anchors after filter by score '
                     'and NMS'.format(len(props_nms)))
        return props_nms
    
class ProposalTargetCreator(object):
    r"""
    From selected ROIs(around 2000, by ProposalCreator), 
    choose 128 samples for training Head.
    """
    def __init__(self, max_pos=32, max_targets=128, pos_iou=0.5, neg_iou_hi=0.5, neg_iou_lo=0.1):
        self.max_pos = max_pos
        self.max_targets = max_targets
        self.pos_iou = pos_iou
        self.neg_iou_hi = neg_iou_hi
        self.neg_iou_lo = neg_iou_lo

    # proposal keys: 'bbox', 'center', 'feat_loc', 'scale_idx', 'ar_idx', 'id', 'obj_score',
    # 'adj_bbox'.
    # will add member after this: 'gt_bbox', 'category', 'iou'
    def targets(self, proposals, gt):
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
            if max_iou < self.neg_iou_hi and max_iou >= self.neg_iou_lo:
                prop['gt_bbox'] = None
                prop['gt_label'] = 0
                prop['category'] = 0
                prop['iou'] = iou
                neg_targets.append(prop)
        pos_targets.sort(key=lambda x: x['iou'], reverse=True)
        pos_targets = pos_targets[:self.max_pos]
        random.shuffle(neg_targets)
        neg_targets = neg_targets[:self.max_targets - len(pos_targets)]
        logging.info('ProposalTargetCreator selected {} positive and {} '
                     'negative targets to train RCNN'.format(len(pos_targets), len(neg_targets)))
        return pos_targets + neg_targets

# conversion btw bbox in image and in feature map
def image2feature(img_size, feat_size, img_bbox):
    x,y,w,h = img_bbox.get_xywh()
    h_rat, w_rat = feat_size[0]/img_size[0], feat_size[1]/img_size[1]
    feat_bbox = (x*w_rat, y*h_rat, w*w_rat, h*h_rat)
    return feat_bbox
def feature2image(img_size, feat_size, feat_bbox):
    x,y,w,h = feat_bbox.get_xywh()
    # TODO: do we do make sure of non-zero feature area here?
    h_rat = img_size[0] / feat_size[0]
    w_rat = img_size[1] / feat_size[1]
    img_bbox = (x*w_rat, y*h_rat, w*w_rat, h*h_rat)
    return img_bbox
    
class ROICropping(object):
    r"""
    It accepts a feature map and region proposals and return a list
    of crops from the feature map, the cropped areas usually have different
    spatial sizes.
    It does not involve any trainable parameter so no need to inherite nn.Module
    """
    def __init__(self):
        pass
    # proposals are results of ProposalTargetCreator or ProposalTargetCreator, 
    # which must have member: 'adj_bbox'
    def crop_v2(self, img_size, feature, proposals):
        img_h, img_w = img_size
        feat_size = feature.shape[-2:]
        crops, category_labels, gt_bboxes, adj_bboxes = [], [], [], []
        for prop in proposals:
            adj_bbox = prop['adj_bbox']
            feat_bbox = image2feature(img_size, feat_size, adj_bbox)
            adj_x, adj_y, adj_w, adj_h = feat_bbox
            # odd that round does not always returns integers
            crop = feature[:,
                           :,
                           int(round(adj_y)):int(round(adj_y+adj_h)),
                           int(round(adj_x)):int(round(adj_x+adj_w))]
            # get rid of zero size feature crops
            # this may happen if adjusted bbox is out of image bounds
            # TODO: do we use a little piece next to the boundary instead of
            # ignoring?
            if crop.numel() == 0:
                continue
            crops.append(crop)
            category_labels.append(prop['category'])
            gt_bboxes.append(prop['gt_bbox'])
            adj_bboxes.append(adj_bbox)
        return crops, adj_bboxes, gt_bboxes, category_labels
    
    # cropping does not depend on if proposals are with targets or not
    def crop(self, img_size, feature, proposals):
        img_h, img_w = img_size
        feat_size = feature.shape[-2:]
        crops, props = [], []
        pos_crops, neg_crops, test_crops = 0, 0, 0
        for prop in proposals:
            adj_bbox = prop['adj_bbox']
            feat_bbox = image2feature(img_size, feat_size, adj_bbox)
            adj_x, adj_y, adj_w, adj_h = feat_bbox
            if adj_x < 0 or adj_y < 0:
                logging.warning('Encounter negative adjustment bbox '
                                'boundaries: {},{},{},{}, gt_label: {}'\
                                .format(adj_x, adj_y, adj_w, adj_h,
                                        prop['gt_label'] if 'gt_label' in prop else None))
            # odd that round does not always returns integers
            # it happens left or upper bound of the proposal is out of the boundaries of
            # the image (i.e. <0), and mostly they are background proposals.
            # in this case, only crop the overlapped part.
            crop = feature[:,
                           :,
                           max(0, int(round(adj_y))):int(round(adj_y+adj_h)),
                           max(0, int(round(adj_x))):int(round(adj_x+adj_w))]
            # get rid of zero size feature crops
            # this may happen if adjusted bbox is out of image bounds
            # TODO: do we use a little piece next to the boundary instead of
            # ignoring?
            if crop.numel() == 0:
                continue
            crops.append(crop)
            props.append(prop)
            if 'gt_label' not in prop:
                test_crops += 1
            elif prop['gt_label'] == 1:
                pos_crops += 1
            elif prop['gt_label'] == 0:
                neg_crops += 1
        logging.info('ROICropping selected {} pos crops, {} neg crops and {} test crops '
                     'to feed to RCNN.'.format(pos_crops, neg_crops, test_crops))
        return crops, props

                            
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
        rois = [roi for roi in rois if roi.numel()>0]
        batch_size = len(rois)
        outs = [self.adaptive_pool(roi) for roi in rois]
        if len(outs) == 0:
            return None
        return torch.cat(outs)
                                                                        

#####################################################
### new implementation using vertorized computing ###
#####################################################

import numpy as np

class AnchorCreator(object):

    MAX_CACHE_ANCHOR = 1000
    CACHE_REPORT_PERIOD = 500
    def __init__(self, base=16, scales=[8, 16, 32],
                 aspect_ratios=[0.5, 1.0, 2.0], device=torch.device('cuda:0')):
        self.device = device
        self.base = base
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.cached = {}
        self.count = 0
        anchor_ws, anchor_hs = [], []
        for s in scales:
            for ar in aspect_ratios:
                anchor_ws.append(base * s * np.sqrt(ar))
                anchor_hs.append(base * s / np.sqrt(ar))
        self.anchor_ws = torch.tensor(anchor_ws, device=device, dtype=torch.float32)
        self.anchor_hs = torch.tensor(anchor_hs, device=device, dtype=torch.float32)

    def to(self, device):
        self.device = device
        self.anchor_ws.to(device)
        self.anchor_hs.to(device)

    def report_cache(self):
        count_info = [[k, v[0]] for k,v in self.cached.items()]
        count_info.sort(key=lambda x:x[1], reverse=True)
        top_count = count_info[:10]
        top_str = ', '.join([':'.join([str_id, str(ct)]) for str_id, ct in top_count])
        rep_str = '\n'.join([
            'AnchorCreator count: {}'.format(self.count),
            'Cache size: {}'.format(len(self.cached)),
            'Top 10 used anchor count: {}'.format(top_str)
        ])
        logging.info(rep_str)

    def __call__(self, img_size, grid):
        str_id = '|'.join([
            ','.join([str(x) for x in img_size]),
            ','.join([str(x) for x in grid])
        ])
        # check if the anchor is in the cached
        if str_id in self.cached:
            self.cached[str_id][0] += 1
            return self.cached[str_id][1]
        anchors = self._create_anchors_(img_size, grid)
        if len(self.cached) < self.MAX_CACHE_ANCHOR:
            self.cached[str_id] = [1, anchors]
        self.count += 1
        if self.count % self.CACHE_REPORT_PERIOD == 0:
            self.report_cache()
        return anchors
        
    def _create_anchors_(self, img_size, grid):
        assert len(img_size) == 2 and len(grid) == 2
        imag_h, imag_w = img_size
        grid_h, grid_w = grid
        grid_dist_h, grid_dist_w = imag_h/grid_h, imag_w/grid_w
        
        center_h = torch.linspace(0, imag_h, grid_h+1,
                                  device=self.device, dtype=torch.float32)[:-1] + grid_dist_h/2
        center_w = torch.linspace(0, imag_w, grid_w+1,
                                  device=self.device, dtype=torch.float32)[:-1] + grid_dist_w/2
        mesh_h, mesh_w = torch.meshgrid(center_h, center_w)
        # NOTE that the corresponding is h <-> y and w <-> x
        anchor_hs = self.anchor_hs.view(-1, 1, 1)
        anchor_ws = self.anchor_ws.view(-1, 1, 1)
        x_min = mesh_w - anchor_ws / 2
        x_max = mesh_w + anchor_ws / 2
        y_min = mesh_h - anchor_hs / 2
        y_max = mesh_h + anchor_hs / 2
        anchors = torch.stack([x_min, y_min, x_max, y_max])
        return anchors

def find_inside_index(anchors, img_size):
    H, W = img_size
    inside = (anchors[0,:]>=0) & (anchors[1,:]>=0) & \
             (anchors[2,:]<=W) & (anchors[3,:]<=H)
    return inside

def random_sample_label(labels, pos_num, tot_num):
    assert pos_num <= tot_num
    pos_args = utils.index_of(labels==1)
    if len(pos_args[0]) > pos_num:
        dis_idx = np.random.choice(
            pos_args[0].cpu().numpy(), size=(len(pos_args[0]) - pos_num), replace=False)
        labels[dis_idx] = -1
    real_n_pos = min(len(pos_args[0]), pos_num)
    n_negs = tot_num - real_n_pos
    neg_args = utils.index_of(labels==0)
    if len(neg_args[0]) > n_negs:
        dis_idx = np.random.choice(
            neg_args[0].cpu().numpy(), size=(len(neg_args[0]) - n_negs), replace=False)
        labels[dis_idx] = -1
    return labels

class AnchorTargetCreator(object):
    def __init__(self, pos_iou=0.7, neg_iou=0.3, max_pos=128, max_targets=256):
        self.pos_iou = pos_iou
        self.neg_iou = neg_iou
        self.max_pos = max_pos
        self.max_targets = max_targets

    def __call__(self, img_size, feat_size, anchors, gt_bbox):
        assert anchors.shape[0] == 4 and gt_bbox.shape[0] == 4
        # TODO: find out why there is a diff btw old and new version
        with torch.no_grad():
            gt_bbox = gt_bbox.to(torch.float32)
            n_anchors, n_gts = anchors.shape[1], gt_bbox.shape[1]
            labels = torch.full((n_anchors,), -1, device=anchors.device, dtype=torch.int8)
            iou_tab = utils.calc_iou(anchors, gt_bbox)
            max_anchor_iou, max_anchor_arg = torch.max(iou_tab, dim=0)
            max_gt_iou, max_gt_arg = torch.max(iou_tab, dim=1)
            # first label negative anchors, some of them might be replaced with positive later
            labels[(max_gt_iou < self.neg_iou)] = 0
            # next label positive anchors
            labels[max_anchor_arg] = 1
            labels[(max_gt_iou >= self.pos_iou)] = 1
            labels = random_sample_label(labels, self.max_pos, self.max_targets)
            bbox_labels = gt_bbox[:,max_gt_arg]
            param = utils.bbox2param(anchors, bbox_labels)
        return labels, param, bbox_labels
        

class ProposalCreator(object):
    def __init__(self, max_pre_nms, max_post_nms, nms_iou, min_size):
        self.max_pre_nms = max_pre_nms
        self.max_post_nms = max_post_nms
        self.nms_iou = nms_iou
        self.min_size = min_size

    def __call__(self, rpn_cls_out, rpn_reg_out, anchors, img_size, scale=1.0):
        assert anchors.shape[0] == 4 and len(anchors.shape) == 2
        n_anchors = anchors.shape[1]
        #min_size = scale * self.min_size # this is the value from simple-faster-rcnn
        min_size = 17 # this is the old version value which is basically 1 in feature map
        H, W = img_size
        with torch.no_grad():
            cls_out = rpn_cls_out.view(2, -1)
            reg_out = rpn_reg_out.view(4, -1)
            scores = torch.softmax(cls_out, 0)[1]
            props_bbox = utils.param2bbox(anchors, reg_out)
            props_bbox = torch.stack([
                torch.clamp(props_bbox[0], 0.0, W),
                torch.clamp(props_bbox[1], 0.0, H),
                torch.clamp(props_bbox[2], 0.0, W),
                torch.clamp(props_bbox[3], 0.0, H)
            ])
            small_area_idx = utils.index_of(
                (props_bbox[2] - props_bbox[0]) * (props_bbox[3] - props_bbox[1]) < min_size
            )
            scores[small_area_idx] = -1
            sort_args = torch.argsort(scores, descending=True)
            sort_args = sort_args[sort_args!=-1]
            top_sort_args = sort_args[:self.max_pre_nms]
            
            props_bbox = props_bbox[:, top_sort_args]
            top_scores = scores[top_sort_args]
            keep = torchvision.ops.nms(props_bbox.t(), top_scores, self.nms_iou)

            keep = keep[:self.max_post_nms]
        return props_bbox[:, keep], top_scores[keep]
        

class ProposalTargetCreator(object):
    r"""
    From selected ROIs(around 2000, by ProposalCreator),
    choose 128 samples for training Head.
    """
    def __init__(self,
                 max_pos=32,
                 max_targets=128,
                 pos_iou=0.5,
                 neg_iou_hi=0.5,
                 neg_iou_lo=0.1):
        self.max_pos = max_pos
        self.max_targets = max_targets
        self.pos_iou = pos_iou
        self.neg_iou_hi = neg_iou_hi
        self.neg_iou_lo = neg_iou_lo

    def __call__(self, props_bbox, gt_bbox, gt_label):
        # TODO: this version does not add gt to train classifier in RCNN
        with torch.no_grad():
            gt_bbox = gt_bbox.to(torch.float32)
            n_props, n_gts = props_bbox.shape[1], gt_bbox.shape[1]
            iou_tab = utils.calc_iou(props_bbox, gt_bbox)
            max_gt_iou, max_gt_arg = torch.max(iou_tab, dim=1)
            label = torch.full((n_props,), -1, device = props_bbox.device, dtype=torch.int8)
            label[max_gt_iou > self.pos_iou] = 1
            label[(max_gt_iou < self.neg_iou_hi) & (max_gt_iou >= self.neg_iou_lo)] = 0
            label = random_sample_label(label, self.max_pos, self.max_targets)
            pos_idx, neg_idx = (label==1), (label==0)
            chosen_idx = pos_idx | neg_idx
            # find class label of each roi, 0 is background
            roi_label = gt_label[max_gt_arg]
            roi_label[neg_idx] = 0
            # find gt bbox for each roi
            roi_gt_bbox = gt_bbox[:,max_gt_arg]
            roi_param = utils.bbox2param(props_bbox, roi_gt_bbox)
        # next only choose rois of non-negative
        return props_bbox[:,chosen_idx], roi_label[chosen_idx], roi_param[:,chosen_idx]


def image2feature(bbox, img_size, feat_size):
    """
    transfer bbox size from image to feature
    """
    h_rat, w_rat = [feat_size[i]/img_size[i] for i in range(2)]
    return bbox * torch.tensor([[w_rat], [h_rat], [w_rat], [h_rat]],
                               device=bbox.device, dtype=torch.float32)
    
    
class ROICropping(object):
    def __init__(self):
        pass

    def __call__(self, feature, props, image_size):
        _, n_chanel, h, w = feature.shape
        feat_size = feature.shape[-2:]
        # process of cropping participates in the computation graph
        bbox_feat = image2feature(props, image_size, feat_size).round().int()
        crops = [feature[0, :, y_min:y_max+1, x_min:x_max+1] \
                 for x_min, y_min, x_max, y_max in bbox_feat.t()]
        return crops


class ROIPooling(nn.Module):
    def __init__(self, out_size):
        super(ROIPooling, self).__init__()
        self.out_size = out_size
        self.adaptive_pool = nn.AdaptiveMaxPool2d(out_size)
        
    def forward(self, rois):
        return torch.stack([self.adaptive_pool(x) for x in rois])
