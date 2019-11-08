import torch
import torch.nn as nn
from . import config

class BBox(object):
    r"""
    Represent a bounding box, it accepts various input and provide conversion.
    x, y here is upper left corner of the box.
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
            self.xywh = (round(cx - w/2), round(cy - h/2), w, h)
            self.center_xywh = center_xywh
        else:
            raise ValueError('No coordinates provided to init of bbox.')
        
        #self._validate_()
        

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
        return (round(x+w/2), round(y+h/w), w, h)
    def _validate_(self):
        x,y,w,h = self.xywh
        assert x>=0 and y>=0 and w>=0 and h>=0
        
    def __str__(self):
        return 'xywh:'+str(self.get_xywh())

class Point(object):
    def __init__(self, x=None, y=None, w=None, h=None):
        self.x = x
        self.y = y
        self.h = h
        self.w = w

class AnchorGenerator(object):
    r"""
    Generate a list of BBox objects that represents anchors of a certain setting.
    """
    def __init__(self, scales, aspect_ratios):
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.aspect_ratios_sqrt = [x**0.5 for x in aspect_ratios]
        
    def generate_anchors(self, img_size, grid):
        assert len(img_size)==2 and len(grid)==2
        assert img_size[0]>=0 and img_size[1]>=0 and grid[0]>=0 and grid[1]>=0
        h_img,  w_img  = img_size
        h_grid, w_grid = grid
        grid_dist_h, grid_dist_w = h_img/h_grid, w_img/w_grid
        for i in range(h_grid):
            for j in range(w_grid):
                i_center = grid_dist_h/2 + grid_dist_h*i
                j_center = grid_dist_w/2 + grid_dist_w*j
                same_center = {'center': Point(y=round(i_center), x=round(j_center)),
                               'anchors': []}
                for scale in self.scales:
                    for ar_sqrt in self.aspect_ratios_sqrt:
                        anchor_h = scale / ar_sqrt
                        anchor_w = scale * ar_sqrt
                        anchor = BBox(center_xywh=(round(j_center),
                                                   round(i_center),
                                                   round(anchor_w),
                                                   round(anchor_h)))
                        x1,y1,x2,y2 = anchor.get_xyxy()
                        # get rid of cross boundary anchors
                        if x1>=0 and x2>=0 and y1>=0 and y2>=0 and \
                           x1<w_img and x2<w_img and y1<h_img and y2<h_img:
                            same_center['anchors'].append(anchor)
                if len(same_center['anchors']) > 0:
                    yield same_center


        
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

