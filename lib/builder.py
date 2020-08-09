from .backbones import ResNet50, VGG16, ResLayerC5, ResNet
from .region import MaxIoUAssigner, RandomSampler, SingleRoIExtractor, BasicRoIExtractor, IoUBalancedNegSampler, \
    ScalableRoIPool, ScalableRoIAlign
from .necks import FPN, BFP
from .losses import FocalLoss, SmoothL1Loss, CrossEntropyLoss, BalancedL1Loss, BoundedIoULoss, GIoULoss, IoULoss, DistributionFocalLoss, QualityFocalLoss
from .detectors import RetinaNet, CascadeRCNN, FCOS
from .heads import RetinaHead, RPNHead, RCNNHead, DoubleHead, GARPNHead, FCOSHead
from torch.optim import SGD
from torchvision.ops import RoIAlign, RoIPool

import copy
from .utils import sum_list



_detectors_ = [RetinaNet, CascadeRCNN, FCOS]
_backbones_ = [ResNet50, VGG16, ResLayerC5, ResNet]
_necks_ = [FPN, BFP]
_heads_ = [RetinaHead, RPNHead, RCNNHead, DoubleHead, GARPNHead, FCOSHead]
_losses_ = [CrossEntropyLoss, SmoothL1Loss, FocalLoss, BalancedL1Loss, BoundedIoULoss, GIoULoss,
            IoULoss, DistributionFocalLoss, QualityFocalLoss]
_roi_extractors_ = [SingleRoIExtractor, BasicRoIExtractor, RoIAlign, RoIPool, ScalableRoIPool, ScalableRoIAlign]
_optimizers_ = [SGD]
_bbox_utils_ = [MaxIoUAssigner, RandomSampler, IoUBalancedNegSampler]

_modules_ = sum_list([_detectors_, _backbones_, _necks_, _heads_, _losses_, _roi_extractors_, _optimizers_, _bbox_utils_])

MODULES = { cls.__name__:cls for cls in _modules_ }


def build_module(cfg, *args, **kwargs):
    cfg = copy.copy(cfg)
    assert 'type' in cfg
    m_type = cfg.pop('type')
    if m_type not in MODULES:
        raise ValueError("'{}' is not registered".format(m_type))
    return MODULES[m_type](*args, **cfg, **kwargs)

