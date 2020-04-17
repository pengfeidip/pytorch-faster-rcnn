from .backbones import ResNet50, VGG16, ResLayerC5
from .region import MaxIoUAssigner, RandomSampler, SingleRoIExtractor
from .necks import FPN
from .losses import FocalLoss, SmoothL1Loss, CrossEntropyLoss
from .detectors import RetinaNet, CascadeRCNN
from .heads import RetinaHead, RPNHead, RCNNHead

import copy
from torch.optim import SGD

_modules_ = [ResNet50, MaxIoUAssigner, RandomSampler, SGD, VGG16, ResLayerC5, SingleRoIExtractor, FPN,
             RetinaNet, RetinaHead, FocalLoss, SmoothL1Loss, CascadeRCNN, RPNHead,
             CrossEntropyLoss, RCNNHead]

MODULES = { cls.__name__:cls for cls in _modules_ }


def build_module(cfg, *args, **kwargs):
    cfg = copy.copy(cfg)
    assert 'type' in cfg
    m_type = cfg.pop('type')
    if m_type not in MODULES:
        raise ValueError("'{}' is not registered".format(m_type))
    return MODULES[m_type](*args, **cfg, **kwargs)

