from .modules import ResNet50, VGG16, ResLayerC5
from .region import MaxIoUAssigner, RandomSampler, SingleRoIExtractor
from .cascade_rcnn import CascadeRCNN
from .heads import RPNHead, BBoxHead

from copy import deepcopy
from torch.optim import SGD

_modules_ = [ResNet50, MaxIoUAssigner, RandomSampler, CascadeRCNN, RPNHead, BBoxHead,
             SGD, VGG16, ResLayerC5, SingleRoIExtractor]

MODULES = { cls.__name__:cls for cls in _modules_ }


def build_module(cfg, *args, **kwargs):
    cfg = deepcopy(cfg)
    assert 'type' in cfg
    m_type = cfg.pop('type')
    if m_type not in MODULES:
        raise ValueError("'{}' is not registered".format(m_type))
    return MODULES[m_type](*args, **cfg, **kwargs)

