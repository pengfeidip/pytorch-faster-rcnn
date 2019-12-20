import torch.nn as nn
import torch, logging
from . import region

def zero_loss(device):
    return torch.tensor(0.0, device=device, requires_grad=True)

class RPNLoss(object):
    def __init__(self, lamb=1.0):
        self.lamb = lamb
        self.ce = nn.CrossEntropyLoss()
        self.smooth_l1 = nn.SmoothL1Loss(reduction='sum')

    def __call__(self, cls_out, reg_out, label, param):
        if label.numel() == 0:
            return zero_loss(cls_out.device)
        cls_loss = self.ce(cls_out.t(), label.long())
        n_samples = len(label)
        pos_arg = (label==1)
        if pos_arg.sum() == 0:
            reg_loss = zero_loss(cls_out.device)
        else:
            reg_loss = self.smooth_l1(reg_out[:, pos_arg], param[:, pos_arg]) / n_samples
        return cls_loss + self.lamb * reg_loss

class RCNNLoss(object):
    def __init__(self, lamb=1.0):
        self.lamb = lamb
        self.ce = nn.CrossEntropyLoss()
        self.smooth_l1 = nn.SmoothL1Loss(reduction='sum')

    def __call__(self, cls_out, reg_out, label, param):
        if label.numel() == 0 or cls_out is None or reg_out is None:
            return zero_loss(label.device)
        label = label.long()
        n_class = cls_out.shape[1]
        n_samples = len(label)
        cls_loss = self.ce(cls_out, label)
        reg_out = reg_out.view(-1, 4, n_class)
        reg_out = reg_out[torch.arange(n_samples), :, label]
        pos_arg = (label>1)
        if pos_arg.sum() == 0:
            reg_loss = zero_loss(label.device)
        else:
            pos_reg = reg_out[pos_arg, :]
            reg_loss = self.smooth_l1(pos_reg, param[:, pos_arg].t()) / n_samples
        return cls_loss + self.lamb * reg_loss

