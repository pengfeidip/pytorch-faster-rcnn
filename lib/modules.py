import torchvision as tv
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import config

class VGGBackbone(nn.Module):
    r"""
    Use the VGG16 CNN as the backbone
    """
    def __init__(self, pretrained=True, device=torch.device('cpu')):
        super(VGGBackbone, self).__init__()
        self.pretrained=pretrained
        self.device=device
        vgg16 = tv.models.vgg16(pretrained=pretrained)
        vgg16.to(device=device)
        self.backbone = vgg16.features[:-1]
        # do not register the original vgg16 network
        self.vgg16 = [vgg16]

    def forward(self, x):
        return self.backbone(x)

def init_module_normal(module, mean=0.0, std=1.0):
    for param in module.parameters():
        torch.nn.init.normal_(param, mean, std)

class RPN(nn.Module):
    r"""
    Region proposal network.
    """
    def __init__(self, num_classes, num_anchors):
        super(RPN, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.classifier = nn.Conv2d(512, num_anchors*2, kernel_size=(1, 1))
        self.regressor  = nn.Conv2d(512, num_anchors*4, kernel_size=(1, 1))
        init_module_normal(self.conv, mean=0.0, std=0.01)
        init_module_normal(self.classifier, mean=0.0, std=0.01)
        init_module_normal(self.regressor, mean=0.0, std=0.01)

    def forward(self, x):
        x = F.relu(self.conv(x))
        return self.classifier(x), self.regressor(x)

class RCNN(nn.Module):
    r"""
    The Fast RCNN or RCNN part of the detector, takes ROIs and classify them and adjust bboxes.
    Use weights of FC layers of a CNN backbone to initialize FC layers of the head, if possible.
    """
    def __init__(self, num_classes, fc1_state_dict=None, fc2_state_dict=None):
        super(RCNN, self).__init__()
        # in_features is 128 x 512 x 7 x 7 where 128 is batch size
        # TO-DO: initialize fc layers with fc layers in VGG16
        self.bn = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(512*7*7, 4096)
        if fc1_state_dict is not None:
            self.fc1.load_state_dict(fc1_state_dict)
        else:
            init_module_normal(self.fc1, mean=0.0, std=0.01)
        self.fc2 = nn.Linear(4096, 4096)
        if fc2_state_dict is not None:
            self.fc2.load_state_dict(fc2_state_dict)
        else:
            init_moduel_normal(self.fc2, mean=0.0, std=0.01)
        self.classifier = nn.Linear(4096, num_classes+1)
        self.regressor  = nn.Linear(4096, (num_classes+1)*4)
        init_module_normal(self.classifier, mean=0.0, std=0.01)
        init_module_normal(self.regressor, mean=0.0, std=0.01)

    # roi_batch is a batch of fixed tensors which is the result of ROIPooling
    def forward(self, roi_batch):
        if roi_batch is None or len(roi_batch) == 0:
            return None, None
        batch_size = roi_batch.shape[0]
        x = self.bn(roi_batch)
        # flatten input rois
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.classifier(x), self.regressor(x)


