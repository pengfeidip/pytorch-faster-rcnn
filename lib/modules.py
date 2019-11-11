import torchvision as tv
import torch
import torch.nn as nn
from . import config

class Backbone(object):
    r"""
    CNN net that extract image features which are feed to RPN and Head.
    """
    def __init__(self, nn_module=None):
        # a torch.nn.module object that 
        self.nn_module

    def __call__(self, in_data):
        return self.nn_module(in_data)


class VGG(Backbone):
    r"""
    Use the VGG16 CNN as the backbone.
    """
    def __init__(self, pretrained=True, device=torch.device('cpu')):
        self.pretrained=pretrained
        self.device=device
        v16 = tv.models.vgg16(pretrained=pretrained)
        v16.to(device=device)
        self.nn_module = v16.features[:-1]


class RPN(nn.Module):
    r"""
    Region proposal network.
    """
    def __init__(self, num_classes, num_anchors):
        super(RPN, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.conv = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU()
        )
        self.classifier = nn.Conv2d(512, num_anchors*2, kernel_size=(1, 1))
        self.regressor  = nn.Conv2d(512, num_anchors*4, kernel_size=(1, 1))

    def forward(self, in_data):
        x = self.conv(in_data)
        return self.classifier(x), self.regressor(x)

class Head(nn.Module):
    r"""
    Takes ROIs and return class and bbox adjustment.
    """
    def __init__(self, num_classes):
        super(Head, self).__init__()
        # in_features is 128 x 512 x 7 x 7 where 128 is batch size
        # TO-DO: initialize fc layers with fc layers in VGG16
        self.fc1 = nn.Linear(512*7*7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.classifier = nn.Linear(4096, num_classes+1)
        self.regressor  = nn.Linear(4096, (num_classes+1)*4)

    # roi_batch is a batch of fixed tensors which is the result of ROIPooling
    def forward(self, roi_batch):
        batch_size = roi_batch.shape[0]
        # flatten input rois
        x = roi_batch.view(batch_size, -1)
        x = self.fc1(roi_batch)
        x = self.fc2(roi_batch)
        return self.classifier(x), self.regressor(x)


