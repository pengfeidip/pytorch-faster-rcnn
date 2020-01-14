import torchvision as tv
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import config


def decompose_vgg16():
    model = tv.models.vgg16(pretrained=True)
    features = list(model.features)[:30]
    classifier = model.classifier

    classifier = list(classifier)
    del classifier[6]
    del classifier[5]
    del classifier[2]
    classifier = nn.Sequential(*classifier)
        
    # freeze top4 conv
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False
            
    return nn.Sequential(*features), classifier

def make_vgg16_backbone():
    vgg16 = tv.models.vgg16(pretrained=True)
    feature_weights = vgg16.features[:30].state_dict()
    backbone = VGG16Backbone()
    backbone.features.load_state_dict(feature_weights)

    cls_weights = nn.Sequential(vgg16.classifier[0], vgg16.classifier[1],
                                vgg16.classifier[3], vgg16.classifier[4]).state_dict()
    classifier = VGG16Classifier()
    classifier.classifier.load_state_dict(cls_weights)

    for layer in backbone.features[:10]:
        for p in layer.parameters():
            p.requires_grad = False

    return backbone, classifier
    

class VGG16Classifier(nn.Module):
    def __init__(self, device=None):
        super(VGG16Classifier, self).__init__()
        self.device=device
        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(inplace=True)
        )
        if device is not None:
            self.to(device)

    def forward(self, x):
        return self.classifier(x)


class VGG16Backbone(nn.Module):
    def __init__(self, device=None):
        super(VGG16Backbone, self).__init__()
        self.device = device
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )
        if device is not None:
            self.to(device)
            
    def forward(self, x):
        return self.features(x)

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

        features = list(vgg16.features)[:30]
        for layer in features[:10]:
            for p in layer.parameters():
                p.requires_grad = False
        
        self.backbone = nn.Sequential(*features)
        # do not register the original vgg16 network
        self.vgg16 = [vgg16]

    def forward(self, x):
        return self.backbone(x)

def init_module_normal(m, mean=0.0, std=1.0):
    m.weight.data.normal_(mean, std)
    m.bias.data.zero_()


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
    def __init__(self, num_classes, cls_fc):
        super(RCNN, self).__init__()
        # in_features is 128 x 512 x 7 x 7 where 128 is batch size
        self.cls_fc = cls_fc
        self.classifier = nn.Linear(4096, num_classes+1)
        self.regressor  = nn.Linear(4096, (num_classes+1)*4)
        init_module_normal(self.classifier, mean=0.0, std=0.01)
        init_module_normal(self.regressor, mean=0.0, std=0.001)
        '''
        self.fc1 = nn.Linear(512*7*7, 4096)
        if fc1_state_dict is not None:
            self.fc1.load_state_dict(fc1_state_dict)
        else:
            init_module_normal(self.fc1, mean=0.0, std=0.01)
        self.fc2 = nn.Linear(4096, 4096)
        if fc2_state_dict is not None:
            self.fc2.load_state_dict(fc2_state_dict)
        else:
            init_module_normal(self.fc2, mean=0.0, std=0.01)
        self.classifier = nn.Linear(4096, num_classes+1)
        self.regressor  = nn.Linear(4096, (num_classes+1)*4)
        init_module_normal(self.classifier, mean=0.0, std=0.01)
        init_module_normal(self.regressor, mean=0.0, std=0.001)
        '''

    # roi_batch is a batch of fixed tensors which is the result of ROIPooling
    def forward(self, roi_batch):
        if roi_batch is None or len(roi_batch) == 0:
            return None, None
        batch_size = roi_batch.shape[0]
        x = roi_batch.view(batch_size, -1)
        fc = self.cls_fc(x)
        cls_out = self.classifier(fc)
        reg_out = self.regressor(fc)
        return cls_out, reg_out


