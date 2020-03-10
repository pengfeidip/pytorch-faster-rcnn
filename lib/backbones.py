import torchvision as tv
import torch.nn as nn
import logging, torch

from torch.nn.modules.batchnorm import _BatchNorm

class VGG16(nn.Module):
    def __init__(self, freeze_first_layers=True, pretrained=True):
        super(VGG16, self).__init__()
        self.pretrained=pretrained
        self.freeze_first_layers=freeze_first_layers
        vgg16 = tv.models.vgg16(pretrained=pretrained)
        features = list(vgg16.features)[:30]
        self.features=nn.Sequential(*features)

        cls = list(vgg16.classifier)
        cls_ = nn.Sequential(cls[0], cls[1], cls[3], cls[4])
        self.classifier_ = [cls_]

        if freeze_first_layers:
            for layer in self.features[:10]:
                for p in layer.parameters():
                    p.requires_grad=False

    def init_weights(self):
        pass
    def get_classifier(self):
        return self.classifier[0]
    def forward(self, x):
        x = self.features(x)
        return x


class ResNet50(nn.Module):
    def __init__(self,
                 out_layers=(3, ),
                 frozen_stages=1,
                 bn_requires_grad=True,
                 pretrained=True):
        super(ResNet50, self).__init__()
        self.frozen_stages = frozen_stages
        self.pretrained = pretrained
        self.out_layers=out_layers

        # borrow key components from pytorch resnet50
        res50 = tv.models.resnet50(pretrained=pretrained)
        self.conv1 = res50.conv1
        self.bn1 = res50.bn1
        self.relu = res50.relu
        self.maxpool = res50.maxpool
        self.layer1 = res50.layer1
        self.layer2 = res50.layer2
        self.layer3 = res50.layer3
        self.layer4 = res50.layer4
        
        self.freeze_stages(frozen_stages)

        self.bn_requires_grad=bn_requires_grad
        if not bn_requires_grad:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.weight.requires_grad=False
                    m.bias.requires_grad=False

                    
    def init_weights(self):
        pass

    def train(self, mode=True):
        super(ResNet50, self).train(mode)
        self.freeze_stages(self.frozen_stages)
        if mode:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def freeze_stages(self, stages):
        if stages >= 0:
            self.bn1.eval()
            for m in [self.conv1, self.bn1]:
                for param in m.parameters():
                    param.requires_grad=False
        for i in range(1, stages+1):
            m = getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad=False

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i in range(1, 5):
            layer = getattr(self, 'layer{}'.format(i))
            x = layer(x)
            if i in self.out_layers:
                outs.append(x)
        return outs

class ResLayerC5(nn.Module):
    def __init__(self, bn_requires_grad=True, pretrained=True):
        self.bn_requires_grad=bn_requires_grad
        super(ResLayerC5, self).__init__()
        res50 = tv.models.resnet50(pretrained=True)
        self.res_layer = res50.layer4

        if not bn_requires_grad:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.weight.requires_grad=False
                    m.bias.requires_grad=False


    def train(self, mode=True):
        super(ResLayerC5, self).train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x):
        return self.res_layer(x)

    def init_weights(self):
        pass
