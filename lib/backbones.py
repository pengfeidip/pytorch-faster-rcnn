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


###
### a simple implementation of ResNet series
###
    
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(Bottleneck, self).__init__()
        is_first = in_channels*4==out_channels
        self.in_channels=in_channels
        self.out_channels=out_channels
        hidden_channels=out_channels//4
        self.hidden_channels=hidden_channels
        self.do_downsample=downsample
        
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3,
                               stride=2 if downsample and not is_first else 1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if downsample:
            self.downsample = make_downsample(
                in_channels, out_channels,
                stride=1 if is_first else 2)

            
    def forward(self, x):
        out = x
        out = self.bn1(self.conv1(out))
        out = self.relu(out)
        out = self.bn2(self.conv2(out))
        out = self.relu(out)
        out = self.bn3(self.conv3(out))

        if self.do_downsample:
            x = self.downsample(x)
        return self.relu(x+out)


def make_reslayer(in_channels, out_channels, num_bns):
    bns = [Bottleneck(in_channels, out_channels, True)]
    for i in range(num_bns-1):
        bns.append(Bottleneck(out_channels, out_channels))
    return nn.Sequential(*bns)


def make_downsample(in_channels, out_channels, stride=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                         nn.BatchNorm2d(out_channels))


RES_CONV_CFG = {
    50 : [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3]}

RES_CHANNELS = [64, 256, 512, 1024, 2048]

TORCH_RESNET = {
    50 : tv.models.resnet50,
    101: tv.models.resnet101,
    152: tv.models.resnet152
}
        
class ResNet(nn.Module):
    def __init__(self, depth=50, frozen_stages=1, out_layers=(1, 2, 3, 4), pretrained=True):
        super(ResNet, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        assert depth in RES_CONV_CFG
        self.frozen_stages = frozen_stages
        self.out_layers = out_layers
        conv_cfg = RES_CONV_CFG[depth]
        self.conv_cfg = conv_cfg
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        for i in range(4):
            setattr(self, 'layer{}'.format(i+1), make_reslayer(RES_CHANNELS[i], RES_CHANNELS[i+1], conv_cfg[i]))

        #self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        #self.fc = nn.Linear(2048, 1000, bias=True)

    def forward(self, x):
        out = x
        out = self.bn1(self.conv1(out))
        out = self.relu(out)
        out = self.maxpool(out)

        feats = []
        for i in range(1, 5):
            layer = getattr(self, 'layer{}'.format(i))
            out = layer(out)
            if i in self.out_layers:
                feats.append(out)
                
        return feats

    def freeze_stages(self, stages):
        if stages >= 0:
            self.bn1.eval()
            for m in [self.conv1, self.bn1]:
                for param in m.parameters():
                    param.requires_grad = False
        for i in range(1, stages+1):
            m = getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(ResNet, self).train(mode)
        self.freeze_stages(self.frozen_stages)
        if mode:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def init_weights(self):
        if self.pretrained:
            torch_resnet = TORCH_RESNET[self.depth](pretrained=True)
            self.load_state_dict(torch_resnet.state_dict())
