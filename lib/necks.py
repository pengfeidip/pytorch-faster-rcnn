import torch
import torch.nn.functional as F
from torch import nn
from mmcv.cnn import xavier_init

"""
class FPN_(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 no_norm_on_lateral=False,
                 norm_cfg=None,
                 with_activation=False):
        super(FPN_, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.with_activation = with_activation
        self.no_norm_on_lateral = no_norm_on_lateral
        self.num_outs = num_outs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.num_ins):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                norm_cfg = None if self.no_norm_on_lateral else norm_cfg,
                with_activation=self.with_activation)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                norm_cfg=norm_cfg,
                with_activation=self.with_activation)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        # assume use all the inputs
        assert len(inputs) == self.num_ins
        lateral_outs = []
        for i, x in enumerate(inputs):
            lateral_outs.append(self.lateral_convs[i](x))
        for i in range(self.num_ins - 1, 0, -1):
            lateral_outs[i-1] += F.interpolate(lateral_outs[i], scale_factor=2, mode='nearest')
        outs = [self.fpn_convs[i](lateral_outs[i]) for i in range(self.num_ins)]
        if self.num_outs > len(outs):
            for i in range(self.num_outs - self.num_ins):
                outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        return tuple(outs)
        
"""     
        
    
class FPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 with_activation=False):
        assert with_activation is False, 'with_activation is not supported for FPN'
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.with_activation = with_activation
        self.num_outs = num_outs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(self.num_ins):
            l_conv = nn.Conv2d(in_channels[i],
                               out_channels,
                               1)
            fpn_conv = nn.Conv2d(out_channels,
                                 out_channels,
                                 3,
                                 padding=1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        # assume use all the inputs
        assert len(inputs) == self.num_ins
        lateral_outs = []
        for i, x in enumerate(inputs):
            lateral_outs.append(self.lateral_convs[i](x))
        for i in range(self.num_ins - 1, 0, -1):
            lateral_outs[i-1] += F.interpolate(lateral_outs[i], scale_factor=2, mode='nearest')
        outs = [self.fpn_convs[i](lateral_outs[i]) for i in range(self.num_ins)]
        if self.num_outs > len(outs):
            for i in range(self.num_outs - self.num_ins):
                outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        return tuple(outs)
        
        
        
    
