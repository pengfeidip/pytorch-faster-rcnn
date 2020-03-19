import torch, logging
import torch.nn.functional as F
from torch import nn
from mmcv.cnn import xavier_init
        
    
class FPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 extra_use_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 with_activation=False):
        super(FPN, self).__init__()
        assert with_activation is False, 'with_activation is not supported for FPN'
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins  = len(self.in_channels)

        assert start_level >= 0
        self.start_level = start_level
        if end_level == -1:
            end_level = self.num_ins
        assert end_level > self.start_level and end_level <= self.num_ins
        self.end_level = end_level
        self.used_ins = self.end_level - self.start_level
        assert num_outs >= self.used_ins
        self.num_outs = num_outs

        self.extra_use_convs=extra_use_convs
        self.extra_convs_on_inputs=extra_convs_on_inputs
        self.relu_before_extra_convs=relu_before_extra_convs

        
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.end_level):
            l_conv = nn.Conv2d(in_channels[i], out_channels, 1)
            fpn_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        extra_levels = self.num_outs - self.used_ins
        if extra_levels > 0 and self.extra_use_convs:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channel = in_channels[self.end_level-1]
                else:
                    in_channel = out_channels

                self.fpn_convs.append(nn.Conv2d(
                    in_channel,
                    out_channels,
                    3,
                    stride=2,
                    padding=1))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        logging.info('Initialized weights for neck.')

    def forward(self, inputs):
        assert len(inputs) == self.num_ins
        start, end = self.start_level, self.end_level
        lateral_outs = [self.lateral_convs[i-start](inputs[i]) \
                        for i in range(start, end)]
        for i in range(self.used_ins-2, -1, -1):
            lateral_outs[i] += F.interpolate(lateral_outs[i+1], scale_factor=2, mode='nearest')
        outs = [self.fpn_convs[i](lateral_outs[i]) for i in range(self.used_ins)]
        if self.num_outs > self.used_ins:
            for i in range(self.used_ins, self.num_outs):
                if self.extra_use_convs:
                    if i == self.used_ins and self.extra_convs_on_inputs:
                        cur_out = self.fpn_convs[i](inputs[self.end_level-1])
                    else:
                        cur_out = self.fpn_convs[i](outs[-1])
                    if self.relu_before_extra_convs:
                        cur_out = F.relu(cur_out)
                    outs.append(cur_out)
                else:
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        return outs
        
        
        
    
