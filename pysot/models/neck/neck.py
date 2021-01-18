# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn


class AdjustLayer(nn.Module):
    def __init__(self, in_channels, out_channels, center_size=7):
        super(AdjustLayer, self).__init__()
        # 1x1 Conv + BN adjust the channels of the features
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            )
        # In SiamRPN++, the padding keeping, then the
        # spatial size of the template feature become
        # 15x15, which will increase computation intensity
        # then crop the center 7x7 as the template feature
        self.center_size = center_size

    def forward(self, x):
        x = self.downsample(x)
        # Guess ???
        # if x is the feature of the template,
        # then do center crop by the specified center size
        if x.size(3) < 20:
            # feature x data format
            # [batch, channel, H, W]
            l = (x.size(3) - self.center_size) // 2
            r = l + self.center_size
            x = x[:, :, l:r, l:r]
        return x


class AdjustAllLayer(nn.Module):
    def __init__(self, in_channels, out_channels, center_size=7):
        super(AdjustAllLayer, self).__init__()
        self.num = len(out_channels)
        if self.num == 1:
            self.downsample = AdjustLayer(in_channels[0],
                                          out_channels[0],
                                          center_size)
        else:
            # create downsample module for
            # multi-level features in ResNet50
            # then to input the multi-RPN modules
            for i in range(self.num):
                self.add_module('downsample'+str(i+2),
                                AdjustLayer(in_channels[i],
                                            out_channels[i],
                                            center_size))

    def forward(self, features):
        if self.num == 1:
            return self.downsample(features)
        else:
            out = []
            for i in range(self.num):
                adj_layer = getattr(self, 'downsample'+str(i+2))
                # In SiamRPN++, the first dim of feature is 3
                # The first dim representing the different 
                # level feature
                out.append(adj_layer(features[i]))
            return out
