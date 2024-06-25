# Copyright (c) 2022 vios-s

import torch.nn as nn
from models.blocks import *

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
       
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(mid_channels),
            # nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Decoder(nn.Module):
    def __init__(self, num_channels):
        super(Decoder, self).__init__()

        self.num_channels = num_channels
        self.anatomy_out_channels = 0

        self.double_conv3 = DoubleConv(64 + self.anatomy_out_channels, 64)
        self.up4 = nn.ConvTranspose2d(64+self.anatomy_out_channels, 64+self.anatomy_out_channels, kernel_size=2, stride=2)
        self.double_conv4 = DoubleConv(64+self.anatomy_out_channels, 64)
        self.outc = nn.Conv2d(64, self.num_channels, kernel_size=1)

    def forward(self, features):
        out = self.double_conv3(features)
        out = self.up4(out)
        out = self.double_conv4(out)
        out = self.outc(out)
        return out