# Copyright (c) 2022 vios-s

import torch.nn as nn
import torch.nn.functional as F
from models.unet_parts import *
from models.blocks import *

class Segmentor(nn.Module):
    def __init__(self, num_classes, vc_num_seg=12):
        super(Segmentor, self).__init__()
        self.num_classes = num_classes
        input_channels =  vc_num_seg
        out_channels = 64

        self.conv1 = DoubleConv(input_channels, out_channels)
        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(64, 64)

        self.outc = OutConv(64, self.num_classes)

    def forward(self, content):
        out = self.conv1(content)
        out = self.up4(out)
        out = self.conv2(out)
        out = self.outc(out)
        out = F.softmax(out, dim=1)
        return out