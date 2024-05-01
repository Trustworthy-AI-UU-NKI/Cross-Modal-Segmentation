import os
import torch
import argparse
import sys
import glob
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from data import MMWHS_single
import torch.optim as optim
import numpy as np


from monai.metrics import DiceMetric
from monai.networks.nets import UNet
import torch.nn as nn
from sklearn.model_selection import KFold 
from monai.losses import DiceLoss
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter


class DiceLossMC(nn.Module):
    def __init__(self, n_classes):
        super(DiceLossMC, self).__init__()
        self.n_classes = n_classes 
        self.loss = DiceLoss()
    
    def forward(self, pred, gt):
        loss = 0
        pred_sm = F.softmax(pred, dim=1)
        for i in range(self.n_classes):
            # skip the background
            if i == 0 :
                continue
            loss_i = self.loss(pred_sm[:, i, :, :].unsqueeze(1), gt[:, i, :, :].unsqueeze(1))
            loss += loss_i
        
        return loss/3


def dice(labels, pred, n_class):
    # Initialize the DiceMetric object
    # set reduction to 'none' to get the score for each class separately
    dice_metric = DiceMetric(reduction="mean_batch")#_batch")
    # print(compact_pred.shape)
    # print(labels.shape)
    compact_pred = torch.argmax(pred, dim=1).unsqueeze(1)
    compact_pred_oh = F.one_hot(compact_pred.long().squeeze(1), n_class).permute(0, 3, 1, 2)

    labels_oh = F.one_hot(labels.long().squeeze(1), n_class).permute(0, 3, 1, 2)
    # print(labels_oh.shape)

    # Compute the Dice score
    dice_metric(y_pred=compact_pred_oh, y=labels_oh)
    metric = dice_metric.aggregate()
    dice_metric.reset()

    return metric.detach().cpu().numpy()


def get_labels(pred):
    # match case statement
    match pred:
        case "MYO":
            labels = [1, 0, 0, 0, 0, 0, 0]
            n_classes = 2
        case "LV":
            labels = [0, 0, 1, 0, 0, 0, 0]
            n_classes = 2
        case "RV":
            labels = [0, 0, 0, 0, 1, 0, 0]
            n_classes = 2
        case "MYO_LV_RV":
            labels = [1, 0, 2, 0, 3, 0, 0]
            n_classes = 4
        case _:
            labels = [1, 1, 1, 1, 1, 1, 1]
            n_classes = 8
    
    return labels, n_classes



""" Full assembly of the parts to form the complete network """

###################################### maybe delete some skips
class UNet_model(nn.Module):
    def __init__(self, n_classes, norm="Batch", n_channels=1, bilinear=True):
        super(UNet_model, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64, norm=norm) #64x288x288 x1

        self.down1 = Down(64, 128)  #128x144x144 x2
        self.down2 = Down(128, 256) #256x72x72 x3
        self.down3 = Down(256, 512) #512x36x36 x4
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor) #512x18x18 x5
        self.up1 = Up(1024, 512 // factor, norm=norm, bilinear=bilinear) #256x36x36 y1
        self.up2 = Up(512, 256 // factor, norm=norm, bilinear=bilinear) #128x72x72 y2
        self.up3 = Up(256, 128 // factor, norm=norm, bilinear=bilinear) #64x144x144 y3
        self.up4 = Up(128, 64, norm=norm, bilinear=bilinear) #64x288x288 y4
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        y1 = self.up1(x5, x4)
        y2 = self.up2(y1, x3)
        y3 = self.up3(y2, x2)
        y4 = self.up4(y3, x1)
        logits = self.outc(y4)
        return logits
        #Layer   # 0     1    2   3   4  5   6   7   8   9


""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, norm="Batch"):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        if norm == "Batch":
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        if norm == "Instance":
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, norm="Batch"):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, norm=norm)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, norm="Batch", bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels = in_channels // 2, norm=norm)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, norm=norm)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

