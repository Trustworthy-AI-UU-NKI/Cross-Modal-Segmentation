import  torch
import torch.nn as nn
from torch.autograd import Variable
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
from layers import *

####################################################################
#------------------------- Discriminators -------------------------#
####################################################################



class Discriminator(nn.Module):
    def __init__(self, channels_in=1, aux=False):
        super(Discriminator, self).__init__()
        self.aux = aux
        self.conv1 = GeneralConv2d(channels_in, 64, kernel_size=4, stride=2, stddev=0.02, padding=0, do_norm=False, relu_factor=0.2, norm_type='Ins')
        self.conv2 = GeneralConv2d(64, 128, kernel_size=4, stride=2, stddev=0.02, padding=0, relu_factor=0.2, norm_type='Ins')
        self.conv3 = GeneralConv2d(128, 256, kernel_size=4, stride=2, stddev=0.02, padding=0, relu_factor=0.2, norm_type='Ins')
        self.conv4 = GeneralConv2d(256, 512, kernel_size=4, stride=1, stddev=0.02, padding=0, relu_factor=0.2, norm_type='Ins')
        if self.aux:
            self.conv5 = GeneralConv2d(512, 1, kernel_size=4, stride=1, stddev=0.02, padding=0, do_norm=False, do_relu=False)
        else:
            self.conv5 = GeneralConv2d(512, 1, kernel_size=4, stride=1, stddev=0.02, padding=0, do_norm=False, do_relu=False)
            
    # input == [B, C, H, W] == [B, 1, 256, 256]
    def forward(self, x):
        out = F.pad(x, (2, 2, 2, 2), mode="constant")
        out = self.conv1(out)
        out = F.pad(out, (2, 2, 2, 2), mode="constant")
        out = self.conv2(out)
        out = F.pad(out, (2, 2, 2, 2), mode="constant")
        out = self.conv3(out)
        out = F.pad(out, (2, 2, 2, 2), mode="constant")
        out = self.conv4(out)
        out = F.pad(out, (2, 2, 2, 2), mode="constant")
        out = self.conv5(out)
        if self.aux:
            return out[:, 0, :, :].unsqueeze(1), out[:, 1, :, :].unsqueeze(1)
        else:
            return out


####################################################################
#-------------------------- Encoders ------------------------------#
####################################################################

# encoderc in tf code
class EncoderCShared(nn.Module):
    def __init__(self, keep_rate=0.75):
        super(EncoderCShared, self).__init__()
        self.layers = nn.Sequential(
            GeneralConv2d(1, 16, kernel_size=7, stride=1, padding=3, norm_type="batch", keep_rate=keep_rate),
            ResNetBlock(16, norm_type="batch", padding="constant", keep_rate=keep_rate),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResNetBlockDS(16, 32, padding="constant", norm_type="batch", keep_rate=keep_rate),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResNetBlockDS(32, 64, padding="constant", norm_type="batch", keep_rate=keep_rate),
            ResNetBlock(64, norm_type="batch", padding="constant", keep_rate=keep_rate),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResNetBlockDS(64, 128, padding="constant", norm_type="batch", keep_rate=keep_rate),
            ResNetBlock(128, norm_type="batch", padding="constant", keep_rate=keep_rate),
            ResNetBlockDS(128, 256, padding="constant", norm_type="batch", keep_rate=keep_rate),
            ResNetBlock(256, norm_type="batch", padding="constant", keep_rate=keep_rate),
            ResNetBlock(256, norm_type="batch", padding="constant", keep_rate=keep_rate),
            ResNetBlock(256, norm_type="batch", padding="constant", keep_rate=keep_rate),
            ResNetBlockDS(256, 512, padding="constant", norm_type="batch", keep_rate=keep_rate),
            ResNetBlock(512, norm_type="batch", padding="constant", keep_rate=keep_rate)
        )

    #input == [B, C, H, W] == [B, 1, 256, 256]
    #output == [B, C, H, W] == [B, 512, 32, 32]
    def forward(self, x):
        return self.layers(x)
    
# encodert and encoders in tf code
class EncoderC(nn.Module):
    def __init__(self, keep_rate=0.75):
        super(EncoderC, self).__init__()
        self.layers = nn.Sequential(
            DRNBlock(512, padding="constant", norm_type="batch", keep_rate=keep_rate),
            DRNBlock(512, padding="constant", norm_type="batch", keep_rate=keep_rate),
            AttentionBlock(512, keep_rate=keep_rate)
        )
    
    #input == [B, C, H, W] == [B, 512, 32, 32]
    #output == [B, C, H, W] == [B, 512, 32, 32]
    def forward(self, x):
        return self.layers(x)
    
# encoder diff a and encoder diff b  in code
class EncoderS(nn.Module):
    def __init__(self, keep_rate=0.75):
        super(EncoderS, self).__init__()
        self.layers = nn.Sequential(
            GeneralConv2d(1, 8, kernel_size=7, stride=1, padding=3, norm_type="batch", keep_rate=keep_rate),
            ResNetBlock(8, norm_type="batch", padding="constant", keep_rate=keep_rate),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResNetBlockDS(8, 16, padding="constant", norm_type="batch", keep_rate=keep_rate),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResNetBlockDS(16, 32, padding="constant", norm_type="batch", keep_rate=keep_rate),
            ResNetBlock(32, norm_type="batch", padding="constant", keep_rate=keep_rate),
            nn.MaxPool2d(kernel_size=2, stride=2),
            GeneralConv2d(32, 32, kernel_size=3, stride=1, padding=1, norm_type="batch", keep_rate=keep_rate),
            GeneralConv2d(32, 32, kernel_size=3, stride=1, padding=1, norm_type="batch", keep_rate=keep_rate)
        )
    # input == [B, C, H, W] == [B, 1, 256, 256]
    # output == [B, C, H, W] == [B, 32, 32, 32]
    def forward(self, x):
        return self.layers(x)
    

####################################################################
#-------------------------- Decoders ------------------------------#
####################################################################

# decoderc in tf code
class DecoderShared(nn.Module):
    def __init__(self):
        super(DecoderShared, self).__init__()
        #in_dim = output channel encoder c (512) + output channel encoder s (32)
        self.layers = nn.Sequential(    
            GeneralConv2d(544, 128, kernel_size=3, stride=1, stddev=0.02, padding=1, norm_type="ins"),
            ResNetBlock(128, padding="constant", norm_type="ins"),
            ResNetBlock(128, padding="constant", norm_type="ins"),
            ResNetBlock(128, padding="constant", norm_type="ins"),
            ResNetBlock(128, padding="constant", norm_type="ins")
        )
    
    # input == [B, C, H, W] == [B, 544, 32, 32]
    # output == [B, C, H, W] == [B, 128, 32, 32]
    def forward(self, x):
        return self.layers(x)

# decodera and decoderb in tf code
class Decoder(nn.Module):
    def __init__(self, skip=False):
        super(Decoder, self).__init__()
        self.skip = skip
        self.layers = nn.Sequential(
            GeneralConv2d(128, 128, kernel_size=3, stride=1, stddev=0.02, padding=1, norm_type="ins"),
            ResNetBlock(128, padding="constant", norm_type="ins"),
            ResNetBlock(128, padding="constant", norm_type="ins"),
            ResNetBlock(128, padding="constant", norm_type="ins"),
            ResNetBlock(128, padding="constant", norm_type="ins"),
            GeneralDeconv2d(128, 64, kernel_size=3, stride=2, stddev=0.02, padding=1, norm_type="ins"),
            GeneralDeconv2d(64, 64, kernel_size=3, stride=2, stddev=0.02, padding=1, norm_type="ins"),
            GeneralDeconv2d(64, 32, kernel_size=3, stride=2, stddev=0.02, padding=1, norm_type="ins"),
            GeneralConv2d(32, 1, kernel_size=7, stride=1, stddev=0.02, padding=3, do_norm=False, do_relu=False)
        )
    
    # input x_de == [B, C, H, W] == [B, 128, 32, 32]
    # input x_img == [B, C, H, W] == [B, 1, 256, 256]
    # output == [B, C, H, W] == [B, 1, 256, 256]
        
    def forward(self, x_de, x_img):
        out = self.layers(x_de)
        if self.skip:
            out_t = F.tanh(out + x_img)
        else:
            out_t = F.tanh(out)
        return out_t

    

####################################################################
#---------------------- Segmentation Network ----------------------#
####################################################################

class SegmentationNetwork(nn.Module):
    def __init__(self, num_classes, keep_rate=0.75):
        super(SegmentationNetwork, self).__init__()
        self.layers = nn.Sequential(
            GeneralConv2d(128, 128, kernel_size=3, stride=1, stddev=0.02, padding=1, norm_type="ins", keep_rate=keep_rate),
            ResNetBlock(128, padding="constant", norm_type="ins"),
            ResNetBlock(128, padding="constant", norm_type="ins"),
            ResNetBlock(128, padding="constant", norm_type="ins"),
            ResNetBlock(128, padding="constant", norm_type="ins"),
            GeneralDeconv2d(128, 64, kernel_size=3, stride=2, stddev=0.02, padding=1, norm_type="ins"),
            GeneralDeconv2d(64, 64, kernel_size=3, stride=2, stddev=0.02, padding=1, norm_type="ins"),
            GeneralDeconv2d(64, 32, kernel_size=3, stride=2, stddev=0.02, padding=1, norm_type="ins"),
            GeneralConv2d(32, num_classes, kernel_size=7, stride=1, stddev=0.02, padding=3, do_norm=False, do_relu=False)
        )
    
    # input == [B, C, H, W] == [B, 512, 32, 32] --> from EncoderC
    # output == [B, C, H, W] == [B, 4, 256, 256] --> 4 channels --> num_classes!!??
    def forward(self, x):
        return self.layers(x)





