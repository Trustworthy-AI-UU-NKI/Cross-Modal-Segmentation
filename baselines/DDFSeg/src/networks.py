import  torch
import torch.nn as nn
from torch.autograd import Variable
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F

####################################################################
#------------------------- Discriminators -------------------------#
####################################################################








####################################################################
#-------------------------- Encoders ------------------------------#
####################################################################






####################################################################
#-------------------------- Decoders ------------------------------#
####################################################################






####################################################################
#---------------------- Segmentation Network ----------------------#
####################################################################






####################################################################
#-------------------- Basic Building Blocks -----------------------#
####################################################################

class CustomLeakyReLU(nn.Module):
    def __init__(self, leak=0.2, alt_relu_impl=False):
        super(CustomLeakyReLU, self).__init__()
        self.leak = leak
        self.alt_relu_impl = alt_relu_impl
        if not alt_relu_impl:
            # Use PyTorch's predefined LeakyReLU for the standard implementation
            self.leaky_relu = nn.LeakyReLU(negative_slope=leak)

    def forward(self, x):
        if self.alt_relu_impl:
            # Alternative implementation of LeakyReLU
            f1 = 0.5 * (1 + self.leak)
            f2 = 0.5 * (1 - self.leak)
            return f1 * x + f2 * torch.abs(x)
        else:
            # Use the predefined LeakyReLU function
            return self.leaky_relu(x)

# For GA, set stddev to 0.02 and biasInit to True
class GeneralConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=7, stride=1, padding='valid',
                 stddev=0.01, do_norm=True, do_relu=True, keep_rate=None, relu_factor=0, norm_type=None, bias_init=False):
        super(GeneralConv2d, self).__init__()
        
        # Convolution
        if padding.lower() == 'valid':
            padding = 0
        elif padding.lower() == 'same':
            padding = kernel_size // 2
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding=padding,
                              bias=(not do_norm))  # No bias if normalization is applied

        # Weight initialization
        nn.init.truncated_normal_(self.conv.weight, std=stddev)
        if bias_init:
            nn.init.constant_(self.conv.bias, 0.0)

        # Dropout
        self.dropout = nn.Dropout(1 - keep_rate) if keep_rate is not None else None

        # Normalization
        if do_norm:
            if norm_type is None:
                raise ValueError("Normalization type not specified")
            elif norm_type.lower() == 'ins':
                self.norm = nn.InstanceNorm2d(output_channels)
            elif norm_type.lower() == 'batch':
                self.norm = nn.BatchNorm2d(output_channels)
            else:
                raise ValueError("Unknown normalization type")
        else:
            self.norm = None

        # Activation
        self.do_relu = do_relu
        self.relu_factor = relu_factor

    def forward(self, x):
        x = self.conv(x)
        if self.dropout:
            x = self.dropout(x)
        if self.norm:
            x = self.norm(x)
        if self.do_relu:
            if self.relu_factor == 0:
                x = F.relu(x)
            else:
                x = F.leaky_relu(x, negative_slope=self.relu_factor)
        return x
    


class DilateConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=7, dilation_rate=2, padding='valid',
                 stddev=0.01, do_norm=True, do_relu=True, keep_rate=None, relu_factor=0, norm_type=None):
        super(DilateConv2d, self).__init__()

        # Adjust padding based on dilation rate to maintain output size
        if padding.lower() == 'valid':
            padding = 0
        elif padding.lower() == 'same':
            padding = ((kernel_size - 1) * (dilation_rate - 1)) // 2

        # Dilated Convolution
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size,
                              dilation=dilation_rate, padding=padding, bias=False)

        # Weight and bias initialization
        nn.init.truncated_normal_(self.conv.weight, std=stddev)
        self.conv.bias = nn.Parameter(torch.zeros(output_channels))

        # Dropout
        self.dropout = nn.Dropout(1 - keep_rate) if keep_rate is not None else None

        # Normalization
        if do_norm:
            if norm_type is None:
                raise ValueError("Normalization type not specified")
            elif norm_type.lower() == 'ins':
                self.norm = nn.InstanceNorm2d(output_channels)
            elif norm_type.lower() == 'batch':
                self.norm = nn.BatchNorm2d(output_channels)
            else:
                raise ValueError("Unknown normalization type")
        else:
            self.norm = None

        # Activation
        self.do_relu = do_relu
        self.relu_factor = relu_factor

    def forward(self, x):
        x = self.conv(x)
        if self.dropout:
            x = self.dropout(x)
        if self.norm:
            x = self.norm(x)
        if self.do_relu:
            if self.relu_factor == 0:
                x = F.relu(x)
            else:
                x = F.leaky_relu(x, negative_slope=self.relu_factor)
        return x


class GeneralDeconv2d(nn.Module):
    def __init__(self, input_channels, output_channels, output_shape, kernel_size=7, stride=1, padding='valid',
                 stddev=0.02, do_norm=True, do_relu=True, relu_factor=0, norm_type=None):
        super(GeneralDeconv2d, self).__init__()

        # Adjust padding based on the desired output
        if padding.lower() == 'valid':
            padding = 0
        elif padding.lower() == 'same':
            padding = kernel_size // 2

        # Transposed Convolution
        self.conv = nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride=stride, padding=padding)

        # Weight and bias initialization
        nn.init.truncated_normal_(self.conv.weight, std=stddev)
        nn.init.constant_(self.conv.bias, 0.0)

        # Normalization
        if do_norm:
            if norm_type is None:
                raise ValueError("Normalization type not specified")
            elif norm_type.lower() == 'ins':
                self.norm = nn.InstanceNorm2d(output_channels)
            elif norm_type.lower() == 'batch':
                self.norm = nn.BatchNorm2d(output_channels)
            else:
                raise ValueError("Unknown normalization type")
        else:
            self.norm = None

        # Activation
        self.do_relu = do_relu
        self.relu_factor = relu_factor

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.do_relu:
            if self.relu_factor == 0:
                x = F.relu(x)
            else:
                x = F.leaky_relu(x, negative_slope=self.relu_factor)
        return x

