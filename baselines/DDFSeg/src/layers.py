
import torch
import torch.nn as nn
import torch.nn.functional as F


####################################################################
#-------------------------- Basic Blocks --------------------------#
####################################################################
# type = 0: build_resnet_block
# type = 1: build_resnet_block_ins

class ResNetBlock(nn.Module):
    def __init__(self, dim, norm_type=None, block_type = 0, padding = "reflect", keep_rate=0.75):
        super(ResNetBlock, self).__init__()
        self.padding = padding
        
        if block_type == 0:
            # Define the first convolutional layers
            self.conv1 = GeneralConv2d(dim, dim, kernel_size=3, stride=1, keep_rate=keep_rate, norm_type=norm_type)
            self.conv2 = GeneralConv2d(dim, dim, kernel_size=3, stride=1, do_relu=False, keep_rate=keep_rate, norm_type=norm_type)
        elif block_type == 1: # instance normalization without dropout
            self.conv1 = GeneralConv2d(dim, dim, kernel_size=3, stride=1, stddev=0.02, norm_type='Ins', bias_init=True)
            self.conv2 = GeneralConv2d(dim, dim, kernel_size=3, stride=1, do_relu=False, stddev=0.02, norm_type='Ins', bias_init=True)

    def forward(self, x):
        # Apply the first convolutional layer with reflect padding
        out = F.pad(x, (1, 1, 1, 1), mode=self.padding)
        out = self.conv1(out)

        # Apply the second convolutional layer with reflect padding
        out = F.pad(out, (1, 1, 1, 1), mode=self.padding)
        out = self.conv2(out)

        # Residual connection
        return F.relu(out + x)
    
class ResNetBlockDS(nn.Module):
    def __init__(self, dim_in, dim_out, norm_type=None, padding = "reflect", keep_rate=0.75):
        super(ResNetBlockDS, self).__init__()
        self.padding = padding
        self.padding_dim = (dim_out - dim_in) // 2

        # Define the first convolutional layers
        self.conv1 = GeneralConv2d(dim_in, dim_out, kernel_size=3, stride=1, keep_rate=keep_rate, norm_type=norm_type)
        self.conv2 = GeneralConv2d(dim_out, dim_out, kernel_size=3, stride=1, do_relu=False, keep_rate=keep_rate, norm_type=norm_type)
        
    def forward(self, x):
        # Apply the first convolutional layer with reflect padding
        out = F.pad(x, (1, 1, 1, 1), mode=self.padding)
        out = self.conv1(out)

        # Apply the second convolutional layer with reflect padding
        out = F.pad(out, (1, 1, 1, 1), mode=self.padding)
        out = self.conv2(out)
        
        padded_x = F.pad(padded_x, (0, 0, 0, 0, self.padding_dim, self.padding_dim), mode=self.padding)

        # Residual connection
        return F.relu(out + padded_x)

class DRNBlock(nn.Module):
    def __init__(self, dim, norm_type=None, padding="reflect", keep_rate=0.75):
        super(DRNBlock, self).__init__()
        self.padding = padding
        self.dconv1 = DilateConv2d(dim, dim, kernel_size=3, keep_rate=keep_rate, norm_type=norm_type)
        self.dconv2 = DilateConv2d(dim, dim, kernel_size=3, do_relu=False, keep_rate=keep_rate, norm_type=norm_type)


    def forward(self, x):
        out = F.pad(x, (2, 2, 2, 2), mode=self.padding)
        out = self.dconv1(out)
        out = F.pad(out, (2, 2, 2, 2), mode=self.padding)
        out = self.dconv2(out)

        return F.relu(out + x)
    

class DRNBlockDS(nn.Module):
    def __init__(self, dim_in, dim_out, norm_type=None, padding="reflect", keep_rate=0.75):
        super(DRNBlockDS, self).__init__()
        self.padding = padding
        self.padding_dim = (dim_out-dim_in) // 2
        self.dconv1 = DilateConv2d(dim_in, dim_out, kernel_size=3, keep_rate=keep_rate, norm_type=norm_type)
        self.dconv2 = DilateConv2d(dim_in, dim_out, kernel_size=3, do_relu=False, keep_rate=keep_rate, norm_type=norm_type)


    def forward(self, x):
        out = F.pad(x, (2, 2, 2, 2), mode=self.padding)
        out = self.dconv1(out)
        out = F.pad(out, (2, 2, 2, 2), mode=self.padding)
        out = self.dconv2(out)

        padded_x = F.pad(x, (0, 0, 0, 0, self.padding_dim, self.padding_dim), mode=self.padding)

        return F.relu(out + padded_x)

class AttentionBlock(nn.Module):
    def __init__(self, ch, keep_rate=0.75):
        super(AttentionBlock, self).__init__()

        # stride and kernel_size is 0 --> no padding needed
        self.f_conv = GeneralConv2d(ch, ch//8, kernel_size=1, stride=1, norm_type = "batch", keep_rate=keep_rate)
        self.g_conv = GeneralConv2d(ch, ch//8, kernel_size=1, stride=1,  norm_type = "batch", keep_rate=keep_rate)
        self.h_conv = GeneralConv2d(ch, ch//2, kernel_size=1, stride=1, norm_type = "batch", keep_rate=keep_rate)
        self.o_conv = GeneralConv2d(ch, ch, kernel_size=1, stride=1, norm_type = "batch", keep_rate=keep_rate)

        self.gamma = nn.Parameter(torch.zeros(1))  # Trainable scalar

    def forward(self, x): 
        batch_size, num_channels, height, width = x.size()

        f = self.f_conv(x)
        f = F.max_pool2d(f, kernel_size=2, stride=2)  # Pooling for f

        g = self.g_conv(x)

        h = self.h_conv(x)
        h = F.max_pool2d(h, kernel_size=2, stride=2)  # Pooling for h

        # Flatten and perform matrix multiplication
        s = torch.matmul(self.hw_flatten(g), self.hw_flatten(f).transpose(-2, -1))
        beta = F.softmax(s, dim=-1)  # Attention map

        o = torch.matmul(beta, self.hw_flatten(h))  # [bs, N, C]
        o = o.view(batch_size, height, width, num_channels // 2).permute(0, 3, 1, 2)

        o = self.out_conv(o)

        x = self.gamma * o + x
        return x

    
    def hw_flatten(x):
        # Flatten height and width dimensions
        return x.view(x.size(0), x.size(1), -1)

    

####################################################################
#------------------------- Basic Layers ---------------------------#
####################################################################

class CustomLeakyReLU(nn.Module):
    def __init__(self, leak=0.2):
        super(CustomLeakyReLU, self).__init__()
        self.leak = leak

    def forward(self, x):
        # Alternative implementation of LeakyReLU
        f1 = 0.5 * (1 + self.leak)
        f2 = 0.5 * (1 - self.leak)
        return f1 * x + f2 * torch.abs(x)

# For GA, set stddev to 0.02 and biasInit to True
class GeneralConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=7, stride=1, padding=0,
                 stddev=0.01, do_norm=True, do_relu=True, keep_rate=None, relu_factor=0, norm_type=None, bias_init=False):
        super(GeneralConv2d, self).__init__()
        

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
    def __init__(self, input_channels, output_channels, kernel_size=7, dilation_rate=2, padding=0,
                 stddev=0.01, do_norm=True, do_relu=True, keep_rate=None, relu_factor=0, norm_type=None):
        super(DilateConv2d, self).__init__()

        # Adjust padding based on dilation rate to maintain output size


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

#Padding = "valid --> 0"
#Padding = "same --> depends on input"
class GeneralDeconv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=7, stride=1, padding=0,
                 stddev=0.02, do_norm=True, do_relu=True, relu_factor=0, norm_type=None):
        super(GeneralDeconv2d, self).__init__()

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
