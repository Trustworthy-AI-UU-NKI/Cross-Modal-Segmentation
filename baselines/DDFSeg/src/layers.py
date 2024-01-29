
import torch
import torch.nn as nn
import torch.nn.functional as F


####################################################################
#---------------------------- Layers ------------------------------#
####################################################################
# type = 0: build_resnet_block
# type = 1: build_resnet_block_ins

class ResNetBlock(nn.Module):
    def __init__(self, dim, norm_type=None, block_type = 0, padding = "REFLECT", keep_rate=0.75):
        super(ResNetBlock, self).__init__()
        self.keep_rate = keep_rate
        self.norm_type = norm_type
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
    def __init__(self, dim_in, dim_out, norm_type=None, padding = "REFLECT", keep_rate=0.75):
        super(ResNetBlockDS, self).__init__()
        self.keep_rate = keep_rate
        self.norm_type = norm_type
        self.padding = padding
        self.padding_dim = (dim_out - dim_in) // 2

        # Define the first convolutional layers
        self.conv1 = GeneralConv2d(dim_in, dim_out, kernel_size=3, stride=1, keep_rate=keep_rate, norm_type=norm_type)
        self.conv2 = GeneralConv2d(dim_in, dim_out, kernel_size=3, stride=1, do_relu=False, keep_rate=keep_rate, norm_type=norm_type)
        
    def forward(self, x):
        # Apply the first convolutional layer with reflect padding
        out = F.pad(x, (1, 1, 1, 1), mode=self.padding)
        out = self.conv1(out)

        # Apply the second convolutional layer with reflect padding
        out = F.pad(out, (1, 1, 1, 1), mode=self.padding)
        out = self.conv2(out)
        

        # Residual connection
        return F.relu(out + x)












####################################################################
#-------------------- Basic Building Blocks -----------------------#
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
