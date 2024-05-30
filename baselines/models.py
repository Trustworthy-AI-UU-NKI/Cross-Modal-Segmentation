import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import UNet

class CustomUNet(UNet):
    def __init__(self, *args, **kwargs):
        super(CustomUNet, self).__init__(*args, **kwargs)
        self.seq = nn.Sequential(
            self.model[0],
            self.model[1].submodule[:],
        )

    def forward(self, x):
        out = self.seq(x)
        return out
    
    