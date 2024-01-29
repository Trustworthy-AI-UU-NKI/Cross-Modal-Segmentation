import  torch
import torch.nn as nn
from torch.autograd import Variable
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F

####################################################################
#------------------------- Discriminators -------------------------#
####################################################################



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
    
    def forward(self, x):
        return x
    

class DiscriminatorAux(nn.Module):
    def __init__(self):
        super(DiscriminatorAux, self).__init__()
    
    def forward(self, x):
        return x




####################################################################
#-------------------------- Encoders ------------------------------#
####################################################################

class EncoderC(nn.Module):
    def __init__(self):
        super(EncoderC, self).__init__()
    
    def forward(self, x):
        return x
    

class EncoderS(nn.Module):
    def __init__(self):
        super(EncoderS, self).__init__()
    
    def forward(self, x):
        return x
    

class EncoderT(nn.Module):
    def __init__(self):
        super(EncoderT, self).__init__()
    
    def forward(self, x):
        return x


# encoder diff a and encoder diff b ???????

####################################################################
#-------------------------- Decoders ------------------------------#
####################################################################

class DecoderC(nn.Module):
    def __init__(self):
        super(DecoderC, self).__init__()
    
    def forward(self, x):
        return x

class DecoderA(nn.Module):
    def __init__(self):
        super(DecoderA, self).__init__()
    
    def forward(self, x):
        return x

class DecoderB(nn.Module):
    def __init__(self):
        super(DecoderB, self).__init__()
    
    def forward(self, x):
        return x
    



####################################################################
#---------------------- Segmentation Network ----------------------#
####################################################################

class SegmentationNetwork(nn.Module):
    def __init__(self):
        super(SegmentationNetwork, self).__init__()
    
    def forward(self, x):
        return x





