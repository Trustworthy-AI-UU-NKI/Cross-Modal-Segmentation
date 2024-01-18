import torch
import pytorch_lightning as pl
from torch.nn import BCEWithLogitsLoss
from monai.data import  decollate_batch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (  
    Activations,
    AsDiscrete,
    Compose,
)

from types import SimpleNamespace
from monai.losses import DiceLoss

class DiscriminatorLossDouble(torch.nn.Module):
    def __init__(self):
        super(DiscriminatorLossDouble, self).__init__()
        self.lossfunc = DiscriminatorLoss()

    def forward(self, prob_real_is_real, prob_fake_is_real, prob_real_is_real_aux, prob_fake_is_real_aux):
        loss = self.lossfunc(prob_real_is_real, prob_fake_is_real)
        loss_aux = self.lossfunc(prob_real_is_real_aux, prob_fake_is_real_aux)
        return loss + loss_aux
    

# Computes the LS-GAN loss as minimized by the discriminator.
class DiscriminatorLoss(torch.nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()

    def forward(self, prob_real_is_real, prob_fake_is_real):
        loss = (torch.mean(torch.pow(prob_real_is_real - 1, 2)) + torch.mean(torch.pow(prob_fake_is_real, 2))) * 0.5
        return loss
    

# Computes the zero loss
class ZeroLoss(torch.nn.Module):
    def __init__(self, lr = 0.01):
        super(ZeroLoss, self).__init__()
        self.l1_loss = torch.nn.L1Loss()
        self.lr =lr # lr5

    def forward(self, fea_A_separate_B, fea_B_separate_A, fea_FA_separate_B, fea_FB_separate_A):
        lossea = 0.01 * self.l1_loss(fea_A_separate_B, torch.zeros_like(fea_A_separate_B))
        lossesb = 0.01 * self.l1_loss(fea_B_separate_A, torch.zeros_like(fea_B_separate_A))
        lossesaf = 0.01 * self.l1_loss(fea_FA_separate_B, torch.zeros_like(fea_FA_separate_B))
        lossesbf = 0.01 * self.l1_loss(fea_FB_separate_A, torch.zeros_like(fea_FB_separate_A))
        return lossea + lossesb + lossesaf + lossesbf
    

class CycleConsistencyLoss(torch.nn.Module):
    def __init__(self, lambda_x):
        super(CycleConsistencyLoss, self).__init__()
        self.lambda_x = lambda_x

    def forward(self, real_images, generated_images):
        # Assuming real_images and generated_images are 4D tensors of shape [batch, channels, height, width]
        return self.lambda_x * torch.mean(torch.abs(real_images - generated_images))

    

class GeneratorLoss(torch.nn.Module):
    def __init__(self, lr_a = 0.01, lr_b = 0.01):
        super(GeneratorLoss, self).__init__()
        self.cycle_loss_a = CycleConsistencyLoss(lambda_x = lr_a)
        self.cycle_loss_b = CycleConsistencyLoss(lambda_x = lr_b)
    
    def forward(self, prob_fake_x_is_real, input_a, input_b, cycle_input_a, cycle_input_b):
        cycle_loss_a = self.cycle_loss_a(input_a, cycle_input_a)
        cycle_loss_b = self.cycle_loss_b(input_b, cycle_input_b)
        lsgan_loss = torch.mean(torch.pow(prob_fake_x_is_real - 1, 2)) * 0.5
        return cycle_loss_a + cycle_loss_b + lsgan_loss
        
class SegmentationLoss(torch.nn.Module):
    def __init__(self, lr_a = 0.01, lr_b = 0.01):
        super(SegmentationLoss, self).__init__()
        self.ce_loss = BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(softmax=True)
    
    def forward(self):
        return
    
class SegmentationLossTarget(torch.nn.Module):
    def __init__(self, lr_a = 0.01, lr_b = 0.01):
        super(SegmentationLossTarget, self).__init__()
        self.seg_loss_source = SegmentationLoss
    
    def forward(self):
        return