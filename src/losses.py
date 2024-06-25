import torch
import torch.nn as nn

# LSGAN generator and discriminator losses
class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()

    def forward(self, prob_real_is_real, prob_fake_is_real):
        loss = (torch.mean(torch.pow(prob_real_is_real - 1, 2)) + torch.mean(torch.pow(prob_fake_is_real, 2))) * 0.5
        return loss
    
class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
    
    def forward(self, prob_fake_repr_is_real):
        lsgan_loss = torch.mean(torch.pow(prob_fake_repr_is_real - 1, 2)) * 0.5
        return lsgan_loss
