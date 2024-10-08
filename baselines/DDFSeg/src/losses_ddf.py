import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss

class DiscriminatorLossDouble(nn.Module):
    def __init__(self):
        super(DiscriminatorLossDouble, self).__init__()
        self.lossfunc = DiscriminatorLoss()

    def forward(self, prob_real_is_real, prob_fake_is_real, prob_real_is_real_aux, prob_fake_is_real_aux):
        loss = self.lossfunc(prob_real_is_real, prob_fake_is_real)
        loss_aux = self.lossfunc(prob_real_is_real_aux, prob_fake_is_real_aux)
        return loss + loss_aux
    

# Computes the LS-GAN loss as minimized by the discriminator.
class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()

    def forward(self, prob_real_is_real, prob_fake_is_real):
        loss = (torch.mean(torch.pow(prob_real_is_real - 1, 2)) + torch.mean(torch.pow(prob_fake_is_real, 2))) * 0.5
        return loss
    

# Computes the zero loss
class ZeroLoss(nn.Module):
    def __init__(self, lr = 0.01):
        super(ZeroLoss, self).__init__()
        self.l1_loss1 = torch.nn.L1Loss()
        self.l1_loss2 = torch.nn.L1Loss()
        self.l1_loss3 = torch.nn.L1Loss()
        self.l1_loss4 = torch.nn.L1Loss()
        self.lr = lr # lr5

    def forward(self, fea_A_separate_B, fea_B_separate_A, fea_FA_separate_B, fea_FB_separate_A):
        lossesa = self.lr * self.l1_loss1(fea_A_separate_B, torch.zeros_like(fea_A_separate_B))
        lossesb = self.lr * self.l1_loss2(fea_B_separate_A, torch.zeros_like(fea_B_separate_A))
        lossesaf = self.lr * self.l1_loss3(fea_FA_separate_B, torch.zeros_like(fea_FA_separate_B))
        lossesbf = self.lr * self.l1_loss4(fea_FB_separate_A, torch.zeros_like(fea_FB_separate_A))
        return lossesa + lossesb + lossesaf + lossesbf
    

class CycleConsistencyLoss(nn.Module):
    def __init__(self, lambda_x):
        super(CycleConsistencyLoss, self).__init__()
        self.lambda_x = lambda_x

    def forward(self, real_images, generated_images):
        # Assuming real_images and generated_images are 4D tensors of shape [batch, channels, height, width]
        return self.lambda_x * torch.mean(torch.abs(real_images - generated_images))

    

class GeneratorLoss(nn.Module):
    def __init__(self, lr_a = 0.01, lr_b = 0.01):
        super(GeneratorLoss, self).__init__()
        self.cycle_loss_a = CycleConsistencyLoss(lambda_x = lr_a)
        self.cycle_loss_b = CycleConsistencyLoss(lambda_x = lr_b)
    
    def forward(self, prob_fake_x_is_real, input_a, input_b, cycle_input_a, cycle_input_b):
        cycle_loss_a = self.cycle_loss_a(input_a, cycle_input_a)
        cycle_loss_b = self.cycle_loss_b(input_b, cycle_input_b)
        lsgan_loss = torch.mean(torch.pow(prob_fake_x_is_real - 1, 2)) * 0.5
        return cycle_loss_a + cycle_loss_b + lsgan_loss

class SoftmaxWeightedLoss(nn.Module):
    def __init__(self):
        super(SoftmaxWeightedLoss, self).__init__()

    def forward(self, logits, gt):

        num_classes = logits.shape[1]  # Assuming logits are of shape [N, C, H, W]
        softmaxpred = F.softmax(logits, dim=1)

        raw_loss = 0
        for i in range(num_classes):
            gti = gt[:, i, ...]
            predi = softmaxpred[:, i, ...]
            weighted = 1 - torch.sum(gti) / torch.sum(gt)
            raw_loss = raw_loss - (weighted * gti * torch.log(torch.clamp(predi, min=0.005, max=1)))
    

        loss = torch.mean(raw_loss)
        return loss
        
class SegmentationLoss(nn.Module):
    def __init__(self, lr_a=0.01, lr_b=0.01):
        super(SegmentationLoss, self).__init__()
        self.wce_loss = SoftmaxWeightedLoss()
        self.dice_loss = DiceLoss(softmax=True)
        self.gen_loss = GeneratorLoss(lr_a=lr_a, lr_b=lr_b)
    
    def forward(self, logits, gt, model, prob_fake_x_is_real, input_a, input_b, cycle_input_a, cycle_input_b):
        gt_oh = F.one_hot(gt.long().squeeze(1), num_classes=logits.shape[1]).permute(0, 3, 2, 1)
        dice_loss = self.dice_loss(logits, gt_oh)
        ce_loss = self.wce_loss(logits, gt_oh)
        
        l2_loss = 0
        for name, param in model.named_parameters():
            l2_loss = l2_loss + (0.0001 * torch.sum(param.detach() ** 2))  # Directly calculate L2 norm


        gen_loss = self.gen_loss(prob_fake_x_is_real, input_a, input_b, cycle_input_a, cycle_input_b)

        return dice_loss + ce_loss + l2_loss + 0.1 * gen_loss
    
class SegmentationLossTarget(nn.Module):
    def __init__(self, lr_a=0.01, lr_b=0.01):
        super(SegmentationLossTarget, self).__init__()
        self.seg_loss_source = SegmentationLoss(lr_a, lr_b)
    
    def forward(self, logits, gt, model_params, prob_fake_x_is_real, input_a, input_b, cycle_input_a, cycle_input_b, loss_f_weight_value, prob_fea_b_is_real, prob_fake_a_aux_is_real):
        seg_loss_source = self.seg_loss_source(logits, gt, model_params, prob_fake_x_is_real, input_a, input_b, cycle_input_a, cycle_input_b)
        ls_gan_loss_f = torch.mean(torch.pow(prob_fea_b_is_real - 1, 2)) * 0.5
        lsgan_loss_a_aux = torch.mean(torch.pow(prob_fake_a_aux_is_real - 1, 2)) * 0.5
        extra_loss = loss_f_weight_value * ls_gan_loss_f + 0.1*lsgan_loss_a_aux
        return seg_loss_source + extra_loss