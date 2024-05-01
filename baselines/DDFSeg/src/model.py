import torch
import torch.nn as nn
import numpy as np
from losses import *
from networks import *
import random
from torch.optim import lr_scheduler


class DDFSeg(nn.Module):
    def __init__(self, args, channel_im = 1, num_classes=2, pool_size = 50):
        super(DDFSeg, self).__init__()

        # hyperparameters
        self.learning_rate = args.lr
        self.learning_rate_seg = args.lr67
        self.learning_rate_5 = args.lr5
        self.lr_A = args.lr_A
        self.lr_B = args.lr_B
        self.num_fake_inputs = 0
        self._pool_size = pool_size
        self.bs = args.bs
        self.img_res = args.resolution
        self.skip = args.skip
        self.keep_rate = args.keep_rate 
        self.val_dice = -1
        self.num_classes = num_classes 
        self.im_channels = channel_im

        self.fake_images_A = np.zeros(
            (self._pool_size, self.bs, 1, self.img_res, self.img_res)
        )
        self.fake_images_B = np.zeros(
            (self._pool_size, self.bs, 1, self.img_res, self.img_res)
        )

        # networks
        self.discriminator_A = Discriminator(channels_in=channel_im, aux=True) # d_A
        self.discriminator_B = Discriminator(channels_in=channel_im,) # d_B
        self.discriminator_F = Discriminator(channels_in=self.num_classes) # d_F
        self.encoder_C_AB = EncoderCShared(keep_rate=self.keep_rate) # e_c
        self.encoder_C_A = EncoderC(keep_rate=self.keep_rate) #e_cs
        self.encoder_C_B = EncoderC(keep_rate=self.keep_rate) # e_ct
        self.encoder_D_A = EncoderS(keep_rate=self.keep_rate) # e_dA
        self.encoder_D_B = EncoderS(keep_rate=self.keep_rate) # e_dB
        self.decoder_AB = DecoderShared() # de_c
        self.decoder_A = Decoder(channels_out=channel_im, skip=self.skip) # de_A
        self.decoder_B = Decoder(channels_out=channel_im, skip=self.skip) # de_B
        self.segmenter = SegmentationNetwork(num_classes=self.num_classes, keep_rate=self.keep_rate) # s_A

        # optimizers + according losses
        # Source (A) Discriminator update
        self.d_A_trainer = torch.optim.Adam(self.discriminator_A.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        self.d_A_loss = DiscriminatorLossDouble()

        # Target (B) Discriminator update
        self.d_B_trainer = torch.optim.Adam(self.discriminator_B.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        self.d_B_loss = DiscriminatorLoss()

        # Domain specific encoders update --> zero loss
        self.dif_trainer = torch.optim.Adam(list(self.encoder_D_A.parameters()) + list(self.encoder_D_B.parameters()), lr=self.learning_rate, betas=(0.5, 0.999))
        self.dif_loss = ZeroLoss(self.learning_rate_5)

        # Generator update --> de_A_vars+de_c_vars+e_c_vars+e_cs_vars+e_dB_vars
        self.g_A_trainer = torch.optim.Adam(list(self.decoder_A.parameters()) + list(self.decoder_AB.parameters()) + list(self.encoder_C_AB.parameters()) 
                                            + list(self.encoder_C_A.parameters()) + list(self.encoder_D_B.parameters()), lr=self.learning_rate, betas=(0.5, 0.999))
        self.g_loss_A = GeneratorLoss(self.lr_A, self.lr_B)

        # Generator update --> de_B_vars+de_c_vars+e_c_vars+e_ct_vars+e_dA_vars
        self.g_B_trainer = torch.optim.Adam(list(self.decoder_B.parameters()) + list(self.decoder_AB.parameters()) + list(self.encoder_C_AB.parameters())
                                            + list(self.encoder_C_B.parameters()) + list(self.encoder_D_A.parameters()), lr=self.learning_rate, betas=(0.5, 0.999))
        self.g_loss_B = GeneratorLoss(self.lr_A, self.lr_B)

        # Updating segmentation network via target images
        self.s_B_trainer = torch.optim.Adam(list(self.encoder_C_AB.parameters()) + list(self.encoder_C_B.parameters()) + list(self.segmenter.parameters()), lr=self.learning_rate_seg)
        self.seg_loss_B = SegmentationLossTarget(self.lr_A, self.lr_B)

        # Updating segmentation network via source images
        self.s_A_trainer = torch.optim.Adam(list(self.encoder_C_AB.parameters()) + list(self.encoder_C_A.parameters()) + list(self.segmenter.parameters()), lr=self.learning_rate_seg)
        self.seg_loss_A = SegmentationLoss(self.lr_A, self.lr_B)

        # Feature Discriminator (Dis_seg)
        self.d_F_trainer = torch.optim.Adam(self.discriminator_F.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        self.d_F_loss = DiscriminatorLoss()

    
    def update_num_fake_inputs(self):
        self.num_fake_inputs += 1


    def set_scheduler(self, last_ep=0):
        # Assuming optimizer is already created and configured
        for param_group in self.s_B_trainer.param_groups:
            param_group['initial_lr'] = param_group['lr']

        self.s_B_sch = self.get_scheduler(self.s_B_trainer, last_ep)

        # Assuming optimizer is already created and configured
        for param_group in self.s_A_trainer.param_groups:
            param_group['initial_lr'] = param_group['lr']
        self.s_A_sch = self.get_scheduler(self.s_A_trainer, last_ep)


    def lambda_rule(self, ep):
        lr_l = self.learning_rate_seg
        if ep > 0 and ep%2==0:
            lr_l = np.multiply(self.learning_rate_seg, 0.9)
        return lr_l
    

    def get_scheduler(self, optimizer, cur_ep=-1):
        def lambda_rule(ep):
            lr_l = self.learning_rate_seg
            if ep > 0 and ep%2==0:
                lr_l = np.multiply(self.learning_rate_seg, 0.9)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=cur_ep)
        return scheduler

    def update_lr(self, ep):
        self.s_B_sch.step()
        self.s_A_sch.step()
        self.learning_rate_seg = self.lambda_rule(ep)


    def fake_image_pool_A(self, fake):
        if self.num_fake_inputs < self._pool_size:
            self.fake_images_A[self.num_fake_inputs] = fake.cpu().detach()
            return fake
        else:
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0, self._pool_size - 1)
                temp = torch.from_numpy(self.fake_images_A[random_id])
                temp_return = temp.to(fake.dtype).to(self.device)
                self.fake_images_A[random_id] = fake.cpu().detach()
                return temp_return
            else:
                return fake
            
    def fake_image_pool_B(self, fake):
        if self.num_fake_inputs < self._pool_size:
            self.fake_images_B[self.num_fake_inputs] = fake.cpu().detach()
            return fake
        else:
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0, self._pool_size - 1)
                temp = torch.from_numpy(self.fake_images_B[random_id])
                temp_return = temp.to(fake.dtype).to(self.device)
                self.fake_images_B[random_id] = fake.cpu().detach()
                return temp_return
            else:
                return fake

    def setgpu(self, device):
        print("putting model to device: ", device)
        self.device = device
        self.encoder_C_AB.to(device)
        self.encoder_C_A.to(device)
        self.encoder_C_B.to(device)
        self.encoder_D_A.to(device)
        self.encoder_D_B.to(device)
        self.decoder_AB.to(device)
        self.decoder_A.to(device)
        self.decoder_B.to(device)
        self.segmenter.to(device)
        self.discriminator_A.to(device)
        self.discriminator_B.to(device)
        self.discriminator_F.to(device)


    def reset_losses(self):
        self.g_loss_A_item = 0
        self.d_B_loss_item = 0
        self.s_B_loss_item = 0
        self.s_A_loss_item = 0
        self.g_loss_B_item = 0
        self.d_A_loss_item = 0
        self.d_F_loss_item = 0
        self.dif_loss_item = 0
        
    def forward_eval(self, images_a, images_b):
        latent_tmpb = self.encoder_C_AB(images_b)
        latent_b = self.encoder_C_B(latent_tmpb)

        pred_mask = self.segmenter(latent_b) 

        latent_tmpa = self.encoder_C_AB(images_a)
        latent_a = self.encoder_C_A(latent_tmpa)
        latent_b_diff = self.encoder_D_B(images_b)

        # couple style from B and content from A to create fake image
        fake_images_tmp_b = self.decoder_AB(torch.cat((latent_a, latent_b_diff), dim=1))
        fake_images_b = self.decoder_B(fake_images_tmp_b, images_a[:, 1, :, :].unsqueeze(1))
        # fake_images_b = self.decoder_B(fake_images_tmp_b, images_a)


        # Find content from fake image
        if self.im_channels == 1:
            latent_fake_btmp = self.encoder_C_AB(torch.cat((fake_images_b, fake_images_b, fake_images_b), dim=1))
        else:
            latent_fake_btmp = self.encoder_C_AB(fake_images_b)

        # latent_fake_btmp = self.encoder_C_AB(fake_images_b)
        latent_fake_b = self.encoder_C_B(latent_fake_btmp)

        pred_mask_fake = self.segmenter(latent_fake_b)

        softmax_output = F.softmax(pred_mask, dim=1)
        softmax_output = softmax_output.clamp(-1e15, 1e15) 
        

        softmax_output_fake = F.softmax(pred_mask_fake, dim=1)
        softmax_output_fake = softmax_output_fake.clamp(-1e15, 1e15)

        return softmax_output, softmax_output_fake


    def forward_G(self, images_a, images_b):
        latent_tmpa = self.encoder_C_AB(images_a)
        latent_tmpb = self.encoder_C_AB(images_b)
        latent_a = self.encoder_C_A(latent_tmpa)
        latent_b = self.encoder_C_B(latent_tmpb)

        latent_a_diff = self.encoder_D_A(images_a)
        latent_b_diff = self.encoder_D_B(images_b)

        fake_images_tmp_b = self.decoder_AB(torch.cat((latent_a, latent_b_diff), dim=1))
        fake_images_tmp_a = self.decoder_AB(torch.cat((latent_b, latent_a_diff), dim=1))
        fake_images_b = self.decoder_B(fake_images_tmp_b, images_a[:, 1, :, :].unsqueeze(1))
        fake_images_a = self.decoder_A(fake_images_tmp_a, images_b[:, 1, :, :].unsqueeze(1))
        # fake_images_b = self.decoder_B(fake_images_tmp_b, images_a)
        # fake_images_a = self.decoder_A(fake_images_tmp_a, images_b)

        # Cross cycle

        if self.im_channels == 1:
            latent_fake_btmp = self.encoder_C_AB(torch.cat((fake_images_b, fake_images_b, fake_images_b), dim=1))
            latent_fake_atmp = self.encoder_C_AB(torch.cat((fake_images_a, fake_images_a, fake_images_a), dim=1))
            latent_fa_diff = self.encoder_D_A(torch.cat((fake_images_a, fake_images_a, fake_images_a), dim=1))
            latent_fb_diff = self.encoder_D_B(torch.cat((fake_images_b, fake_images_b, fake_images_b), dim=1))
        else:
            latent_fake_btmp = self.encoder_C_AB(fake_images_b)
            latent_fake_atmp = self.encoder_C_AB(fake_images_a)
            latent_fa_diff = self.encoder_D_A(fake_images_a)
            latent_fb_diff = self.encoder_D_B(fake_images_b)
        

        latent_fake_a = self.encoder_C_A(latent_fake_atmp)
        latent_fake_b = self.encoder_C_B(latent_fake_btmp)

        cycle_images_tmp_b = self.decoder_AB(torch.cat((latent_fake_a, latent_fb_diff), dim=1))
        cycle_images_tmp_a = self.decoder_AB(torch.cat((latent_fake_b, latent_fa_diff), dim=1))
        cycle_images_b = self.decoder_B(cycle_images_tmp_b, fake_images_a)
        cycle_images_a = self.decoder_A(cycle_images_tmp_a, fake_images_b)

        return {"fake_images_a": fake_images_a,
                "fake_images_b": fake_images_b,
                "cycle_images_a": cycle_images_a,
                "cycle_images_b": cycle_images_b}
        
    def forward_GA_DB(self, images_a, images_b):
        # print("In forward pass GA DB")
        forward_GA_res = self.forward_G(images_a, images_b)
        prob_fake_a_is_real, _ = self.discriminator_A(forward_GA_res["fake_images_a"])
        
        prob_real_b_is_real = self.discriminator_B(images_b[:, 1, :, :].unsqueeze(1).detach())
        # prob_real_b_is_real = self.discriminator_B(images_b.detach())

        fake_pool_b = self.fake_image_pool_B(forward_GA_res["fake_images_b"].detach())

        prob_fake_pool_b_is_real = self.discriminator_B(fake_pool_b.detach())

        return {
                "cycle_images_a": forward_GA_res["cycle_images_a"],
                "cycle_images_b": forward_GA_res["cycle_images_b"], 
                "prob_real_b_is_real": prob_real_b_is_real,
                "prob_fake_pool_b_is_real": prob_fake_pool_b_is_real,
                "prob_fake_a_is_real": prob_fake_a_is_real
        }

    def forward_GB_DA(self, images_a, images_b):
        # print("In forward pass GB DA")
        forward_GB_res = self.forward_G(images_a, images_b)
        prob_real_a_is_real, prob_real_a_aux = self.discriminator_A(images_a[:, 1, :, :].unsqueeze(1).detach())
        # prob_real_a_is_real, prob_real_a_aux = self.discriminator_A(images_a.detach())

        prob_cycle_a_is_real, prob_cycle_a_aux_is_real = self.discriminator_A(forward_GB_res["cycle_images_a"].detach())
        prob_fake_b_is_real = self.discriminator_B(forward_GB_res["fake_images_b"])

        fake_pool_a = self.fake_image_pool_A(forward_GB_res["fake_images_b"].detach())
        prob_fake_pool_a_is_real, prob_fake_pool_a_aux_is_real = self.discriminator_A(fake_pool_a.detach())

        return {
                "cycle_images_a": forward_GB_res["cycle_images_a"],
                "cycle_images_b": forward_GB_res["cycle_images_b"], 
                "prob_fake_b_is_real": prob_fake_b_is_real, 
                "prob_real_a_is_real": prob_real_a_is_real, 
                "prob_fake_pool_a_is_real": prob_fake_pool_a_is_real,
                "prob_cycle_a_aux_is_real": prob_cycle_a_aux_is_real,
                "prob_fake_pool_a_aux_is_real": prob_fake_pool_a_aux_is_real
        }

    def update_GA_DB(self, images_a, images_b):
        res = self.forward_GA_DB(images_a, images_b)
        loss_gA = self.g_loss_A(res["prob_fake_a_is_real"], images_a[:, 1, :, :].unsqueeze(1).detach(), images_b[:, 1, :, :].unsqueeze(1).detach(), res["cycle_images_a"], res["cycle_images_b"])
        # loss_gA = self.g_loss_A(res["prob_fake_a_is_real"], images_a.detach(), images_b.detach(), res["cycle_images_a"], res["cycle_images_b"])
        
        loss_dB = self.d_B_loss(res["prob_real_b_is_real"], res["prob_fake_pool_b_is_real"])

        # update GA 
        # print("Update GA")
        self.g_A_trainer.zero_grad()
        loss_gA.backward()
        self.g_loss_A_item += loss_gA.item()
        self.g_A_trainer.step()

        # update DB
        # print("Update DB")
        self.d_B_trainer.zero_grad()
        loss_dB.backward()
        self.d_B_loss_item += loss_dB.item()
        self.d_B_trainer.step()

    def update_GB_DA(self, images_a, images_b):
        res = self.forward_GB_DA(images_a, images_b)

        loss_gB = self.g_loss_B(res["prob_fake_b_is_real"], images_a[:, 1, :, :].unsqueeze(1).detach(), images_b[:, 1, :, :].unsqueeze(1).detach(), res["cycle_images_a"], res["cycle_images_b"])
        # loss_gB = self.g_loss_B(res["prob_fake_b_is_real"], images_a.detach(), images_b.detach(), res["cycle_images_a"], res["cycle_images_b"])
        
        loss_dA = self.d_A_loss(res["prob_real_a_is_real"], res["prob_fake_pool_a_is_real"], res["prob_cycle_a_aux_is_real"], res["prob_fake_pool_a_aux_is_real"])
    
        # update GB
        # print("Update GB")
        self.g_B_trainer.zero_grad()
        loss_gB.backward()
        self.g_loss_B_item += loss_gB.item()
        self.g_B_trainer.step()

        # update DA
        # print("Update DA")
        self.d_A_trainer.zero_grad()
        loss_dA.backward()
        self.d_A_loss_item += loss_dA.item()
        self.d_A_trainer.step()

    def forward_SB(self, images_a, images_b):
        latent_tmpa = self.encoder_C_AB(images_a)
        latent_tmpb = self.encoder_C_AB(images_b)
        latent_a = self.encoder_C_A(latent_tmpa)
        latent_b = self.encoder_C_B(latent_tmpb)

        pred_mask_b = self.segmenter(latent_b)

        latent_a_diff = self.encoder_D_A(images_a)
        latent_b_diff = self.encoder_D_B(images_b)

        fake_images_tmp_b = self.decoder_AB(torch.cat((latent_a, latent_b_diff), dim=1))
        fake_images_tmp_a = self.decoder_AB(torch.cat((latent_b, latent_a_diff), dim=1))
        fake_images_b = self.decoder_B(fake_images_tmp_b, images_a[:, 1, :, :].unsqueeze(1))
        fake_images_a = self.decoder_A(fake_images_tmp_a, images_b[:, 1, :, :].unsqueeze(1))
        # fake_images_b = self.decoder_B(fake_images_tmp_b, images_a)
        # fake_images_a = self.decoder_A(fake_images_tmp_a, images_b)

        # Cross cycle

        if self.im_channels == 1:
            latent_fake_atmp = self.encoder_C_AB(torch.cat((fake_images_a, fake_images_a, fake_images_a), dim=1))
            latent_fake_btmp = self.encoder_C_AB(torch.cat((fake_images_b, fake_images_b, fake_images_b), dim=1))
            latent_fa_diff = self.encoder_D_A(torch.cat((fake_images_a, fake_images_a, fake_images_a), dim=1))
            latent_fb_diff = self.encoder_D_B(torch.cat((fake_images_b, fake_images_b, fake_images_b), dim=1))
        else:
            latent_fake_btmp = self.encoder_C_AB(fake_images_b)
            latent_fake_atmp = self.encoder_C_AB(fake_images_a)
            latent_fa_diff = self.encoder_D_A(fake_images_a)
            latent_fb_diff = self.encoder_D_B(fake_images_b)


        latent_fake_a = self.encoder_C_A(latent_fake_atmp)
        latent_fake_b = self.encoder_C_B(latent_fake_btmp)


        cycle_images_tmp_b = self.decoder_AB(torch.cat((latent_fake_a, latent_fb_diff), dim=1))
        cycle_images_tmp_a = self.decoder_AB(torch.cat((latent_fake_b, latent_fa_diff), dim=1))
        cycle_images_b = self.decoder_B(cycle_images_tmp_b, fake_images_a)
        cycle_images_a = self.decoder_A(cycle_images_tmp_a, fake_images_b)

        # Discriminators
        pred_mask_fake_b = self.segmenter(latent_fake_b)

        prob_fake_a_is_real, prob_fake_a_aux_is_real = self.discriminator_A(fake_images_a)
        prob_fake_b_is_real = self.discriminator_B(fake_images_b)
        
        prob_fea_b_is_real = self.discriminator_F(pred_mask_b)

        return {"cycle_images_a": cycle_images_a,
                "cycle_images_b": cycle_images_b,
                "prob_fake_b_is_real": prob_fake_b_is_real,
                "pred_mask_fake_b": pred_mask_fake_b,
                "prob_fea_b_is_real": prob_fea_b_is_real, 
                "prob_fake_a_aux_is_real": prob_fake_a_aux_is_real
        }
    
    def forward_SA(self, images_a, images_b):
        # print("In forward pass of SA")
        latent_tmpa = self.encoder_C_AB(images_a)
        latent_tmpb = self.encoder_C_AB(images_b)
        latent_a = self.encoder_C_A(latent_tmpa)
        latent_b = self.encoder_C_B(latent_tmpb)

        pred_mask_b = self.segmenter(latent_b)

        latent_a_diff = self.encoder_D_A(images_a)
        latent_b_diff = self.encoder_D_B(images_b)

        fake_images_tmp_b = self.decoder_AB(torch.cat((latent_a, latent_b_diff), dim=1))
        fake_images_tmp_a = self.decoder_AB(torch.cat((latent_b, latent_a_diff), dim=1))
        fake_images_b = self.decoder_B(fake_images_tmp_b, images_a[:, 1, :, :].unsqueeze(1))
        fake_images_a = self.decoder_A(fake_images_tmp_a, images_b[:, 1, :, :].unsqueeze(1))
        # fake_images_b = self.decoder_B(fake_images_tmp_b, images_a)
        # fake_images_a = self.decoder_A(fake_images_tmp_a, images_b)

        # Cross cycle

        if self.im_channels == 1:
            latent_fake_atmp = self.encoder_C_AB(torch.cat((fake_images_a, fake_images_a, fake_images_a), dim=1))
            latent_fake_btmp = self.encoder_C_AB(torch.cat((fake_images_b, fake_images_b, fake_images_b), dim=1))
            latent_fa_diff = self.encoder_D_A(torch.cat((fake_images_a, fake_images_a, fake_images_a), dim=1))
            latent_fb_diff = self.encoder_D_B(torch.cat((fake_images_b, fake_images_b, fake_images_b), dim=1))
        else:
            latent_fake_btmp = self.encoder_C_AB(fake_images_b)
            latent_fake_atmp = self.encoder_C_AB(fake_images_a)
            latent_fa_diff = self.encoder_D_A(fake_images_a)
            latent_fb_diff = self.encoder_D_B(fake_images_b)


        latent_fake_a = self.encoder_C_A(latent_fake_atmp)
        latent_fake_b = self.encoder_C_B(latent_fake_btmp)

        cycle_images_tmp_b = self.decoder_AB(torch.cat((latent_fake_a, latent_fb_diff), dim=1))
        cycle_images_tmp_a = self.decoder_AB(torch.cat((latent_fake_b, latent_fa_diff), dim=1))
        cycle_images_b = self.decoder_B(cycle_images_tmp_b, fake_images_a)
        cycle_images_a = self.decoder_A(cycle_images_tmp_a, fake_images_b)

        pred_mask_real_a = self.segmenter(latent_a)
        pred_mask_fake_b = self.segmenter(latent_fake_b)

        prob_fea_fake_b_is_real = self.discriminator_F(pred_mask_fake_b.detach())

        # Discriminators

        prob_fake_a_is_real, prob_fake_a_aux_is_real = self.discriminator_A(fake_images_a)
        prob_fea_b_is_real = self.discriminator_F(pred_mask_b.detach())

        return {"cycle_images_a": cycle_images_a,
                "cycle_images_b": cycle_images_b,
                "prob_fake_a_is_real": prob_fake_a_is_real,
                "pred_mask_real_a" : pred_mask_real_a, 
                "prob_fea_fake_b_is_real": prob_fea_fake_b_is_real, 
                "prob_fea_b_is_real": prob_fea_b_is_real
        }

    def update_SB(self, images_a, images_b, labels_a, loss_f_weight_value):
        res = self.forward_SB(images_a, images_b)

        loss_sB = self.seg_loss_B(res["pred_mask_fake_b"], labels_a.detach(), self.segmenter, res["prob_fake_b_is_real"], images_a[:, 1, :, :].unsqueeze(1).detach(), images_b[:, 1, :, :].unsqueeze(1).detach(), res["cycle_images_a"], res["cycle_images_b"], loss_f_weight_value, res["prob_fea_b_is_real"], res["prob_fake_a_aux_is_real"])
        # loss_sB = self.seg_loss_B(res["pred_mask_fake_b"], labels_a.detach(), self.segmenter, res["prob_fake_b_is_real"], images_a.detach(), images_b.detach(), res["cycle_images_a"], res["cycle_images_b"], loss_f_weight_value, res["prob_fea_b_is_real"], res["prob_fake_a_aux_is_real"])

        # update SB
        # print("Update SB")
        self.s_B_trainer.zero_grad()
        loss_sB.backward()
        self.s_B_loss_item += loss_sB.item()
        self.s_B_trainer.step()

    def update_SA(self, images_a, images_b, labels_a):
        res = self.forward_SA(images_a, images_b)

        loss_sA = self.seg_loss_A(res["pred_mask_real_a"], labels_a.detach(), self.segmenter, res["prob_fake_a_is_real"], images_a[:, 1, :, :].unsqueeze(1).detach(), images_b[:, 1, :, :].unsqueeze(1).detach(), res["cycle_images_a"], res["cycle_images_b"])
        # loss_sA = self.seg_loss_A(res["pred_mask_real_a"], labels_a.detach(), self.segmenter, res["prob_fake_a_is_real"], images_a.detach(), images_b.detach(), res["cycle_images_a"], res["cycle_images_b"])
        
        loss_dF = self.d_F_loss(res["prob_fea_fake_b_is_real"], res["prob_fea_b_is_real"])
        
        # print("Update SA")
        self.s_A_trainer.zero_grad()
        loss_sA.backward()
        self.s_A_loss_item += loss_sA.item()
        self.s_A_trainer.step()

        # update DF
        # print("Update DF")
        self.d_F_trainer.zero_grad()
        loss_dF.backward()
        self.d_F_loss_item += loss_dF.item()
        self.d_F_trainer.step()
    
    def forward_Diff(self, images_a, images_b):
        # print("In forward pass of diff")
        latent_tmpa = self.encoder_C_AB(images_a)
        latent_tmpb = self.encoder_C_AB(images_b)
        latent_a = self.encoder_C_A(latent_tmpa)
        latent_b = self.encoder_C_B(latent_tmpb)


        latent_a_diff = self.encoder_D_A(images_a)
        latent_b_diff = self.encoder_D_B(images_b)
        A_separate_B = self.encoder_D_B(images_a)
        B_separate_A = self.encoder_D_A(images_b)

        fake_images_tmp_b = self.decoder_AB(torch.cat((latent_a, latent_b_diff), dim=1))
        fake_images_tmp_a = self.decoder_AB(torch.cat((latent_b, latent_a_diff), dim=1))
        fake_images_b = self.decoder_B(fake_images_tmp_b, images_a[:, 1, :, :].unsqueeze(1))
        fake_images_a = self.decoder_A(fake_images_tmp_a, images_b[:, 1, :, :].unsqueeze(1))
        # fake_images_b = self.decoder_B(fake_images_tmp_b, images_a)
        # fake_images_a = self.decoder_A(fake_images_tmp_a, images_b)

        # Cross cycle
        # print("cross cycle forward pass")

        FA_separate_B = self.encoder_D_B(torch.cat((fake_images_a, fake_images_a, fake_images_a), dim=1))
        FB_separate_A = self.encoder_D_A(torch.cat((fake_images_b, fake_images_b, fake_images_b), dim=1))
        # FA_separate_B = self.encoder_D_B(fake_images_a)
        # FB_separate_A = self.encoder_D_A(fake_images_b)
        

        return {
                "fea_A_separate_B": A_separate_B,
                "fea_B_separate_A": B_separate_A,
                "fea_FA_separate_B": FA_separate_B,
                "fea_FB_separate_A": FB_separate_A
        }

    def update_Diff(self, images_a, images_b):
        res = self.forward_Diff(images_a, images_b)
        loss_diff = self.dif_loss(res["fea_A_separate_B"], res["fea_B_separate_A"], res["fea_FA_separate_B"], res["fea_FB_separate_A"])
        
        # print("Update diff")
        self.dif_trainer.zero_grad()
        loss_diff.backward()
        self.dif_loss_item += loss_diff.item()
        self.dif_trainer.step()

    def update(self, images_a, images_b, labels_a, loss_f_weight_value):
        self.update_GA_DB(images_a, images_b)
        self.update_SB(images_a, images_b, labels_a, loss_f_weight_value)
        self.update_SA(images_a, images_b, labels_a)
        self.update_GB_DA(images_a, images_b)
        self.update_Diff(images_a, images_b)

    # Now only for testing
    def resume(self, model_dir):
        checkpoint = torch.load(model_dir)
        self.encoder_C_AB.load_state_dict(checkpoint["encoder_C_AB"]) 
        self.encoder_C_B.load_state_dict(checkpoint["encoder_C_B"])
        self.segmenter.load_state_dict(checkpoint["segmenter"])
        
    
    def save(self, filename, ep):
        state = {
                'ep' : ep,
                'discriminator_A': self.discriminator_A.state_dict(),
                'discriminator_B': self.discriminator_B.state_dict(),
                'discriminator_F': self.discriminator_F.state_dict(),
                'encoder_C_AB': self.encoder_C_AB.state_dict(),
                'encoder_C_A': self.encoder_C_A.state_dict(),
                'encoder_C_B': self.encoder_C_B.state_dict(),
                'encoder_D_A': self.encoder_D_A.state_dict(),
                'encoder_D_B': self.encoder_D_B.state_dict(),
                'decoder_AB': self.decoder_AB.state_dict(),
                'decoder_A': self.decoder_A.state_dict(),
                'decoder_B': self.decoder_B.state_dict(),
                'segmenter': self.segmenter.state_dict(),
                'd_A_trainer': self.d_A_trainer.state_dict(),
                'd_B_trainer': self.d_B_trainer.state_dict(),
                'dif_trainer': self.dif_trainer.state_dict(),
                'g_A_trainer': self.g_A_trainer.state_dict(),
                'g_B_trainer': self.g_B_trainer.state_dict(),
                's_B_trainer': self.s_B_trainer.state_dict(),
                's_A_trainer': self.s_A_trainer.state_dict(),
                'd_F_trainer': self.d_F_trainer.state_dict()}
        
        torch.save(state, filename)


