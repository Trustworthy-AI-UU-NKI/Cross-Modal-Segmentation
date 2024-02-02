import networks
import torch
import torch.nn as nn
import numpy as np
from losses import *
from networks import *
import random


class DDFSeg(nn.Module):
    def __init__(self, args, pool_size = 50):
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
        self.skip = args.skip # (True)
        self.keep_rate = args.keep_rate # 0.75 --> BEHALVE 559??

        
        self.fake_images_A = np.zeros(
            (self._pool_size, self.bs, 1, self.img_res, self.img_res)
        )
        self.fake_images_B = np.zeros(
            (self._pool_size, self.bs, 1, self.img_res, self.img_res)
        )

        # networks
        self.discriminator_A = Discriminator(aux=True) # d_A
        self.discriminator_B = Discriminator() # d_B
        self.discriminator_F = Discriminator() # d_F
        self.encoder_C_AB = EncoderCShared(keep_rate=self.keep_rate) # e_c
        self.encoder_C_A = EncoderC(keep_rate=self.keep_rate) #e_cs
        self.encoder_C_B = EncoderC(keep_rate=self.keep_rate) # e_ct
        self.encoder_D_A = EncoderS(keep_rate=self.keep_rate) # e_dA
        self.encoder_D_B = EncoderS(keep_rate=self.keep_rate) # e_dB
        self.decoder_AB = DecoderShared() # de_c
        self.decoder_A = Decoder(skip=self.skip) # de_A
        self.decoder_B = Decoder(skip=self.skip) # de_B
        self.segmenter = SegmentationNetwork(keep_rate=self.keep_rate) # s_A

        # optimizers + according losses
        # Source (A) Discriminator update
        self.d_A_trainer = torch.optim.Adam(self.discriminator_A.parameters(), lr=self.learning_rate, beta1=0.5)
        self.d_A_loss = DiscriminatorLossDouble()

        # Target (B) Discriminator update
        self.d_B_trainer = torch.optim.Adam(self.discriminator_B.parameters(), lr=self.learning_rate, beta1=0.5)
        self.d_B_loss = DiscriminatorLoss()

        # Domain specific encoders update --> zero loss
        self.dif_trainer = torch.optim.Adam(list(self.encoder_D_A.parameters()) + list(self.encoder_D_B.parameters()), lr=self.learning_rate, beta1=0.5)
        self.dif_loss = ZeroLoss(self.learning_rate_5)

        # Generator update --> de_A_vars+de_c_vars+e_c_vars+e_cs_vars+e_dB_vars
        self.g_A_trainer = torch.optim.Adam(list(self.decoder_A.parameters()) + list(self.decoder_AB.parameters()) + list(self.encoder_C_AB.parameters()) 
                                            + list(self.encoder_C_A.parameters() + list(self.encoder_D_B.parameters())), lr=self.learning_rate, beta1=0.5)
        self.g_loss_A = GeneratorLoss(self.lr_a, self.lr_b)

        # Generator update --> de_B_vars+de_c_vars+e_c_vars+e_ct_vars+e_dA_vars
        self.g_B_trainer = torch.optim.Adam(list(self.decoder_B.parameters()) + list(self.decoder_AB.parameters()) + list(self.encoder_C_AB.parameters())
                                            + list(self.encoder_C_B.parameters()) + list(self.encoder_D_A.parameters()), lr=self.learning_rate, beta1=0.5)
        self.g_loss_B = GeneratorLoss(self.lr_a, self.lr_b)

        # Updating segmentation network via target images
        self.s_B_trainer = torch.optim.Adam(list(self.encoder_C_AB.parameters()) + list(self.encoder_C_B) + list(self.segmenter.parameters()), lr=self.learning_rate_seg)
        self.seg_loss_B = SegmentationLossTarget(self.lr_a, self.lr_b)

        # Updating segmentation network via source images
        self.s_A_trainer = torch.optim.Adam(list(self.encoder_C_AB.parameters()) + list(self.encoder_C_B) + list(self.segmenter.parameters()), lr=self.learning_rate_seg)
        self.seg_loss_A = SegmentationLoss(self.lr_a, self.lr_b)

        # Feature Discriminator (Dis_seg)
        self.d_F_trainer = torch.optim.Adam(self.discriminator_F.parameters(), lr=self.learning_rate, beta1=0.5)
        self.d_F_loss = DiscriminatorLoss()


    
    # Set scheduler??
    
    def update_num_fake_inputs(self):
        self.num_fake_inputs += 1


    def set_lr_scheduler(self, last_ep=0):
        # TO DO
        # self.d_A_trainer_sch = get_
        return
    

    def fake_image_pool_A(self, fake):
        if self.num_fake_inputs < self._pool_size:
            self.fake_images_A[self.num_fake_inputs] = fake
            return fake
        else:
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0, self._pool_size - 1)
                temp = self.fake_images_A[random_id]
                self.fake_images_A[random_id] = fake
                return temp
            else:
                return fake
            
    def fake_image_pool_B(self, fake):
        if self.num_fake_inputs < self._pool_size:
            self.fake_images_B[self.num_fake_inputs] = fake
            return fake
        else:
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0, self._pool_size - 1)
                temp = self.fake_images_B[random_id]
                self.fake_images_B[random_id] = fake
                return temp
            else:
                return fake

    def setgpu(self, gpu):
        self.gpu = gpu
        self.encoder_C_AB.cuda(self.gpu)
        self.encoder_C_A.cuda(self.gpu)
        self.encoder_C_B.cuda(self.gpu)
        self.encoder_D_A.cuda(self.gpu)
        self.encoder_D_B.cuda(self.gpu)
        self.decoder_AB.cuda(self.gpu)
        self.decoder_A.cuda(self.gpu)
        self.decoder_B.cuda(self.gpu)
        self.segmenter.cuda(self.gpu)
        self.discriminator_A.cuda(self.gpu)
        self.discriminator_B.cuda(self.gpu)


        # etc

    # Forward pass through the network --> only encoders and decoders 
    def forward(self, images_a, images_b):
        latent_tmpa = self.encoder_C_AB(images_a)
        latent_tmpb = self.encoder_C_AB(images_b)
        latent_a = self.encoder_C_A(latent_tmpa)
        latent_b = self.encoder_C_B(latent_tmpb)

        pred_mask_b = self.segmenter(latent_b)

        latent_a_diff = self.encoder_D_A(images_a)
        latent_b_diff = self.encoder_D_B(images_b)
        A_separate_B = self.encoder_D_B(images_a)
        B_separate_A = self.encoder_D_A(images_b)

        fake_images_tmp_b = self.decoder_AB(torch.cat((latent_a, latent_b_diff), dim=1))
        fake_images_tmp_a = self.decoder_AB(torch.cat((latent_b, latent_a_diff), dim=1))
        fake_images_b = self.decoder_B(fake_images_tmp_b, images_a)
        fake_images_a = self.decoder_A(fake_images_tmp_a, images_b)

        # Cross cycle

        latent_fake_atmp = self.encoder_C_AB(fake_images_a)
        latent_fake_btmp = self.encoder_C_AB(fake_images_b)
        latent_fake_a = self.encoder_C_A(latent_fake_atmp)
        latent_fake_b = self.encoder_C_B(latent_fake_btmp)

        latent_fa_diff = self.encoder_D_A(fake_images_a)
        latent_fb_diff = self.encoder_D_B(fake_images_b)
        FA_separate_B = self.encoder_D_B(fake_images_a)
        FB_separate_A = self.encoder_D_A(fake_images_b)

        cycle_images_tmp_b = self.decoder_AB(torch.cat((latent_fake_a, latent_fb_diff), dim=1))
        cycle_images_tmp_a = self.decoder_AB(torch.cat((latent_fake_b, latent_fa_diff), dim=1))
        cycle_images_b = self.decoder_B(cycle_images_tmp_b, fake_images_a)
        cycle_images_a = self.decoder_A(cycle_images_tmp_a, fake_images_b)


        pred_mask_fake_b = self.segmenter(latent_fake_b)
        pred_mask_real_a = self.segmenter(latent_a)


        # Discriminators

        prob_real_a_is_real, prob_real_a_aux = self.discriminator_A(images_a)
        prob_real_b_is_real = self.discriminator_B(images_b)

        prob_cycle_a_is_real, prob_cycle_a_aux_is_real = self.discriminator_A(cycle_images_a.detach())

        prob_fake_a_is_real, prob_fake_a_aux_is_real = self.discriminator_A(fake_images_a.detach())
        prob_fake_b_is_real = self.discriminator_B(fake_images_b.detach())
        
        prob_fea_fake_b_is_real = self.discriminator_F(pred_mask_fake_b.detach())
        prob_fea_b_is_real = self.discriminator_F(pred_mask_b.detach())


        # Update fake pool images and forward pass through last Discriminator
        fake_pool_a = self.fake_image_pool_A(fake_images_a.detach())
        fake_pool_b = self.fake_image_pool_B(fake_images_b.detach())
        self.num_fake_inputs += 1

        prob_fake_pool_a_is_real, prob_fake_pool_a_aux_is_real = self.discriminator_A(fake_pool_a)
        prob_fake_pool_b_is_real = self.discriminator_B(fake_pool_b)


        return {"fake_images_a": fake_images_a,
                "fake_images_b": fake_images_b,
                "cycle_images_a": cycle_images_a,
                "cycle_images_b": cycle_images_b,
                "prob_fake_a_is_real": prob_fake_a_is_real,
                "prob_fake_b_is_real": prob_fake_b_is_real,
                
                }
    

    def update_G(self, images_a, images_b, labels_a, loss_f_weight_value):
        # forward pass through the generator network
        # CHECK where retain_graph = True is nec?
        #   I think in loss_gA.backward(), loss_sB.backward() and loss_sA.backward() might be an issue?
        res = self.forward_generator(images_a, images_b)

        loss_gA = self.g_loss_A_item(res["prob_fake_a_is_real"], images_a, images_b, res["cycle_images_a"], res["cycle_images_b"])
        loss_gB = self.g_loss_B_item(res["prob_fake_b_is_real"], images_a, images_b, res["cycle_images_a"], res["cycle_images_b"])
        loss_dB = self.d_B_loss(res["prob_real_b_is_real"], res["self.prob_fake_pool_b_is_real"])
    
        loss_sB = self.seg_loss_B(res["pred_mask_fake_b"], labels_a, self.segmenter.parameters(), res["prob_fake_b_is_real"], images_a, images_b, 
                                  res["cycle_images_a"], res["cycle_images_b"], loss_f_weight_value, res["prob_fea_b_is_real"], res["prob_fake_a_aux_is_real"])

        loss_sA = self.seg_loss_A(res["pred_mask_real_a"], labels_a, self.segmenter.parameters(), res["prob_fake_a_is_real"], images_a, images_b,
                                  res["cycle_images_a"], res["cycle_images_b"])
        
        loss_dA = self.d_A_loss(res["prob_real_a_is_real"], res["prob_fake_pool_a_is_real"], res["prob_fake_a_is_real"], res["prob_cycle_a_aux_is_real"], res["prob_fake_pool_a_aux_is_real"])
        loss_dF = self.d_F_loss(res["prob_fea_fake_b_is_real"], res["prob_fea_b_is_real"])
        loss_diff = self.dif_loss(res["fea_A_separate_B"], res["fea_B_separate_A"], res["fea_FA_separate_B"], res["fea_FB_separate_A"])

        # update GA 
        self.g_A_trainer.zero_grad()
        loss_gA.backward(retain_graph=True)
        self.g_loss_A_item = loss_gA.item()
        self.g_A_trainer.step()

        # update DB
        self.d_B_trainer.zero_grad()
        loss_dB.backward()
        self.d_B_loss_item = loss_dB.item()
        self.d_B_trainer.step()
        
        # update SB
        self.s_B_trainer.zero_grad()
        loss_sB.backward(retain_graph=True)
        self.s_B_loss_item = loss_sB.item()
        self.s_B_trainer.step()

        # update SA
        self.s_A_trainer.zero_grad()
        loss_sA.backward(retain_graph=True)
        self.s_A_loss_item = loss_sA.item()
        self.s_A_trainer.step()

        # update GB
        self.g_B_trainer.zero_grad()
        loss_gB.backward()
        self.g_loss_B_item = loss_gB.item()
        self.g_B_trainer.step()

        # update DA
        self.d_A_trainer.zero_grad()
        loss_dA.backward()
        self.d_A_loss_item = loss_dA.item()
        self.d_A_trainer.step()

        # update DF
        self.d_F_trainer.zero_grad()
        loss_dF.backward()
        self.d_F_loss_item = loss_dF.item()
        self.d_F_trainer.step()

        # update diff
        self.dif_trainer.zero_grad()
        loss_diff.backward()
        self.dif_loss_item = loss_diff.item()
        self.dif_trainer.step()

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir)
        # etc
        
    
    def save(self, filename, ep, total_it):
        state = {
                'disA': self.disA.state_dict(),
                'disA2': self.disA2.state_dict(),
                'disB': self.disB.state_dict(),
                'disB2': self.disB2.state_dict(),
                'disContent': self.disContent.state_dict(),
                'enc_c': self.enc_c.state_dict(),
                'enc_a': self.enc_a.state_dict(),
                'gen': self.gen.state_dict(),
                'disA_opt': self.disA_opt.state_dict(),
                'disA2_opt': self.disA2_opt.state_dict(),
                'disB_opt': self.disB_opt.state_dict(),
                'disB2_opt': self.disB2_opt.state_dict(),
                'disContent_opt': self.disContent_opt.state_dict(),
                'enc_c_opt': self.enc_c_opt.state_dict(),
                'enc_a_opt': self.enc_a_opt.state_dict(),
                'gen_opt': self.gen_opt.state_dict(),
                'ep': ep,
                'total_it': total_it
                }
        torch.save(state, filename)


