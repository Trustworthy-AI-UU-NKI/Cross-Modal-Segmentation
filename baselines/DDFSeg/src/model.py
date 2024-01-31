import networks
import torch
import torch.nn as nn
import numpy as np
from losses import *
from networks import *


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
        self.discriminator_A = Discriminator() # d_A
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
        latent_a = self.encoder_C_A(latent_tmpa)
        
        latent_tmpb = self.encoder_C_AB(images_b)
        latent_b = self.encoder_C_B(latent_tmpb)

        pred_mask_b = self.forward_segmentation(images_b)

        latent_a_diff = self.encoder_D_A(images_a)
        latent_b_diff = self.encoder_D_B(images_b)

        fake_images_tmp_b = self.decoder_AB(torch.cat((latent_a, latent_b_diff), dim=1))
        fake_images_b = self.decoder_B(fake_images_tmp_b, images_a)

        fake_images_tmp_a = self.decoder_AB(torch.cat((latent_b, latent_a_diff), dim=1))
        fake_images_a = self.decoder_A(fake_images_tmp_a, images_b)

        return fake_images_a, fake_images_b




    def backward_G():
        

    def update_G(self, images_a, images_b, labels_a, loss_f_weight_value):
        # forward pass through the generator network
        fake_images_a, fake_images_a, cycle_images_a, cycle_images_b = self.forward_generator(images_a, images_b)

        loss_gA = self.g_loss_A_item(prob_fake_x_is_real, images_a, images_b, cycle_input_a=cycle_images_a, cycle_input_b=cycle_images_b)
        loss_gB = self.g_loss_B_item(prob_fake_x_is_real, images_a, images_b, cycle_input_a=cycle_images_a, cycle_input_b=cycle_images_b)
        
        # update GA 
        self.g_A_trainer.zero_grad()
        loss_gA.backward(retain_graph=True)
        self.g_loss_A_item = loss_gA.item()
        self.g_A_trainer.step()

        # and update GB
        self.g_B_trainer.zero_grad()
        loss_gB.backward()
        self.g_loss_B_item = loss_gB.item()
        self.g_B_trainer.step()


        return fake_B_temp, fake_A_temp 


        # fake_B_temp B is the result of exectuting self.fake_images_b ????
        #sess.run exectutes [] with the settings of the feed_dict
        # _, fake_B_temp, summary_str = sess.run(
        #     [self.g_A_trainer,
        #         self.fake_images_b,
        #         self.g_A_loss_summ],
        #     feed_dict={
        #         self.input_a:
        #             inputs['images_i'],
        #         self.input_b:
        #             inputs['images_j'],
        #         self.gt_a:
        #             inputs['gts_i'],
        #         self.learning_rate: curr_lr,
        #         self.keep_rate:keep_rate_value,
        #         self.loss_f_weight: loss_f_weight_value,
        #     }
        # )
        # writer.add_summary(summary_str, epoch * max_inter + i)


    def update_DB(self, images_a, images_b, fake_B, loss_f_weight_value):
        # Optimizing the D_B network
        # _, summary_str = sess.run(
        #     [self.d_B_trainer, self.d_B_loss_summ],
        #     feed_dict={
        #         self.input_a:
        #             inputs['images_i'],
        #         self.input_b:
        #             inputs['images_j'],
        #         self.learning_rate: curr_lr,
        #         self.fake_pool_B: fake_B_temp1,
        #         self.keep_rate: keep_rate_value,
        #         self.is_training: is_training_value,
        #         self.loss_f_weight: loss_f_weight_value,
        #     }
        # )
        # writer.add_summary(summary_str, epoch * max_inter + i)


    def update_SB(self, images_a, images_b, labels_a, loss_f_weight_value):
        # Optimizing the S_B network
        # _, summary_str = sess.run(
        #     [self.s_B_trainer, self.s_B_loss_merge_summ],
        #     feed_dict={
        #         self.input_a:
        #             inputs['images_i'],
        #         self.input_b:
        #             inputs['images_j'],
        #         self.gt_a:
        #             inputs['gts_i'],
        #         self.learning_rate_seg: curr_lr_seg,
        #         self.keep_rate: keep_rate_value,
        #         self.is_training: is_training_value,
        #         self.loss_f_weight: loss_f_weight_value,
        #     }

        # )
        # writer.add_summary(summary_str, epoch * max_inter + i)


    def update_SA(self, images_a, images_b, labels_a, loss_f_weight_value):
        # Optimizing the S_A network
        # _, summary_str = sess.run(
        #     [self.s_A_trainer, self.s_B_loss_merge_summ],
        #     feed_dict={
        #         self.input_a:
        #             inputs['images_i'],
        #         self.input_b:
        #             inputs['images_j'],
        #         self.gt_a:
        #             inputs['gts_i'],
        #         self.learning_rate_seg: curr_lr_seg,
        #         self.keep_rate: keep_rate_value,
        #         self.is_training: is_training_value,
        #         self.loss_f_weight: loss_f_weight_value,
        #     }

        # )
        # writer.add_summary(summary_str, epoch * max_inter + i)


    def update_GB(self, images_a, images_b, labels_a, loss_f_weight_value):
        # Optimizing the G_B network
        # _, fake_A_temp, summary_str = sess.run(
        #     [self.g_B_trainer,
        #         self.fake_images_a,
        #         self.g_B_loss_summ],
        #     feed_dict={
        #         self.input_a:
        #             inputs['images_i'],
        #         self.input_b:
        #             inputs['images_j'],
        #         self.learning_rate: curr_lr,
        #         self.gt_a: inputs['gts_i'],
        #         self.keep_rate: keep_rate_value,
        #         self.is_training: is_training_value,
        #         self.loss_f_weight: loss_f_weight_value,
        #     }
        # )
        # writer.add_summary(summary_str, epoch * max_inter + i)

    def update_DA(self, images_a, images_b, fake_A, loss_f_weight_value):
        # Optimizing the D_A network
        # _, summary_str = sess.run(
        #     [self.d_A_trainer, self.d_A_loss_summ],
        #     feed_dict={
        #         self.input_a:
        #             inputs['images_i'],
        #         self.input_b:
        #             inputs['images_j'],
        #         self.learning_rate: curr_lr,
        #         self.fake_pool_A: fake_A_temp1,
        #         self.keep_rate: keep_rate_value,
        #         self.is_training: is_training_value,
        #         self.loss_f_weight: loss_f_weight_value,
        #     }
        # )
        # writer.add_summary(summary_str, epoch * max_inter + i)
    
    def update_DF(self, images_a, images_b, loss_f_weight_value):
        # Optimizing the D_F network
        # _, summary_str = sess.run(
        #     [self.d_F_trainer, self.d_F_loss_summ],
        #     feed_dict={
        #         self.input_a:
        #             inputs['images_i'],
        #         self.input_b:
        #             inputs['images_j'],
        #         self.learning_rate: curr_lr,
        #         self.keep_rate: keep_rate_value,
        #         self.is_training: is_training_value,
        #         self.loss_f_weight: loss_f_weight_value,
        #     }
        # )
        # writer.add_summary(summary_str, epoch * max_inter + i)

    def update_Dif():
        # Optimizing the Dif network
        # _, summary_str = sess.run(
        #     [self.dif_trainer, self.dif_loss_summ],
        #     feed_dict={
        #         self.input_a:
        #                 inputs['images_i'],
        #         self.input_b:
        #                 inputs['images_j'],
        #         self.learning_rate: curr_lr,
        #         self.keep_rate: keep_rate_value,
        #         self.is_training: is_training_value,
        #         self.loss_f_weight: loss_f_weight_value,
        #     }
        #     )
        # writer.add_summary(summary_str, epoch * max_inter + i)

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


