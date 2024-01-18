import networks
import torch
import torch.nn as nn

class DDFSeg(nn.Module):
    def __init__(self, args):
        super(DDFSeg, self).__init__()
        self.base_lr = args.lr
        self.num_dake_inputs = 0
        # etc....

    def initialize_model(self):
    # initialize with gaussian weights (????)
        self.disA.apply(networks.gaussian_weights_init)
    
    # Set scheduler??
    
    def update_num_fake_inputs(self):
        self.num_fake_inputs += 1



    def setgpu(self, gpu):
        self.gpu = gpu
        self.parts.cuda(self.gpu)
        # etc

    def forward(self, x):
        # TO DO
        return x

    def update_parts(self):
        # TO DO
        self.parts_optimizer.zero_grad()
        loss = self.backward()
        self.loss_item = loss.item()
        self.parts_optimizer.step()



    def update_GA(self, images_a, images_b, labels_a, loss_f_weight_value):
        # Optimizing the G_A network
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
        #         self.is_training:is_training_value,
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


