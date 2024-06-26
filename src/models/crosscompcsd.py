import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models.encoder import *
from models.decoder import *
from models.segmentor import *
from models.discriminator import *
from models.weight_init import *
from composition.model import *

from composition.losses import ClusterLoss
from monai.losses import DiceLoss
from losses import *

from utils import *

# Class of our proposed method           
class CrossCSD(nn.Module):
    def __init__(self, args, device, image_channels, num_classes, vMF_kappa, fold_nr):
        super(CrossCSD, self).__init__()

        self.device = device
        self.num_classes = num_classes
        self.weight_init = args.weight_init
        self.opt_clus = -1
        self.vc_num  = args.vc_num
        self.true_clu_loss = True
        self.fold = fold_nr

        # Get modules
        self.activation_layer = ActivationLayer(vMF_kappa)
        self.encoder_source = Encoder(image_channels)
        self.decoder_source = Decoder(image_channels)
        self.encoder_target = Encoder(image_channels)
        self.decoder_target = Decoder(image_channels)
        self.segmentor = Segmentor(num_classes, args.vc_num)
        self.discriminator_source = DiscriminatorD()
        self.discriminator_target = DiscriminatorD()
        
        # Initialize weights
        self.initialize_model()

        # Initialize all optimizers
        self.optimizer_source = optim.Adam(list(self.encoder_source.parameters()) + list(self.decoder_source.parameters()) + list(self.encoder_target.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))
        self.scheduler_source = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_source, 'max', patience=args.k2)

        self.optimizer_target = optim.Adam(list(self.encoder_source.parameters()) + list(self.encoder_target.parameters()) + list(self.decoder_target.parameters()) + list(self.conv1o1.parameters())+ list(self.segmentor.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))
        self.scheduler_target = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_target, 'max', patience=args.k2)

        self.optimizer_disc_source = optim.Adam(self.discriminator_source.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
        self.optimizer_disc_target = optim.Adam(self.discriminator_target.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

        # Losses
        self.l1_distance = nn.L1Loss().to(device)
        self.cluster_loss = ClusterLoss().to(device)
        self.dice_loss = DiceLoss(include_background=False, softmax=True).to(device)
        self.gen_loss = GeneratorLoss()
        self.disc_loss_s = DiscriminatorLoss()
        self.disc_loss_t = DiscriminatorLoss()

    
    # Get vMFKernels and initialize the network
    def initialize_model(self):
        initialize_weights(self, self.weight_init)
        weights = torch.zeros([self.vc_num, 64, 1, 1]).type(torch.FloatTensor)
        nn.init.xavier_normal_(weights)
        self.conv1o1 = Conv1o1Layer(weights, self.device)
    
    # Forward pass through the compositional layer
    def comp_layer_forward(self, features):
        kernels = self.conv1o1.weight
        vc_activations = self.conv1o1(features[self.layer])
        vmf_activations = self.activation_layer(vc_activations) 
        norm_vmf_activations = torch.zeros_like(vmf_activations)
        norm_vmf_activations = norm_vmf_activations.to(self.device)
        for i in range(vmf_activations.size(0)):
            norm_vmf_activations[i, :, :, :] = F.normalize(vmf_activations[i, :, :, :], p=1, dim=0)
        self.vmf_activations = norm_vmf_activations
        self.vc_activations = vc_activations
        return norm_vmf_activations, kernels

    # Forward pass for evaluation
    def forward_eval(self, x_s, x_s_label):
        # Cross reconstruction
        features_s = self.encoder_source(x_s)
        fake_image_t = self.decoder_target(features_s[self.layer]) # style from tareget domain, content of source domains
        fake_features_t = self.encoder_target(fake_image_t) 
        cross_rec_s = self.decoder_source(fake_features_t[self.layer])

        # Calculate losses of evaluation data
        cross_rec_loss_s = self.l1_distance(cross_rec_s, x_s)

        # Get compositional features of true and fake target image
        com_fake_features_t, _ = self.comp_layer_forward(fake_features_t)

        # Get predicted segmentation mask of fake target images for evaluation
        pre_fake_seg_t = self.segmentor(com_fake_features_t)
        compact_pred_fake_t = torch.argmax(pre_fake_seg_t, dim=1).unsqueeze(1)

        # Calculate metrics
        dsc_target_fake = dice(x_s_label, pre_fake_seg_t, self.num_classes)
        dsc_target_fake_total = torch.mean(dsc_target_fake[1:])
        assd_target_fake = assd(x_s_label, pre_fake_seg_t, self.num_classes, pix_dim=x_s.meta["pixdim"][1])

        # For tensorboard logging
        metrics_dict = {'Source/total': cross_rec_loss_s.item(),
                        'Target/DSC_fake': dsc_target_fake_total, 'Target/DSC_fake_0': dsc_target_fake[0], 
                        'Target/DSC_fake_1': dsc_target_fake[1], 'Target/assd_fake': assd_target_fake.item()}

        images_dict = {'Source/image': x_s,
                       'Source/Cross_reconstructed': cross_rec_s, 'Target/fake': fake_image_t, 
                         'Target/Fake_predicted_seg': compact_pred_fake_t, 'Source/label': x_s_label}


        visuals_dict = {'Fake_Target': com_fake_features_t}
        return metrics_dict, images_dict, visuals_dict
    
    # Forward pass for testing
    def forward_test(self, x_t, x_t_label):
        features_t = self.encoder_target(x_t)
        com_features_t, _ = self.comp_layer_forward(features_t)
        pre_seg_t = self.segmentor(com_features_t)
        compact_pred_t = torch.argmax(pre_seg_t, dim=1).unsqueeze(1)

        # Calculate metrics
        dsc_target = dice(x_t_label, pre_seg_t, self.num_classes)
        assd_target = assd(x_t_label, pre_seg_t, self.num_classes, pix_dim=x_t.meta["pixdim"][1])
        
        metrics_dict = {'Target/DSC_0': dsc_target[0], 
                        'Target/DSC_1': dsc_target[1], 'Target/assd': assd_target.item()}

        # For tensorboard logging
        images_dict = {'Target/image': x_t, 'Target/label': x_t_label,
                       'Target/predicted_seg': compact_pred_t}

        visuals_dict = {'Target': com_features_t}
        return metrics_dict, images_dict, visuals_dict

    # Forward pass for training
    def forward(self, x_s, x_t):
        features_s = self.encoder_source(x_s)
        features_t = self.encoder_target(x_t)
        rec_s = self.decoder_source(features_s[self.layer])
        rec_t = self.decoder_target(features_t[self.layer])

        fake_image_t = self.decoder_target(features_s[self.layer]) # style from target domain, content of source domain image
        fake_image_s = self.decoder_source(features_t[self.layer]) #style from source domain, content of target domain image
        
        fake_features_s = self.encoder_source(fake_image_s)
        fake_features_t = self.encoder_target(fake_image_t)
        cross_rec_s = self.decoder_source(fake_features_t[self.layer])
        cross_rec_t = self.decoder_target(fake_features_s[self.layer])

        # Get compositional features and predicted segmentation mask of fake target image
        com_fake_features_t, kernels = self.comp_layer_forward(fake_features_t)
        pre_fake_seg_t = self.segmentor(com_fake_features_t)

        results = {"rec_s": rec_s, "cross_img_s": cross_rec_s, "fake_img_s": fake_image_s, "rec_t": rec_t, "cross_img_t": cross_rec_t, "fake_img_t": fake_image_t, 
                    "feats_s": features_s[self.layer], "feats_t": features_t[self.layer], "fake_feats_t": fake_features_t[self.layer], "pre_fake_seg_t": pre_fake_seg_t,  "kernels": kernels} # "pre_seg_s": pre_seg_s,
        return results

    # Update step in training loop
    def update(self, img_s, label_s, img_t):
        # Forward pass
        results = self.forward(img_s, img_t)

        # Discriminators output
        prob_true_source_is_true = self.discriminator_source(img_s.detach())
        prob_fake_source_is_true = self.discriminator_source(results["fake_img_s"].detach())
        prob_true_target_is_true = self.discriminator_target(img_t.detach())
        prob_fake_target_is_true = self.discriminator_target(results["fake_img_t"].detach())
        
        # Get loss Discriminators
        disc_loss_s = self.disc_loss_s(prob_true_source_is_true, prob_fake_source_is_true) 
        disc_loss_t = self.disc_loss_t(prob_true_target_is_true, prob_fake_target_is_true)

        # Update source discriminator  (line 1 algorithm 1)
        self.optimizer_disc_source.zero_grad()
        disc_loss_s.backward()
        nn.utils.clip_grad_value_(self.discriminator_source.parameters(), 0.1)
        self.optimizer_disc_source.step()

        # Update target discriminator  (line 2 algorithm 1)
        self.optimizer_disc_target.zero_grad()
        disc_loss_t.backward()
        nn.utils.clip_grad_value_(self.discriminator_target.parameters(), 0.1)
        self.optimizer_disc_target.step()

        # Get output source discriminator
        prob_fake_source_is_true_gen = self.discriminator_source(results["fake_img_s"])

        # Get source losses
        gen_loss_s = self.gen_loss(prob_fake_source_is_true_gen)
        cross_reco_loss_s = self.l1_distance(results["cross_img_s"], img_s)
        batch_loss_s = cross_reco_loss_s + gen_loss_s

        # Update source (line 3 algorithm 1)
        self.optimizer_source.zero_grad()
        batch_loss_s.backward()
        nn.utils.clip_grad_value_(self.parameters(), 0.1)
        self.optimizer_source.step()
        
        # Another forward pass for correct gradient computation
        results = self.forward(img_s, img_t)

        # Get output target discriminator
        prob_fake_target_is_true_gen = self.discriminator_target(results["fake_img_t"])

        # Get target losses
        gen_loss_t = self.gen_loss(prob_fake_target_is_true_gen)
        cross_reco_loss_t = self.l1_distance(results["cross_img_t"], img_t.detach())
        clu_loss_t = self.cluster_loss(results["feats_t"].detach(), results["kernels"])
        label_s_oh = F.one_hot(label_s.long().squeeze(1), num_classes=self.num_classes).permute(0, 3, 1, 2)
        dice_loss_t = self.dice_loss(results["pre_fake_seg_t"], label_s_oh)
        batch_loss_t = cross_reco_loss_t + gen_loss_t + dice_loss_t + clu_loss_t
        
        # Update target (line 4 algorithm 1)
        self.optimizer_target.zero_grad()
        batch_loss_t.backward()
        nn.utils.clip_grad_value_(self.parameters(), 0.1)
        self.optimizer_target.step()

        batch_loss_it = batch_loss_s.item() + batch_loss_t.item()

        losses = {'loss/source/batch_loss': batch_loss_s.item(), 'loss/target/batch_loss': batch_loss_t.item(),
                    'loss/source/gen_loss': gen_loss_s.item(), 'loss/source/cross_reco_loss': cross_reco_loss_s.item(),
                    'loss/target/gen_loss': gen_loss_t.item(), 'loss/target/cross_reco_loss': cross_reco_loss_t.item(), 
                    f'loss/target/{self.true_clu_loss}_cluster_loss': clu_loss_t.item(),
                    'loss/target/dice_loss': dice_loss_t.item(), 'loss/total': batch_loss_it,
                    'loss/source/disc_loss': disc_loss_s.item(),  'loss/target/disc_loss': disc_loss_t.item()}    
        return losses
    
    def resume(self, model_file, cpu = False):
        if cpu:
            checkpoint = torch.load(model_file, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(model_file)
        self.load_state_dict(checkpoint)

