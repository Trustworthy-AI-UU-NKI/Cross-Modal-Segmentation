# imports 
import numpy as np
from monai.transforms import (
    Compose,
    AsDiscrete
)
from monai.data import DataLoader
import os
import matplotlib.pyplot as plt
import glob
import pytorch_lightning as pl
import torch

from monai.networks.nets import UNet
from baselines.UNet import UNet_model
from baselines.dataloaders import MMWHS, CHAOS
import sys
import random

sys.path.insert(0, 'baselines/vMFNet/')
from baselines.vMFNet.models.compcsd import CompCSD

sys.path.insert(0, 'src/')
from src.models.crosscompcsd import CrossCSD

sys.path.insert(0, 'baselines/DDFSeg/src')
from baselines.DDFSeg.src.model import DDFSeg

# Arguments for proposed model
class Args:
    def __init__(self):
        self.weight_init = "xavier"
        self.vc_num = 10
        self.k2 = 10
        self.learning_rate = 0.0001

# Arguments for DDFSeg 
class ArgsDDF:
    def __init__(self):
        self.lr = 0.0002
        self.lr67 = 0.001
        self.lr5 = 0.01
        self.lr_A = 10
        self.lr_B = 10
        self.resolution = 256
        self.skip = True
        self.keep_rate = 0.75
        self.bs = 4

# Get output UNet and ResUNet
def get_output_unet(image, model_name, device, unet=True):
    # Load appropriate model
    if unet:
        model = UNet_model(2).to(device)
    else: 
        # This is the ResUNet
        model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(device)
    pretrained_model = glob.glob(os.path.join(model_name, "*.pth"))[0]
    model.load_state_dict(torch.load(pretrained_model))
    model.eval()

    # Get predicted mask
    post_trans = Compose([AsDiscrete(argmax=True)])
    outputs = model.forward(image.to(device)).detach().cpu()
    vis_outputs = post_trans(outputs[0])
    masked_output = np.ma.masked_where(vis_outputs[0, :, :] == 0, vis_outputs[0, :, :])
    return masked_output

# Get output proposed model
def get_output_pr(image, model_name, device):
    # Load appropriate model
    args = Args()
    pretrained_model = glob.glob(os.path.join(model_name, "*.pth"))[0]
    model = CrossCSD(args, device, 1, 2, vMF_kappa=30, fold_nr=0)
    model.to(device)
    model.resume(pretrained_model)
    model.eval()
    com_features_t, compact_pred_t, pre_seg_t = model.test(image)

    compact_pred_t = compact_pred_t[0, 0, :, :].detach().cpu()
    masked_output = np.ma.masked_where(compact_pred_t == 0, compact_pred_t)
    return masked_output

# Get output vMFNet
def get_output_vmfnet(image, model_name, device):
    # Load appropriate model
    pretrained_model = glob.glob(os.path.join(model_name, "*.pth"))[0]
    model = CompCSD(device, 1, 8, 12, num_classes=2, z_length=8, vMF_kappa=30, init='xavier')
    model.initialize(pretrained_model, init="xavier") # Does not matter for testing
    model.to(device)
    model.resume(pretrained_model)
    model.eval()

    # Get predicted mask
    rec, pre_seg, features, kernels, L_visuals = model(image)
    compact_pred = torch.argmax(pre_seg, dim=1).unsqueeze(1).detach().cpu()
    masked_output = np.ma.masked_where(compact_pred[0, 0, :, :] == 0, compact_pred[0, 0, :, :])
    return masked_output

# Get output DDFseg
def get_output_ddfseg(image, model_name, device):
    # Load appropriate model
    pretrained_model = glob.glob(os.path.join(model_name, "*.pth"))[0]
    args = ArgsDDF()
    model = DDFSeg(args, num_classes=2)
    model.setgpu(device)
    model.resume(pretrained_model)
    model.eval()

    # Get predicted mask
    pred_mask = model.forward_test(torch.cat((image, image, image), dim=1))
    compact_pred = torch.argmax(pred_mask, dim=1).unsqueeze(1).detach().cpu()
    masked_output = np.ma.masked_where(compact_pred[0, 0, :, :] == 0, compact_pred[0, 0, :, :])
    return masked_output

# Visualize results and store in pdf for better resolution
def visualize_baselines(tiles, data_1, data_2, data_3, source='MRI', target='CT', label="MYO", cmap_col='spring'):
    fig = plt.figure("visualize", (24, 8))

    for i in range(len(tiles)):
        plt.subplot(3, len(tiles), i + 1)
        plt.title(tiles[i], fontsize=25)
        plt.imshow(data_1[0], cmap="gray")
        if not (i==0):
            plt.imshow(data_1[i], alpha=0.6, cmap=cmap_col) 
        plt.axis("off")
        
        plt.subplot(3, len(tiles), i + 1 + len(tiles))
        plt.imshow(data_2[0], cmap="gray")
        if not (i==0):
            plt.imshow(data_2[i], alpha=0.6, cmap=cmap_col) 
        plt.axis("off")

        plt.subplot(3, len(tiles), i + 1 + 2*len(tiles))
        plt.imshow(data_3[0], cmap="gray")
        if not (i==0):
            plt.imshow(data_3[i], alpha=0.6, cmap=cmap_col) 
        plt.axis("off")
    
    plt.tight_layout() 
    plt.savefig(f'results/vis_{source}->{target}_{label}.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    plt.close(fig)

# Get results for all baselines
def show_results(device, data_dir, paths, test_cases_fold_0, i1, i2, i3, source, target, label):
    
    print(label)
    if label == "MYO":
        labels = [1, 0, 0, 0, 0, 0, 0]
        dataset_test = MMWHS(data_dir, test_cases_fold_0, labels) 
        cmap_col='winter'
    elif label == "LV":
        labels = [0, 0, 1, 0, 0, 0, 0]
        dataset_test = MMWHS(data_dir, test_cases_fold_0, labels)
        cmap_col='spring'
    elif label == "RV":
        labels = [0, 0, 0, 0, 1, 0, 0]
        dataset_test = MMWHS(data_dir, test_cases_fold_0, labels)
        cmap_col='cool'
    elif label == "Liver":
        labels = [1, 0, 0, 0]
        dataset_test = CHAOS(data_dir, test_cases_fold_0, labels)
        cmap_col='autumn' 
    elif label == "Spleen":
        labels = [0, 0, 0, 1]
        dataset_test = CHAOS(data_dir, test_cases_fold_0, labels) 
        cmap_col='Wistia'
    
    test_loader = DataLoader(dataset_test, batch_size=1)
    for i, (im, lab) in enumerate(test_loader):
        if i == i1:
            image1 = im.to(device)
            gt1 = lab
            output_FS_1 = get_output_unet(image1, paths["name_FS"], device)
            output_NA_1 = get_output_unet(image1, paths["name_NA"], device)
            output_DU_1 = get_output_unet(image1, paths["name_DU"], device)
            output_DR_1 = get_output_unet(image1, paths["name_DR"], device, False)
            output_VM_1 = get_output_vmfnet(image1, paths["name_VMFNET"], device)
            output_PR_1 = get_output_pr(image1, paths["name_PR"], device)
            output_DDF_1 = get_output_ddfseg(image1, paths["name_DDF"], device)
        
        if i == i2:
            image2 = im.to(device)
            gt2 = lab
            output_FS_2 = get_output_unet(image2, paths["name_FS"], device)
            output_NA_2 = get_output_unet(image2, paths["name_NA"], device)
            output_DU_2 = get_output_unet(image2, paths["name_DU"], device)
            output_DR_2 = get_output_unet(image2, paths["name_DR"], device, False)
            output_VM_2 = get_output_vmfnet(image2, paths["name_VMFNET"], device)
            output_PR_2 = get_output_pr(image2, paths["name_PR"], device)
            output_DDF_2 = get_output_ddfseg(image2, paths["name_DDF"], device)
        
        if i == i3:
            image3 = im.to(device)
            gt3 = lab
            output_FS_3 = get_output_unet(image3, paths["name_FS"], device)
            output_NA_3 = get_output_unet(image3, paths["name_NA"], device)
            output_DU_3 = get_output_unet(image3, paths["name_DU"], device)
            output_DR_3 = get_output_unet(image3, paths["name_DR"], device, False)
            output_VM_3 = get_output_vmfnet(image3, paths["name_VMFNET"], device)
            output_DDF_3 = get_output_ddfseg(image3, paths["name_DDF"], device)
            output_PR_3 = get_output_pr(image3, paths["name_PR"], device)

    masked_gt1 = np.ma.masked_where(gt1[0, 0, :, :] == 0, gt1[0, 0, :, :])
    masked_gt2 = np.ma.masked_where(gt2[0, 0, :, :] == 0, gt2[0, 0, :, :])
    masked_gt3 = np.ma.masked_where(gt3[0, 0, :, :] == 0, gt3[0, 0, :, :])
    tiles = ["Input Image", "UNet (NA)","vMFNet", "DDFSeg", "DRIT + UNet", "DRIT + RUNet", "Proposed", "UNet (FS)", "Ground Truth"]
    data_1 = [image1[0, 0, :, :].detach().cpu(), output_NA_1, output_VM_1, output_DDF_1,  output_DU_1, output_DR_1, output_PR_1, output_FS_1, masked_gt1]
    data_2 = [image2[0, 0, :, :].detach().cpu(), output_NA_2, output_VM_2, output_DDF_2,  output_DU_2, output_DR_2, output_PR_2, output_FS_2, masked_gt2]
    data_3 = [image3[0, 0, :, :].detach().cpu(), output_NA_3, output_VM_3, output_DDF_3,  output_DU_3, output_DR_3, output_PR_3, output_FS_3, masked_gt3]

    visualize_baselines(tiles, data_1, data_2, data_3, source=source, target=target, label=label, cmap_col=cmap_col)

def main():
    pl.seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # change test folds and max_images accordingly
    test_cases_fold_2 = [2, 13, 16, 18]
    max_images = 137
    i1 = random.randrange(max_images)
    i2 = random.randrange(max_images)
    i3 = random.randrange(max_images)
    
    # Change data_dir to the path of the data and stored models accordingly
    data_dir = "data/CHAOS/T1"
    paths_liver_T1 = {
        "name_NA": "baselines/checkpoints_models/UNet_trained_on_T2_Liver/fold_2/",
        "name_FS": "baselines/checkpoints_models/UNet_trained_on_T1_Liver/fold_2/",
        "name_DU": "baselines/checkpoints_models/UNet_trained_on_fake_T1_Liver/fold_2/",
        "name_DR": "baselines/checkpoints_models/ResUNet_trained_on_fake_T1_Liver/fold_2/",
        "name_VMFNET": "baselines/vMFNet/checkpoints/single_T2_liver/fold_2/",
        "name_PR": "src/checkpoints_true/proposed_liver/TargetT1/xavier_init_true_opt_10_newmetric/fold_2/",
        "name_DDF": "baselines/DDFSeg/results/Liver_CHAOS_T1/fold_2/"
    }
    
    show_results(device, data_dir, paths_liver_T1, test_cases_fold_2, i1, i2, i3, source="T2", target="T1", label="Liver")


if __name__ == '__main__':
    main()