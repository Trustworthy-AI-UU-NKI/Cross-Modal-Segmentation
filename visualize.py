# imports 
import numpy as np
from monai.transforms import (
    Compose,
    LoadImaged,
    LoadImage,
    ScaleIntensityd,
    MapLabelValued,
    AsDiscrete
)
from monai.data import  decollate_batch

from monai.data import Dataset, DataLoader
import os

import matplotlib.pyplot as plt
import glob
import pytorch_lightning as pl
import torch

import torch

from monai.networks.nets import UNet

from baselines.helpers import UNet_model
from baselines.data import MMWHS_single, CHAOS
import sys

sys.path.insert(0, 'vMFNet/')
from vMFNet.models.crosscompcsd import CrossCSD
from vMFNet.models.compcsd import CompCSD

class Args:
    def __init__(self):
        self.pretrain = "xavier"
        self.weight_init = "xavier"
        self.init = "xavier"
        self.layer = 8
        self.vc_num = 10
        self.true_clu_loss = True
        self.k2 = 10
        self.learning_rate = 0.0001

def get_output_unet(image, model_name, device, unet=True):
    if unet:
        model = UNet_model(2).to(device)
    else: 
        model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(device)
    pretrained_model = glob.glob(os.path.join(model_name, "*.pth"))[0]
    # Load the state dictionary
    model.load_state_dict(torch.load(pretrained_model))
    model.eval()
    post_trans = Compose([AsDiscrete(argmax=True)])
    outputs = model.forward(image.to(device)).detach().cpu()
    vis_outputs = post_trans(outputs[0])
    masked_output = np.ma.masked_where(vis_outputs[0, :, :] == 0, vis_outputs[0, :, :])
    return masked_output

def get_output_pr(image, model_name, device):
    args = Args()
    pretrained_model = glob.glob(os.path.join(model_name, "*.pth"))[0]
    model = CrossCSD(args, device, 1, 2, vMF_kappa=30, fold_nr=0)
    model.to(device)
    model.resume(pretrained_model)
    model.eval()
    com_features_t, compact_pred_t = model.forward_test(image)
    compact_pred_t = compact_pred_t.detach().cpu()
    masked_output = np.ma.masked_where(compact_pred_t[0, 0, :, :] == 0, compact_pred_t[0, 0, :, :])
    return masked_output

def get_output_vmfnet(image, model_name, device):
    pretrained_model = glob.glob(os.path.join(model_name, "*.pth"))[0]
    model = CompCSD(device, 1, 8, 12, num_classes=2, z_length=8, vMF_kappa=30, init='xavier')
    model.initialize(pretrained_model, init="xavier") # Does not matter for testing
    model.to(device)
    model.resume(pretrained_model)
    model.eval()

    rec, pre_seg, features, kernels, L_visuals = model(image)
   
    compact_pred = torch.argmax(pre_seg, dim=1).unsqueeze(1).detach().cpu()
    masked_output = np.ma.masked_where(compact_pred[0, 0, :, :] == 0, compact_pred[0, 0, :, :])
    return masked_output


def visualize_baselines(tiles, data_1, data_2, data_3, source='MRI', target='CT', label="MYO", cmap_col='spring'):
    fig = plt.figure("visualize", (24, 10))

    for i in range(len(tiles)):
        plt.subplot(3, len(tiles), i + 1)
        plt.title(tiles[i], fontsize=25)
        plt.imshow(data_1[0], cmap="gray")
        if not (i==0):
            plt.imshow(data_1[i], alpha=0.4, cmap=cmap_col) 
        plt.axis("off")
        
        plt.subplot(3, len(tiles), i + 1 + len(tiles))
        plt.imshow(data_2[0], cmap="gray")
        if not (i==0):
            plt.imshow(data_2[i], alpha=0.4, cmap=cmap_col) 
        plt.axis("off")

        plt.subplot(3, len(tiles), i + 1 + 2*len(tiles))
        plt.imshow(data_3[0], cmap="gray")
        if not (i==0):
            plt.imshow(data_3[i], alpha=0.4, cmap=cmap_col) 
        plt.axis("off")
    
    plt.tight_layout() 
    plt.savefig(f'results_{source}->{target}_{label}.png', format='png', bbox_inches='tight')
    plt.show()
    plt.close(fig)


def show_results(device, data_dir, paths, test_cases_fold_0, i1, i2, i3, source, target, label):
    
    print(label)
    if label == "MYO":
        labels = [1, 0, 0, 0, 0, 0, 0]
        dataset_test = MMWHS_single(data_dir, test_cases_fold_0, labels) 
        cmap_col='winter'
    elif label == "LV":
        labels = [0, 0, 1, 0, 0, 0, 0]
        dataset_test = MMWHS_single(data_dir, test_cases_fold_0, labels)
        cmap_col='spring'
    elif label == "RV":
        labels = [0, 0, 0, 0, 1, 0, 0]
        dataset_test = MMWHS_single(data_dir, test_cases_fold_0, labels)
        cmap_col='cool'
    elif label == "Liver":
        labels = [1, 0, 0, 0]
        dataset_test = CHAOS(data_dir, test_cases_fold_0, labels)
        cmap_col='autumn' 
    elif label == "Spleen":
        labels = [0, 0, 0, 1]
        dataset_test = CHAOS(data_dir, test_cases_fold_0, labels) 
        cmap_col='Wistia'
    
    test_loader = DataLoader(dataset_test, batch_size=1, num_workers=4)
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
        
        if i == i2:
            image2 = im.to(device)
            gt2 = lab
            output_FS_2 = get_output_unet(image2, paths["name_FS"], device)
            output_NA_2 = get_output_unet(image2, paths["name_NA"], device)
            output_DU_2 = get_output_unet(image2, paths["name_DU"], device)
            output_DR_2 = get_output_unet(image2, paths["name_DR"], device, False)
            output_VM_2 = get_output_vmfnet(image2, paths["name_VMFNET"], device)
            output_PR_2 = get_output_pr(image2, paths["name_PR"], device)
        
        if i == i3:
            image3 = im.to(device)
            gt3 = lab
            output_FS_3 = get_output_unet(image3, paths["name_FS"], device)
            output_NA_3 = get_output_unet(image3, paths["name_NA"], device)
            output_DU_3 = get_output_unet(image3, paths["name_DU"], device)
            output_DR_3 = get_output_unet(image3, paths["name_DR"], device, False)
            output_VM_3 = get_output_vmfnet(image3, paths["name_VMFNET"], device)
            output_PR_3 = get_output_pr(image3, paths["name_PR"], device)

    masked_gt1 = np.ma.masked_where(gt1[0, 0, :, :] == 0, gt1[0, 0, :, :])
    masked_gt2 = np.ma.masked_where(gt2[0, 0, :, :] == 0, gt2[0, 0, :, :])
    masked_gt3 = np.ma.masked_where(gt3[0, 0, :, :] == 0, gt3[0, 0, :, :])
    tiles = ["Input Image", "UNet (NA)","vMFNet", "DRIT + UNet", "DRIT + RUNet", "Proposed", "UNet (FS)", "Ground Truth"]
    data_1 = [image1[0, 0, :, :].detach().cpu(), output_NA_1, output_VM_1, output_DU_1, output_DR_1, output_PR_1, output_FS_1, masked_gt1]
    data_2 = [image2[0, 0, :, :].detach().cpu(), output_NA_2, output_VM_2, output_DU_2, output_DR_2, output_PR_2, output_FS_2, masked_gt2]
    data_3 = [image3[0, 0, :, :].detach().cpu(), output_NA_3, output_VM_3, output_DU_3, output_DR_3, output_PR_3, output_FS_3, masked_gt3]

    visualize_baselines(tiles, data_1, data_2, data_3, source=source, target=target, label=label, cmap_col=cmap_col)

def main():
    pl.seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    test_cases_fold_0 = [0, 1, 15, 17]

    data_dir = "data/other/CT_withGT_proc"
    paths_MYO_CT = {
        "name_FS": "baselines/checkpoints/Training_CT_UNET_MYO/fold_0/",
        "name_NA": "baselines/checkpoints/Training_MRI_UNET_MYO/fold_0/",
        "name_DU": "baselines/checkpoints/Training_fake_CT_UNET_MYO/fold_0/",
        "name_DR": "baselines/checkpoints/Training_fake_CT_MYO/fold_0/",
        "name_VMFNET": "vMFNet/checkpoints/single_MRI_MYO/fold_0/",
        "name_PR": "vMFNet/checkpoints/proposed_MYO/xavier_init_true_opt_10_newmetric/fold_0/",
    }
    i1 = 0
    i2 = 30
    i3 = 40
    show_results(device, data_dir, paths_MYO_CT, test_cases_fold_0, i1, i2, i3, source="MRI", target="CT", label="MYO")

    # paths_LV_CT = {
    #     "name_FS": "baselines/checkpoints/Training_CT_UNET_LV/fold_0/",
    #     "name_NA": "baselines/checkpoints/Training_MRI_UNET_LV/fold_0/",
    #     "name_DU": "baselines/checkpoints/Training_fake_CT_UNET_LV/fold_0/",
    #     "name_DR": "baselines/checkpoints/Training_fake_CT_LV/fold_0/",
    #     "name_VMFNET": "vMFNet/checkpoints/single_MRI_LV/fold_0/",
    #     "name_PR": "vMFNet/checkpoints/proposed_LV/xavier_init_true_opt_10_newmetric/fold_0/",
    # }
    # i1 = 0
    # i2 = 0
    # i3 = 40
    # show_results(device, data_dir, paths_LV_CT, test_cases_fold_0, i1, i2, i3, source="MRI", target="CT", label="LV")

    # paths_RV_CT = {
    #     "name_FS": "baselines/checkpoints/Training_CT_UNET_RV/fold_0/",
    #     "name_NA": "baselines/checkpoints/Training_MRI_UNET_RV/fold_0/",
    #     "name_DU": "baselines/checkpoints/Training_fake_CT_UNET_RV/fold_0/",
    #     "name_DR": "baselines/checkpoints/Training_fake_CT_RV/fold_0/", 
    #     "name_VMFNET": "vMFNet/checkpoints/single_MRI_RV/fold_0/",
    #     "name_PR": "vMFNet/checkpoints/proposed_RV/xavier_init_true_opt_10_newmetric/fold_0/"
    # }
    # i1 = 0
    # i2 = 0
    # i3 = 40
    # show_results(device, data_dir, paths_RV_CT, test_cases_fold_0, i1, i2, i3, source="MRI", target="CT", label="RV")


    # data_dir = "data/other/MR_withGT_proc"
    # paths_MYO_MRI = {
    #     "name_NA": "baselines/checkpoints/Training_CT_UNET_MYO/fold_0/",
    #     "name_FS": "baselines/checkpoints/Training_MRI_UNET_MYO/fold_0/",
    #     "name_DU": "baselines/checkpoints/Training_fake_MR_UNET_MYO/fold_0/",
    #     "name_DR": "baselines/checkpoints/Training_fake_MR_MYO/fold_0/",
    #     "name_VMFNET": "vMFNet/checkpoints/single_CT_MYO/fold_0/",
    #     "name_PR": "vMFNet/checkpoints/proposed_MYO/xavier_init_true_opt_10_newmetric_TMRI/fold_0/",
    # }
    # i1 = 0
    # i2 = 0
    # i3 = 40
    # show_results(device, data_dir, paths_MYO_MRI, test_cases_fold_0, i1, i2, i3, source="CT", target="MRI", label="MYO")
    
    # paths_LV_MRI = {
    #     "name_NA": "baselines/checkpoints/Training_CT_UNET_LV/fold_0/",
    #     "name_FS": "baselines/checkpoints/Training_MRI_UNET_LV/fold_0/",
    #     "name_DU": "baselines/checkpoints/Training_fake_MR_UNET_LV/fold_0/",
    #     "name_DR": "baselines/checkpoints/Training_fake_MR_LV/fold_0/",
    #     "name_VMFNET": "vMFNet/checkpoints/single_CT_LV/fold_0/",
    #     "name_PR": "vMFNet/checkpoints/proposed_LV/xavier_init_true_opt_10_newmetric_TMRI/fold_0/",
    # }
    # i1 = 0
    # i2 = 0
    # i3 = 40
    # show_results(device, data_dir, paths_LV_MRI, test_cases_fold_0, i1, i2, i3, source="CT", target="MRI", label="LV")


    # paths_RV_MRI = {
    #     "name_NA": "baselines/checkpoints/Training_CT_UNET_RV/fold_0/",
    #     "name_FS": "baselines/checkpoints/Training_MRI_UNET_RV/fold_0/",
    #     "name_DU": "baselines/checkpoints/Training_fake_MR_UNET_RV/fold_0/",
    #     "name_DR": "baselines/checkpoints/Training_fake_MR_RV/fold_0/", 
    #     "name_VMFNET": "vMFNet/checkpoints/single_CT_RV/fold_0/",
    #     "name_PR": "vMFNet/checkpoints/proposed_RV/xavier_init_true_opt_10_newmetric_TMRI/fold_0/"
    # }
    # i1 = 0
    # i2 = 0
    # i3 = 40
    # show_results(device, data_dir, paths_RV_MRI, test_cases_fold_0, i1, i2, i3, source="CT", target="MRI", label="RV")

    # data_dir = "data/preprocessed_chaos/T1"
    # paths_liver_T1 = {
    #     "name_NA": "baselines/checkpoints/Training_T2_UNET_liver/fold_0/",
    #     "name_FS": "baselines/checkpoints/Training_T1_UNET_liver/fold_0/",
    #     "name_DU": "baselines/checkpoints/Training_fake_T1_UNET_liver/fold_0/",
    #     "name_DR": "baselines/checkpoints/Training_fake_T1_liver/fold_0/",
    #     "name_VMFNET": "vMFNet/checkpoints/single_T2_liver/fold_0/",
    #     "name_PR": "vMFNet/checkpoints/proposed_liver/TargetT1/xavier_init_true_opt_10_newmetric/fold_0/"
    # }
    # i1 = 0
    # i2 = 0
    # i3 = 40
    # show_results(device, data_dir, paths_liver_T1, test_cases_fold_0, i1, i2, i3, source="T2", target="T1", label="Liver")

    # paths_spleen_T1 = {"name_NA": "baselines/checkpoints/Training_T2_UNET_Spleen/fold_0/",
    #     "name_FS": "baselines/checkpoints/Training_T1_UNET_Spleen/fold_0/",
    #     "name_DU": "baselines/checkpoints/Training_fake_T1_UNET_Spleen/fold_0/",
    #     "name_DR": "baselines/checkpoints/Training_fake_T1_Spleen/fold_0/",
    #     "name_VMFNET": "vMFNet/checkpoints/single_T2_spleen/fold_0/",
    #     "name_PR": "vMFNet/checkpoints/proposed_spleen/TargetT1/xavier_init_true_opt_10_newmetric/fold_0/"
    # }
    # i1 = 0
    # i2 = 0
    # i3 = 40
    # show_results(device, data_dir, paths_spleen_T1, test_cases_fold_0, i1, i2, i3, source="T2", target="T1", label="Spleen")

    # data_dir = "data/preprocessed_chaos/T2"
    # paths_liver_T2 = {
    #     "name_NA": "baselines/checkpoints/Training_T1_UNET_liver/fold_0/",
    #     "name_FS": "baselines/checkpoints/Training_T2_UNET_liver/fold_0/",
    #     "name_DU": "baselines/checkpoints/Training_fake_T2_UNET_liver/fold_0/",
    #     "name_DR": "baselines/checkpoints/Training_fake_T2_liver/fold_0/",
    #     "name_VMFNET": "vMFNet/checkpoints/single_T1_liver/fold_0/",
    #     "name_PR": "vMFNet/checkpoints/proposed_liver/TargetT2/xavier_init_true_opt_10_newmetric/fold_0/"
    # }
    # i1 = 0
    # i2 = 0
    # i3 = 40
    # show_results(device, data_dir, paths_liver_T2, test_cases_fold_0, i1, i2, i3, source="T1", target="T2", label="Liver")

    # paths_spleen_T2 = {
    #     "name_NA": "baselines/checkpoints/Training_T1_UNET_Spleen/fold_0/",
    #     "name_FS": "baselines/checkpoints/Training_T2_UNET_Spleen/fold_0/",
    #     "name_DU": "baselines/checkpoints/Training_fake_T2_UNET_Spleen/fold_0/",
    #     "name_DR": "baselines/checkpoints/Training_fake_T2_Spleen/fold_0/",
    #     "name_VMFNET": "vMFNet/checkpoints/single_T1_spleen/fold_0/",
    #     "name_PR": "vMFNet/checkpoints/proposed_spleen/TargetT2/xavier_init_true_opt_10_newmetric/fold_0/"
    # }
    # i1 = 0
    # i2 = 0
    # i3 = 40
    # show_results(device, data_dir, paths_spleen_T2, test_cases_fold_0, i1, i2, i3, source="T1", target="T2", label="Spleen")

if __name__ == '__main__':
    main()