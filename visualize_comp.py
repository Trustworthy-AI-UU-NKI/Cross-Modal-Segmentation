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

from baselines.UNet import UNet_model
from baselines.dataloaders import MMWHS, CHAOS
import sys

sys.path.insert(0, 'src/')
from src.models.crosscompcsd import CrossCSD

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

def get_output(image, model_name, device):
    args = Args()
    pretrained_model = glob.glob(os.path.join(model_name, "*.pth"))[0]
    model = CrossCSD(args, device, 1, 2, vMF_kappa=30, fold_nr=0)
    model.to(device)
    model.resume(pretrained_model)
    model.eval()
    com_features_t, compact_pred_t = model.forward_test(image)
    compact_pred_t = compact_pred_t[0, 0, :, :].detach().cpu()
    com_features_t = com_features_t[0].detach().cpu()
    return com_features_t, compact_pred_t


def visualize(names, data1, data2, data3, label, source='T1', target='T2'):
    fig = plt.figure("visualize", (24, 6))

    for i in range(len(names)):
        plt.subplot(3, len(names), i + 1)
        plt.title(names[i], fontsize=21)
        plt.imshow(data1[i], cmap="gray")
        plt.axis("off")

        plt.subplot(3, len(names), i + 1 + len(names))
        plt.imshow(data2[i], cmap="gray")
        plt.axis("off")

        plt.subplot(3, len(names), i + 1 + 2*len(names))
        plt.imshow(data3[i], cmap="gray")
        plt.axis("off")
    
    plt.tight_layout() 
    plt.savefig(f'results/comp_{source}->{target}_{label}.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    plt.close(fig)

def get_data(device, data_dir, path, test_cases_fold_0, source, target, label, item1=0, item2=10, item3=20):
  
    print(label)
    if label == "MYO":
        labels = [1, 0, 0, 0, 0, 0, 0]
        dataset_test = MMWHS(data_dir, test_cases_fold_0, labels) 
    elif label == "LV":
        labels = [0, 0, 1, 0, 0, 0, 0]
        dataset_test = MMWHS(data_dir, test_cases_fold_0, labels)
    elif label == "RV":
        labels = [0, 0, 0, 0, 1, 0, 0]
        dataset_test = MMWHS(data_dir, test_cases_fold_0, labels)
    elif label == "Liver":
        labels = [1, 0, 0, 0]
        dataset_test = CHAOS(data_dir, test_cases_fold_0, labels) 

    test_loader = DataLoader(dataset_test, batch_size=1)
   
    for i, (im, lab) in enumerate(test_loader):
        if i == item1:
            image1 = im.to(device)
            gt1 = lab
            com_features1, output_PR_Liver1 = get_output(image1, path, device)
        
        if i == item2:
            image2 = im.to(device)
            gt2 = lab
            com_features2, output_PR_Liver2 = get_output(image2, path, device)
        
        if i == item3:
            image3 = im.to(device)
            gt3 = lab
            com_features3, output_PR_Liver3 = get_output(image3, path, device)
            
            
        
    tiles = ["Image", "GT",  "Predicted", r"$Z_{vMF}(1)$", r"$Z_{vMF}(2)$", r"$Z_{vMF}(3)$", r"$Z_{vMF}(4)$", r"$Z_{vMF}(5)$", 
             r"$Z_{vMF}(6)$", r"$Z_{vMF}(7)$", r"$Z_{vMF}(8)$", r"$Z_{vMF}(9)$",  r"$Z_{vMF}(10)$"]
    data1 = [image1[0, 0, :, :].detach().cpu(), gt1[0, 0, :, :], output_PR_Liver1.detach().cpu(), com_features1[0], com_features1[1], com_features1[2], com_features1[3], com_features1[4],
                com_features1[5], com_features1[6], com_features1[7], com_features1[8], com_features1[9]]
    data2 = [image2[0, 0, :, :].detach().cpu(), gt2[0, 0, :, :], output_PR_Liver2.detach().cpu(), com_features2[0], com_features2[1], com_features2[2], com_features2[3], com_features2[4],
                com_features2[5], com_features2[6], com_features2[7], com_features2[8], com_features2[9]]
    data3 = [image3[0, 0, :, :].detach().cpu(), gt3[0, 0, :, :], output_PR_Liver3.detach().cpu(), com_features3[0], com_features3[1], com_features3[2], com_features3[3], com_features3[4],
                com_features3[5], com_features3[6], com_features3[7], com_features3[8], com_features3[9]]

    visualize(tiles, data1, data2, data3, label, source=source, target=target)
    # return data1#, data2, data3


def main():
    pl.seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    test_cases_fold_0 = [0, 1, 15, 17]
    test_cases_fold_1 = [3, 5, 8, 11]
    test_cases_fold_2 = [2, 13, 16, 18]
    test_cases_fold_3 = [4, 9, 12, 19]
    test_cases_fold_4 = [6, 7, 10, 14]

    # TO Create Fig 3.1 in Paper!!
    # data_dir = "data/other/CT_withGT_proc"
    # path = "src/checkpoints/proposed_MYO/xavier_init_true_opt_10_newmetric/fold_1/"
    # data_MYO = get_data(device, data_dir, path, test_cases_fold_1, "MRI", "CT", "MYO", item1=37)
    # path = "src/checkpoints/proposed_LV/xavier_init_true_opt_10_newmetric/fold_1/"
    # data_LV = get_data(device, data_dir, path, test_cases_fold_1, "MRI", "CT", "LV", item1=55)
    # path = "src/checkpoints/proposed_RV/xavier_init_true_opt_10_newmetric/fold_1/"
    # data_RV = get_data(device, data_dir, path, test_cases_fold_1, "MRI", "CT", "RV", item1=10)
    # tiles = ["Image", "GT",  "Predicted", r"$Z_{vMF}(1)$", r"$Z_{vMF}(2)$", r"$Z_{vMF}(3)$", r"$Z_{vMF}(4)$", r"$Z_{vMF}(5)$", 
    #          r"$Z_{vMF}(6)$", r"$Z_{vMF}(7)$", r"$Z_{vMF}(8)$", r"$Z_{vMF}(9)$",  r"$Z_{vMF}(10)$"]
    # visualize(tiles, data_MYO, data_LV, data_RV, "MYO_LV_RV", source="MRI", target="CT")

    # To create Fig 5.3 in thesis!!
    # data_dir = "data/other/CT_withGT_proc"
    # path = "src/checkpoints/proposed_MYO/xavier_init_true_opt_10_newmetric/fold_1/"
    # get_data(device, data_dir, path, test_cases_fold_1, "MRI", "CT", "MYO", item1=8, item2=44)  
    # path = "src/checkpoints/proposed_LV/xavier_init_true_opt_10_newmetric/fold_1/"
    # get_data(device, data_dir, path, test_cases_fold_1, "MRI", "CT", "LV", item1=25, item2=58)
    # path = "src/checkpoints/proposed_RV/xavier_init_true_opt_10_newmetric/fold_1/"
    # get_data(device, data_dir, path, test_cases_fold_1, "MRI", "CT", "RV", item1=6, item2=35)

    # data_dir = "data/other/MR_withGT_proc"
    # path = "src/checkpoints/proposed_MYO/xavier_init_true_opt_10_newmetric_TMRI/fold_1/"
    # get_data(device, data_dir, path, test_cases_fold_1, "CT", "MRI", "MYO", item1=10, item2=31)
    # path = "src/checkpoints/proposed_LV/xavier_init_true_opt_10_newmetric_TMRI/fold_1/"
    # get_data(device, data_dir, path, test_cases_fold_1, "CT", "MRI", "LV", item1=35, item2=61)
    # path = "src/checkpoints/proposed_RV/xavier_init_true_opt_10_newmetric_TMRI/fold_1/"
    # get_data(device, data_dir, path, test_cases_fold_1, "CT", "MRI", "RV", item1=7, item2=42)
    
    data_dir = "data/preprocessed_chaos/T1"
    path = "src/checkpoints/proposed_liver/TargetT1/xavier_init_true_opt_10_newmetric/fold_4/"
    get_data(device, data_dir, path, test_cases_fold_4, "T2", "T1", "Liver", item1=29, item2=48, item3=101)

    # data_dir = "data/preprocessed_chaos/T2"
    # path = "src/checkpoints/proposed_liver/TargetT2/xavier_init_true_opt_10_newmetric/fold_4/"
    # get_data(device, data_dir, path, test_cases_fold_4, "T1", "T2", "Liver", item1=57, item2=65, item3=40)

if __name__ == '__main__':
    main()