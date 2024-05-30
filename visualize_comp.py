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
    compact_pred_t = compact_pred_t.detach().cpu()
    masked_output = np.ma.masked_where(compact_pred_t[0, 0, :, :] == 0, compact_pred_t[0, 0, :, :])
    return com_features_t, masked_output


def visualize(names, data1, data2, label, source='T1', target='T2'):
    fig = plt.figure("visualize", (16, 4))

    for i in range(len(names)):
        plt.subplot(2, len(names), i + 1)
        plt.title(names[i], fontsize=27)
        plt.imshow(data1[i], cmap="gray")
        plt.axis("off")

        plt.subplot(2, len(names), i + 1 + len(names))
        plt.imshow(data2[i], cmap="gray")
        plt.axis("off")
    
    plt.tight_layout() 
    plt.savefig(f'results_{source}->{target}_{label}.png', format='png', bbox_inches='tight')
    plt.show()
    plt.close(fig)

def get_data(device, data_dir, path, test_cases_fold_0, source, target, label, item1=0, item2=10):
  
    print(label)
    if label == "MYO":
        labels = [1, 0, 0, 0, 0, 0, 0]
        dataset_test = MMWHS_single(data_dir, test_cases_fold_0, labels) 
    elif label == "LV":
        labels = [0, 0, 1, 0, 0, 0, 0]
        dataset_test = MMWHS_single(data_dir, test_cases_fold_0, labels)
    elif label == "RV":
        labels = [0, 0, 0, 0, 1, 0, 0]
        dataset_test = MMWHS_single(data_dir, test_cases_fold_0, labels)
    elif label == "Liver":
        labels = [1, 0, 0, 0]
        dataset_test = CHAOS(data_dir, test_cases_fold_0, labels) 
    elif label == "Spleen":
        labels = [0, 0, 0, 1]
        dataset_test = CHAOS(data_dir, test_cases_fold_0, labels) 
    test_loader = DataLoader(dataset_test, batch_size=1, num_workers=4)
   
    for i, (im, lab) in enumerate(test_loader):
        if i == item1:
            image1 = im.to(device)
            gt1 = lab
            com_features1, output_PR_Liver1 = get_output(image1, path, device)
        
        if i == item2:
            image2 = im.to(device)
            gt2 = lab
            com_features2, output_PR_Liver2 = get_output(image2, path, device)
            break
            
    tiles = ["Image", "GT",  "Predited", r"$Z_{vMF}(1)$", r"$Z_{vMF}(2)$", r"$Z_{vMF}(3)$", r"$Z_{vMF}(4)$", r"$Z_{vMF}(5)$", 
             r"$Z_{vMF}(6)$", r"$Z_{vMF}(7)$", r"$Z_{vMF}(8)$", r"$Z_{vMF}(9)$",  r"$Z_{vMF}(10)$"]
    data1= [image1[0, 0, :, :].detach().cpu(), gt1, output_PR_Liver1, com_features1[0], com_features1[1], com_features1[2], com_features1[3], com_features1[4],
                com_features1[5], com_features1[6], com_features1[7], com_features1[8], com_features1[9]]
    data2= [image2[0, 0, :, :].detach().cpu(), gt2, output_PR_Liver2, com_features2[0], com_features2[1], com_features2[2], com_features2[3], com_features2[4],
                com_features2[5], com_features2[6], com_features2[7], com_features2[8], com_features2[9]]

    visualize(tiles, data1, data2, label, source=source, target=target)


def main():
    pl.seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    test_cases_fold_0 = [0, 1, 15, 17]

    data_dir = "data/other/CT_withGT_proc"
    
    path = "vMFNet/checkpoints/proposed_MYO/xavier_init_true_opt_10_newmetric/fold_0/"
    get_data(device, data_dir, path, test_cases_fold_0, "MRI", "CT", "MYO", item1=0, item2=10)
    path = "vMFNet/checkpoints/proposed_LV/xavier_init_true_opt_10_newmetric/fold_0/"
    get_data(device, data_dir, path, test_cases_fold_0, "MRI", "CT", "LV", item1=0, item2=10)
    path = "vMFNet/checkpoints/proposed_RV/xavier_init_true_opt_10_newmetric/fold_0/"
    get_data(device, data_dir, path, test_cases_fold_0, "MRI", "CT", "RV", item1=0, item2=10)

    data_dir = "data/other/MR_withGT_proc"
    path = "vMFNet/checkpoints/proposed_MYO/xavier_init_true_opt_10_newmetric_TMRI/fold_0/"
    get_data(device, data_dir, path, test_cases_fold_0, "CT", "MRI", "MYO", item1=0, item2=10)
    path = "vMFNet/checkpoints/proposed_LV/xavier_init_true_opt_10_newmetric_TMRI/fold_0/"
    get_data(device, data_dir, path, test_cases_fold_0, "CT", "MRI", "LV", item1=0, item2=10)
    path = "vMFNet/checkpoints/proposed_RV/xavier_init_true_opt_10_newmetric_TMRI/fold_0/"
    get_data(device, data_dir, path, test_cases_fold_0, "CT", "MRI", "RV", item1=0, item2=10)
    

    data_dir = "data/preprocessed_chaos/T1"
    path = "vMFNet/checkpoints/proposed_liver/TargetT1/xavier_init_true_opt_10_newmetric/fold_0/"
    get_data(device, data_dir, path, test_cases_fold_0, "T2", "T1", "Liver", item1=0, item2=10)
    path ="vMFNet/checkpoints/proposed_spleen/TargetT1/xavier_init_true_opt_10_newmetric/fold_0/"
    get_data(device, data_dir, path, test_cases_fold_0, "T2", "T1", "Spleen", item1=0, item2=10)


    data_dir = "data/preprocessed_chaos/T2"
    path = "vMFNet/checkpoints/proposed_liver/TargetT2/xavier_init_true_opt_10_newmetric/fold_0/"
    get_data(device, data_dir, path, test_cases_fold_0, "T1", "T2", "Liver", item1=0, item2=10)
    path ="vMFNet/checkpoints/proposed_spleen/TargetT2/xavier_init_true_opt_10_newmetric/fold_0/"
    get_data(device, data_dir, path, test_cases_fold_0, "T1", "T2", "Spleen", item1=0, item2=10)

if __name__ == '__main__':
    main()