# imports 
import sys
import os
import random
import torch
import glob

from monai.data import DataLoader
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from baselines.dataloaders import MMWHS, CHAOS
sys.path.insert(0, 'src/')
from src.models.crosscompcsd import CrossCSD

# Arguments for proposed model
class Args:
    def __init__(self):
        self.weight_init = "xavier"
        self.vc_num = 10
        self.k2 = 10
        self.learning_rate = 0.0001

# Get output model
def get_output(image, model_name, device):
    args = Args()
    pretrained_model = glob.glob(os.path.join(model_name, "*.pth"))[0]
    model = CrossCSD(args, device, 1, 2, vMF_kappa=30, fold_nr=0)
    model.to(device)
    model.resume(pretrained_model)
    model.eval()
    com_features_t, compact_pred_t, pre_seg_t = model.test(image)

    compact_pred_t = compact_pred_t[0, 0, :, :].detach().cpu()
    com_features_t = com_features_t[0].detach().cpu()
    return com_features_t, compact_pred_t

# Visulize results and store in pdf for better resolution
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

# Get results of trained model
def get_data(device, data_dir, path, test_cases_fold_0, label, item=0):
  
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
        if i == item:
            image1 = im.to(device)
            gt1 = lab
            com_features1, output_PR_Liver1 = get_output(image1, path, device)
            break
        
            
    data1 = [image1[0, 0, :, :].detach().cpu(), gt1[0, 0, :, :], output_PR_Liver1.detach().cpu(), com_features1[0], com_features1[1], com_features1[2], com_features1[3], com_features1[4],
                com_features1[5], com_features1[6], com_features1[7], com_features1[8], com_features1[9]]
    
    return data1


def main():
    pl.seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # change test folds and max_images accordingly
    test_cases_fold_1 = [3, 5, 8, 11]
    max_images = 64
    item1 = random.randrange(max_images)
    item2 = random.randrange(max_images)
    item3 = random.randrange(max_images)

    # Change data_dir to the path of the data
    data_dir = "data/MMWHS/CT_withGT_proc"

    # Change all the paths to the stored models accordingly
    path = "src/checkpoints_true/proposed_MYO/xavier_init_true_opt_10_newmetric/fold_1/"
    data_MYO = get_data(device, data_dir, path, test_cases_fold_1, "MYO", item=item1)

    path = "src/checkpoints_true/proposed_LV/xavier_init_true_opt_10_newmetric/fold_1/"
    data_LV = get_data(device, data_dir, path, test_cases_fold_1, "LV", item=item2)

    path = "src/checkpoints_true/proposed_RV/xavier_init_true_opt_10_newmetric/fold_1/"
    data_RV = get_data(device, data_dir, path, test_cases_fold_1, "RV", item=item3)
    tiles = ["Image", "GT",  "Predicted", r"$Z_{vMF}(1)$", r"$Z_{vMF}(2)$", r"$Z_{vMF}(3)$", r"$Z_{vMF}(4)$", r"$Z_{vMF}(5)$", 
             r"$Z_{vMF}(6)$", r"$Z_{vMF}(7)$", r"$Z_{vMF}(8)$", r"$Z_{vMF}(9)$",  r"$Z_{vMF}(10)$"]
    
    visualize(tiles, data_MYO, data_LV, data_RV, "MYO_LV_RV", source="MRI", target="CT")

if __name__ == '__main__':
    main()