import torch
from torch.utils.data import DataLoader
from dataset import MMWHS_single, CHAOS_single
from model import DDFSeg
import argparse
import sys
import numpy as np
from utils import *
import glob

from sklearn.model_selection import KFold
import os

# Test function
def test(save_dir, data_loader, device, num_classes):

    # Get trained model weigths
    pretrained_model = glob.glob(os.path.join(save_dir, "*.pth"))
    if pretrained_model == []:
        print("no pretrained model found!")
        quit()
    else:
        model_file = pretrained_model[0]

    # Load model
    print("Loading model")
    model = DDFSeg(args, num_classes)
    model.setgpu(device)
    model.resume(model_file)
    model.eval()

    print("Testing")
    dice_class_tot = 0
    dice_bg_tot = 0
    assd_classes_tot = 0
    assd_len = len(data_loader)
    dsc_len = len(data_loader)
    
    # Test loop
    for it, (images, labels) in enumerate(data_loader):
        with torch.no_grad():
            images = images.to(device)
            labels = labels.to(device) # one channel with different numbers
            pred_mask_b = model.forward_test(images) # multiple channel with softmax

            # Calulate DSC and ASSD
            dice_b_mean = dice(labels.cpu(), pred_mask_b.cpu(), model.num_classes).numpy()
            assd_classes = assd(labels.cpu(), pred_mask_b.cpu(), num_classes, pixdim=images.meta["pixdim"][1]).numpy()

            # skip when either label or pred is 0
            if not np.isinf(assd_classes).any():
                assd_classes_tot += assd_classes
            else:
                assd_len -= 1
            
            # Skip disce class when no label is there. Background is always there
            if torch.all(labels==0):
                dsc_len -= 1
            else:
                dice_class_tot += dice_b_mean[1]
           
        dice_bg_tot += dice_b_mean[0]
    
    if assd_len == 0:
        assd_len = 1 # to avoid division by zero
        print("assd_len is very very high because predicted mask is 0 everywhere!!!")
    
    assd_classes_tot /= assd_len
    dice_class_tot /= dsc_len
    dice_bg_tot /= len(data_loader)

    return dice_class_tot, dice_bg_tot, assd_classes_tot


def main(args):
    set_seed(args.seed)
    labels, num_classes = get_labels(args.pred)
    print("Prediction: ", args.pred)

    # Get dataset
    if args.data_type == "MMWHS":
        dataset_type = MMWHS_single
    elif args.data_type == "CHOAS":
        dataset_type = CHAOS_single
    else:
        raise ValueError(f"Data type {args.data_type} not supported")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cases = range(0,20)
    kf = KFold(n_splits=args.k_folds, shuffle=True)
    fold = 0
    dsc_scores_BG = []
    dsc_scores = []
    assd_scores = []

    # Test all folds
    for fold_train_val, fold_test in kf.split(cases):
        save_dir = os.path.join(args.resume, f'fold_{fold}')
        os.makedirs(save_dir, exist_ok=True)
        
        # Get data
        print("loading test data")
        dataset_test = dataset_type(args.test_data_dir, fold_test, labels) 
        test_loader = DataLoader(dataset_test, batch_size=1, num_workers=4)

        # Test model
        dsc, dsc_bg, assd = test(save_dir, test_loader, device, num_classes)
        dsc_scores.append(dsc)
        dsc_scores_BG.append(dsc_bg)
        assd_scores.append(assd)

        fold += 1
    
    dsc_scores_BG = np.array(dsc_scores_BG)
    mean_dsc_BG = np.mean(dsc_scores_BG)
    std_dsc_BG = np.std(dsc_scores_BG)
    print("FINAL RESULTS BG")
    print("DSC_0: ", dsc_scores_BG)
    print(f"Mean DSC_0: {mean_dsc_BG}, Std DSC_0: {std_dsc_BG}")


    dsc_scores = np.array(dsc_scores)
    mean_dsc = np.mean(dsc_scores)
    std_dsc = np.std(dsc_scores)
    print("FINAL RESULTS")
    print("DSC_1: ", dsc_scores)
    print(f"Mean DSC_1: {mean_dsc}, Std DSC_1: {std_dsc}")

    assd_scores = np.array(assd_scores)
    mean_assd = np.mean(assd_scores)
    std_assd = np.std(assd_scores)
    print("ASSD: ", assd_scores)
    print(f"Mean ASSD: {mean_assd}, Std ASSD: {std_assd}")
   

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train the DDFSeg model on the MM-WHS dataset')
    
    # data loader related
    parser.add_argument('--test_data_dir', type=str, default='../../../data/MMWHS/CT_withGT_proc/', help='path of data to domain 1 - source domain')
    parser.add_argument('--resume', type=str, default='../results/MYO_MMWHS/', help='specified the dir of saved models for resume the training')

    # testing related
    parser.add_argument('--pred', default='MYO', type=str,help='Prediction of which label') # MYO, LV, RV, Liver
    parser.add_argument('--seed', default=42, type=int,help='Seed to use for reproducing results')
    parser.add_argument('--lr', default=0.0002, type=float, help='Learning rate')
    parser.add_argument('--lr67', default=0.001, type=float, help='Learning rate for the segmentor')
    parser.add_argument('--lr5', default=0.01, type=float, help='Learning rate for the zero_loss')   
    parser.add_argument('--lr_A', default=10, type=float, help='Learning rate (lambda A)')
    parser.add_argument('--lr_B', default=10, type=float, help='Learning rate (lambda B)')
    parser.add_argument('--bs', default=1, type=int,help='batch_size')
    parser.add_argument('--keep_rate', default=0.75, type=float, help='Keep rate for dropout')
    parser.add_argument('--resolution', default=256, type=int, help='Resolution of input images')
    parser.add_argument('--skip', default=True, type=bool, help='Skip connection in the generator')
    parser.add_argument('--data_type', default="MMWHS", type=str, help='Type of data to use; MMWHS or CHAOS')
    parser.add_argument('--k_folds', default=5, type=int, help='Number of folds for cross validation')
    
    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)

    main(args)
