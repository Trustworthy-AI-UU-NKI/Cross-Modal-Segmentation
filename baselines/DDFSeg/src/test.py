import torch
from torch.utils.data import DataLoader
from dataset import MMWHS_single
from saver import Saver
from model import DDFSeg
import argparse
import sys
import pytorch_lightning as pl
import numpy as np
from utils import *
import glob

from sklearn.model_selection import KFold
from monai.metrics import DiceMetric
import os

def test(save_dir, data_loader, device, num_classes):
    pretrained_model = glob.glob(os.path.join(save_dir, "*.pth"))

    if pretrained_model == []:
        print("no pretrained model found!")
        quit()
    else:
        model_file = pretrained_model[0]

    print("Loading model")
    model = DDFSeg(args, num_classes)
    model.setgpu(device)
    model.resume(model_file)
    model.eval()

    print("Testing")
    dice_class_tot = 0
    dice_bg_tot = 0
    assd_classes_tot = 0
    for it, (images, labels) in enumerate(data_loader):
        with torch.no_grad():
        # compute self.dice_b_mean --> with keep_rate = 1
            images = images.to(device)
            labels = labels.to(device) # one channel with different numbers
            pred_mask_b = model.forward_eval(images) # multiple channel with softmax
            dice_b_mean = dice(labels.cpu(), pred_mask_b.cpu(), model.num_classes)
            assd_classes = assd(labels, pred_mask_b, num_classes, pixdim=images.meta["pixdim"][1])
           
        dice_class_tot += dice_b_mean.cpu().numpy()[1]
        dice_bg_tot += dice_b_mean.cpu().numpy()[0]
        assd_classes_tot += assd_classes.item()
    

    assd_classes_tot /= len(data_loader)
    dice_class_tot /= len(data_loader)
    dice_bg_tot /= len(data_loader)

    return dice_class_tot, dice_bg_tot, assd_classes_tot


def main(args):
    labels, num_classes = get_labels(args.pred)

    # MMWHS Dataset
    if args.data_type == "MMWHS":
        dataset_type = MMWHS_single
        in_channels = 1
    elif args.data_type == "RetinalVessel":
        NotImplementedError
        # dataset_type = Retinal_Vessel
        # in_channels = 3
    else:
        raise ValueError(f"Data type {args.data_type} not supported")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cases = range(0,20)
    kf = KFold(n_splits=args.k_folds, shuffle=True)
    fold = 0
    dsc_scores_BG = []
    dsc_scores = []
    assd_scores = []
    # for fold_train, fold_test_val in kf.split(cases):
    for fold_train_val, fold_test in kf.split(cases):
        save_dir = os.path.join(args.result_dir,  os.path.join(args.name, f'fold_{fold}'))
        os.makedirs(save_dir, exist_ok=True)
        log_dir = os.path.join(args.display_dir, os.path.join(args.name, f'fold_{fold}'))
        os.makedirs(log_dir, exist_ok=True)
        
        print("loading test data")
        dataset_test = dataset_type(args, labels, fold_test, fold_test) 
        test_loader = DataLoader(dataset_test, batch_size=1, num_workers=4)

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
    parser.add_argument('--test_data_dir', type=str, default='../../../data/other/CT_withGT_proc/', help='path of data to domain 1 - source domain')
    parser.add_argument('--resume', type=str, default='../results/trialepoch100/', help='specified the dir of saved models for resume the training')

    # testing related
    parser.add_argument('--pred', default='MYO', type=str,help='Prediction of which label') # MYO, LV, RV, MYO_RV, MYO_LV_R
    parser.add_argument('--seed', default=42, type=int,help='Seed to use for reproducing results')
    parser.add_argument('--lr', default=0.0002, type=float, help='Learning rate')
    parser.add_argument('--lr67', default=0.001, type=float, help='Learning rate for the segmentor')
    parser.add_argument('--lr5', default=0.01, type=float, help='Learning rate for the zero_loss')   
    parser.add_argument('--lr_A', default=10, type=float, help='Learning rate (lambda A)')
    parser.add_argument('--lr_B', default=10, type=float, help='Learning rate (lambda B)')
    parser.add_argument('--bs', default=4, type=int,help='batch_size')
    parser.add_argument('--keep_rate', default=0.75, type=float, help='Keep rate for dropout')
    parser.add_argument('--resolution', default=256, type=int, help='Resolution of input images')
    parser.add_argument('--skip', default=True, type=bool, help='Skip connection in the generator')
    
    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)

    main(args)
