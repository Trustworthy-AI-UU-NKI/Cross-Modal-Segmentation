import torch
from torch.utils.data import DataLoader
from dataset import MMWHS_single
from saver import Saver
from model import DDFSeg
import argparse
import sys
import pytorch_lightning as pl
import numpy as np
from utils import dice
import glob

from sklearn.model_selection import KFold
from monai.metrics import DiceMetric
import os

def test(model_file, data_loader, device, num_classes):

    print("Loading model")
    model = DDFSeg(args, num_classes)
    model.setgpu(device)
    model.resume(model_file)
    model.eval()

    dice_scores = []
    print("Testing")
    for it, (images, labels) in enumerate(data_loader):
        with torch.no_grad():
        # compute self.dice_b_mean --> with keep_rate = 1
            images = images.to(device)
            labels = labels.to(device) # one channel with different numbers
            pred_mask_b = model.forward_eval(images) # multiple channel with softmax
            dice_b_mean = dice(labels.cpu(), pred_mask_b.cpu(), model.num_classes)
           
        dice_scores.append(dice_b_mean)
    

    dice_scores = np.array(dice_scores)
    mean_dsc = np.mean(dice_scores)

    print("Mean Dice Score:", mean_dsc)
    return mean_dsc


def main(args):
    pl.seed_everything(args.seed)

    if args.pred == "MYO":
        labels = [1, 0, 0, 0, 0, 0, 0]
        num_classes = 2
    else:
        labels = [1, 2, 3, 4, 5, 6, 7]
        num_classes = 8
    
    test_fold = [18, 19]

    dataset_test = MMWHS_single(args, test_fold, labels)
    test_loader = DataLoader(dataset_test, batch_size=args.bs, drop_last=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_filenames = glob.glob(os.path.join(args.resume, "fold_*/*.pth"))
    
    dsc_scores = []
    pretrained_found = False
    for it, model_file in enumerate(pretrained_filenames):
        print(f"Testing model {it+1}/{len(pretrained_filenames)}")
        dsc = test(model_file, test_loader, device, num_classes)
        dsc_scores.append(dsc)
        pretrained_found = True

    if pretrained_found:
        dsc_scores = np.array(dsc_scores)
        mean_dsc = np.mean(dsc_scores)
        std_dsc = np.std(dsc_scores)
        print("FINAL RESULTS")
        print(f"Mean DSC: {mean_dsc}, Std DSC: {std_dsc}")
    else:
        print("No pretrained models found")



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train the DDFSeg model on the MM-WHS dataset')
    
    # data loader related
    parser.add_argument('--test_data_dir', type=str, default='../../../data/other/CT_withGT_proc/annotated/', help='path of data to domain 1 - source domain')
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
