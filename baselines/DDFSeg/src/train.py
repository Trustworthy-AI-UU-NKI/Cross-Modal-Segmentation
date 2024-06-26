import torch
from torch.utils.data import DataLoader
from dataset import MMWHS, CHAOS
from saver import Saver
from model import DDFSeg
import argparse
import sys
import pytorch_lightning as pl
import numpy as np
from utils import *

from sklearn.model_selection import KFold
from monai.metrics import DiceMetric
import os

#  Validation function
def validation(model, data_loader, saver, device, ep, n_classes):
    dice_classes_tot_fake = np.zeros(n_classes)
    model.eval()
    print("Validation")

    # Validation loop
    for it, (images_a, labels_a, images_b, labels_b) in enumerate(data_loader):
        with torch.no_grad():
        # compute self.dice_b_mean --> with keep_rate = 1
            images_a = images_a.to(device)
            images_b = images_b.to(device)

            # Predict mask of fake target image
            pred_mask_b_false = model.forward_eval(images_a, images_b) # multiple channel with softmax
        
            # Validat only with source labels, NOT with starget labels!!!
            dice_b_mean_false = dice(labels_a.cpu(), pred_mask_b_false.cpu(), model.num_classes)
    

        dice_classes_tot_fake += dice_b_mean_false.cpu().numpy()

    dice_classes_tot_fake /= len(data_loader)

    # We count only DSC of the class, not of the background
    saver.write_val_dsc(ep, dice_classes_tot_fake[1])

    # Save best model based on source images
    if dice_classes_tot_fake[1] > model.val_dice:
        model.val_dice = dice_classes_tot_fake[1]
        saver.write_model(ep, model, dice_classes_tot_fake[1])
        print("new validation dice score on false target images:", dice_classes_tot_fake[1])


# Training function
def train(train_loader, val_loader, fold, device, args, num_classes, len_train_data, in_channels):
    # model
    print('\n--- load model ---')
    model = DDFSeg(args, channel_im=in_channels, num_classes=num_classes)
    model.setgpu(device)

    # saver for display and output
    saver = Saver(fold, args, num_classes)

    # train
    print('\n--- train ---')

    model.set_scheduler()

    total_it = 0

    # train loop
    for ep in range(args.epochs):
        print("In the epoch ", ep)
        model.train()

        # Sort of lr scheduler
        if ep < 5:
            loss_f_weight_value = 0.0
        elif ep < 7:
            loss_f_weight_value = 0.1 * (ep - 4.0) / (7.0 - 4.0)
        else:
            loss_f_weight_value = 0.1

        model.reset_losses()

        for it, (images_a, labels_a, images_b, labels_b) in enumerate(train_loader):

            # input data
            images_a = images_a.to(device)
            images_b = images_b.to(device)
            labels_a = labels_a.to(device)

            # update model
            model.update(images_a, images_b, labels_a, loss_f_weight_value)

            total_it += 1
            model.update_num_fake_inputs()

        model.update_lr(ep)
        
        saver.write_display(ep, model)

        # Save the best val model
        if ep % 5 == 0:
            validation(model, val_loader, saver, device, ep, num_classes)
  


def main(args):
    set_seed(args.seed)
    labels, num_classes = get_labels(args.pred)

    # Get Dataset
    if args.data_type == "MMWHS":
        dataset_type = MMWHS
        in_channels = 1
    elif args.data_type == "CHAOS":
        dataset_type = CHAOS
        in_channels = 1
    else:
        raise ValueError(f"Data type {args.data_type} not supported")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cases = range(0,20)
    kf = KFold(n_splits=args.k_folds, shuffle=True)
    fold = 0

    
    # K-fold cross-validation
    for fold_train_val, fold_test in kf.split(cases):
        save_dir = os.path.join(args.result_dir,  os.path.join(args.name, f'fold_{fold}'))
        os.makedirs(save_dir, exist_ok=True)
        log_dir = os.path.join(args.display_dir, os.path.join(args.name, f'fold_{fold}'))
        os.makedirs(log_dir, exist_ok=True)
        
        # Get train and validation data
        print("loading train data")
        dataset_train = dataset_type(args, labels, fold_train_val, fold_train_val)
        train_loader = DataLoader(dataset_train, batch_size=args.bs, shuffle=True, num_workers=4, drop_last=True)
        print("loading val data")
        dataset_val = dataset_type(args, labels, fold_test, fold_test) 
        val_loader = DataLoader(dataset_val, batch_size=1, num_workers=4, drop_last=True)
        len_train_data = dataset_train.__len__()

        # Train model
        train(train_loader, val_loader, fold, device, args, num_classes, len_train_data, in_channels) 
        fold += 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train the DDFSeg model on the MM-WHS dataset')
    
    # data loader related
    parser.add_argument('--data_dir1', type=str, default='../../../data/MMWHS/MR_withGT_proc/', help='path of data to domain 1 - source domain')
    parser.add_argument('--data_dir2', type=str, default='../../../data/MMWHS/CT_withGT_proc/', help='path of data to domain 2 - target domain')
    parser.add_argument('--resume', type=str, default=None, help='specified the dir of saved models for resume the training')

    # ouptput related
    parser.add_argument('--name', type=str, default='reproduce', help='folder name to save outputs')
    parser.add_argument('--display_dir', type=str, default='../logs', help='path for saving display results')
    parser.add_argument('--result_dir', type=str, default='../results', help='path for saving result images and models')
    parser.add_argument('--img_save_freq', type=int, default=5, help='freq (epoch) of saving models')
    parser.add_argument('--modality', type=str, default='MRI', help='source modality')

    # training related --> Aanpassen
    parser.add_argument('--pred', default='MYO', type=str,help='Prediction of which label') # MYO, LV, RV, Liver
    parser.add_argument('--k_folds', default=5, type=int, help='Number of folds for K-Fold Cross-Validation')
    parser.add_argument('--epochs', default=200, type=int, help='Max number of epochs')
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
    parser.add_argument('--data_type', default='MMWHS', type=str, help='Dataset used: MMWHS or CHAOS')
    
    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)

    main(args)
