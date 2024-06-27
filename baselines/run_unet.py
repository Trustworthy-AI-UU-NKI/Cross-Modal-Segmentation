import os
import torch
import argparse
import sys
import glob
import pytorch_lightning as pl
import random
import numpy as np
import torch.optim as optim

from torch.utils.data import DataLoader
from dataloaders import MMWHS, CHAOS

from monai.networks.nets import UNet # ResUNet
from UNet import UNet_model
from sklearn.model_selection import KFold 
from monai.losses import DiceLoss
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from helpers import *

# train function
def train(args, dir_checkpoint_fold, device, n_classes, train_loader, val_loader):
    
    # Get UNet or ResUNet
    if args.model == "ResUNet":
        model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=n_classes,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
    else:
        model = UNet_model(n_classes)

    model.to(device)

    best_dice = 0
    global_step = 0

    # Loss and Optimizer
    dice_loss = DiceLoss(include_background=False, softmax=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    writer = SummaryWriter(log_dir=dir_checkpoint_fold)

    # Train loop
    for epoch in range(args.epochs):
        model.train()
        for image, label in train_loader:
            image = image.to(device)
            label = label.to(device)

            # Forward pass
            outputs = model(image)

            # Calculate loss
            labels_oh = F.one_hot(label.long().squeeze(1), n_classes).permute(0, 3, 1, 2)
            loss = dice_loss(outputs, labels_oh)

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log
            writer.add_scalar('Loss/train', loss, global_step)
            global_step += 1

        # Validate
        model.eval()
        dice_tot = 0
        dice_classes_tot = np.zeros(n_classes)
        with torch.no_grad():
            for image, label in val_loader:
                image = image.to(device)
                label = label.to(device)

                # Forward pass for validation
                outputs = model(image)

                # Calculate validation metrics
                dice_classes = dice(label, outputs, n_classes)
                dice_classes_tot += dice_classes
                dice_tot += np.mean(dice_classes[1:])

            dice_tot /= len(val_loader)
            dice_classes_tot /= len(val_loader)

            writer.add_scalar('Dice/val', dice_tot, epoch)
            for item in range(n_classes):
                writer.add_scalar(f'Dice/val_{item}', dice_classes_tot[item], epoch)

        # Save model with highest validation score
        if dice_tot > best_dice:
            best_dice = dice_tot
            existing_model_files = glob.glob(f"{dir_checkpoint_fold}/*.pth")
            # Delete the file previous best model
            for file_path in existing_model_files:
                try:
                    os.remove(file_path)
                    print(f"Deleted old model file: {file_path}")
                except OSError as e:
                    print(f"Error deleting file {file_path}: {e}")
                    
            torch.save(model.state_dict(), os.path.join(dir_checkpoint_fold, f'best_model_{epoch}.pth'))
            print(f"Saved best model with dice {dice_tot} at epoch {epoch}")


# K-Fold Cross-Validation
def k_fold(args, dir_checkpoint, dataset_type, labels, device, n_classes):
    cases = range(0,20)
    kf = KFold(n_splits=args.k_folds, shuffle=True)
    fold = 0

    # For k folds
    for fold_train, fold_test in kf.split(cases):
        print(f"Fold {fold}")
        dir_checkpoint_fold = os.path.join(dir_checkpoint, f'fold_{fold}')            
        os.makedirs(dir_checkpoint_fold, exist_ok=True)

        # Data directories for DRIT are a bit different, as for each fold we have a seperate dataset from drit
        if args.drit:
            data_dir = os.path.join(args.data_dir, f"run_fold_{fold}")
        else:
            data_dir = args.data_dir

        # Split train and validation
        random.shuffle(fold_train)
        fold_train2 = fold_train[:13]
        fold_val2 = fold_train[13:]

        # Get data loaders
        dataset_train = dataset_type(data_dir, fold_train2, labels)
        train_loader = DataLoader(dataset_train, batch_size=args.bs, shuffle=True, num_workers=4)
        dataset_val = dataset_type(data_dir, fold_val2, labels) 
        val_loader = DataLoader(dataset_val, batch_size=args.bs, num_workers=4)

        # Train this fold
        train(args, dir_checkpoint_fold, device, n_classes, train_loader, val_loader)
        fold += 1


def main(args):

    pl.seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    filename = f'{args.model}_{args.name}_{args.pred}'    
    dir_checkpoint = os.path.join('checkpoints/', filename)
    os.makedirs(dir_checkpoint, exist_ok=True)
    print(args.pred)

    labels, n_classes = get_labels(args.pred)

    # Get Dataset
    if args.data_type == "MMWHS":
        dataset_type = MMWHS
    elif args.data_type == "CHAOS":
        dataset_type = CHAOS
    else:
        raise ValueError(f"Data type {args.data_type} not supported")
    
    # K-fold cross-validation
    k_fold(args, dir_checkpoint, dataset_type, labels, device, n_classes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Unet model on the MM-WHS dataset')

    # Other hyperparameters
    parser.add_argument('--data_dir', default='../data/MMWHS/CT_withGT_proc', type=str,
                        help='Directory where to look for the data. For jobs on Lisa, this should be $TMPDIR.')
    
    parser.add_argument('--data_dir_test', default='../data/MMWHS/CT_withGT_proc', type=str,
                        help='Directory where to look for the data. For jobs on Lisa, this should be $TMPDIR.')

    parser.add_argument('--epochs', default=200, type=int, help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int, help='Seed to use for reproducing results')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--bs', default=4, type=int,help='batch_size')
    parser.add_argument('--data_type', default='MMWHS', type=str, help='Dataset types: MMWHS or CHAOS')
    parser.add_argument('--name', default='trained_on_CT', type=str, help='Name of run') 
    parser.add_argument('--pred', default='MYO', type=str,
                        help='Prediction of which label') # MYO, LV, RV, MYO_RV, MYO_LV_RV, Liver
    
    parser.add_argument('--model', default='UNet', type=str,help='ResUNet or UNet')
    
    parser.add_argument('--drit', action='store_true')
    
     # Add k-folds argument
    parser.add_argument('--k_folds', default=6, type=int, help='Number of folds for K-Fold Cross-Validation')


    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)

    main(args)