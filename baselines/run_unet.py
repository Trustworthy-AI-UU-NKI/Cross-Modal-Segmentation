import os
import torch
import argparse
import sys
import glob
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from data import MMWHS_single, Retinal_Vessel_single
import torch.optim as optim
import numpy as np

from monai.networks.nets import UNet
from sklearn.model_selection import KFold 
from monai.losses import DiceLoss
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from helpers import *

def train(args, dir_checkpoint_fold, device, n_classes, train_loader, val_loader, in_channels=1):
    
    if args.model == "ResUnet":
        model = UNet(
            spatial_dims=2,
            in_channels=in_channels,
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
    dice_loss = DiceLoss(include_background=False, softmax=True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    writer = SummaryWriter(log_dir=dir_checkpoint_fold)

    for epoch in range(args.epochs):
        model.train()
        for image, label in train_loader:
            image = image.to(device)
            label = label.to(device)

            outputs = model(image)
            labels_oh = F.one_hot(label.long().squeeze(1), n_classes).permute(0, 3, 1, 2)
            loss = dice_loss(outputs, labels_oh)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('Loss/train', loss, global_step)
            global_step += 1

        model.eval()
        dice_tot = 0
        dice_classes_tot = np.zeros(n_classes)
        with torch.no_grad():
            for image, label in val_loader:
                image = image.to(device)
                label = label.to(device)

                outputs = model(image)
                dice_classes = dice(label, outputs, n_classes)
                dice_classes_tot += dice_classes
                dice_tot += np.mean(dice_classes[1:])

            dice_tot /= len(val_loader)
            dice_classes_tot /= len(val_loader)

            writer.add_scalar('Dice/val', dice_tot, epoch)
            for item in range(n_classes):
                writer.add_scalar(f'Dice/val_{item}', dice_classes_tot[item], epoch)

        if dice_tot > best_dice:
            best_dice = dice_tot
            existing_model_files = glob.glob(f"{dir_checkpoint_fold}/*.pth")
            # Delete each found model file
            for file_path in existing_model_files:
                try:
                    os.remove(file_path)
                    print(f"Deleted old model file: {file_path}")
                except OSError as e:
                    print(f"Error deleting file {file_path}: {e}")
                    
            torch.save(model.state_dict(), os.path.join(dir_checkpoint_fold, f'best_model_{epoch}.pth'))
            print(f"Saved best model with dice {dice_tot} at epoch {epoch}")


def test(model_file, test_loader, n_classes, in_channels, device, model_type):

    if model_type == "ResUnet":
        model = UNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=n_classes,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
    else:
        model = UNet_model(n_classes)

    #init_weights_norm(model)
    model.to(device)
    print(f"Testing model {model_file}")
    # Load the state dictionary
    model.load_state_dict(torch.load(model_file))
    model.eval()

    dice_tot = 0
    dice_classes_tot = np.zeros(n_classes)
    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)

            outputs = model(image)
            dice_classes = dice(label, outputs, n_classes)
            # dice_tot += dice_classes.item()
            dice_classes_tot += dice_classes
            dice_tot += np.mean(dice_classes)

        dice_tot /= len(test_loader)
        dice_classes_tot /= len(test_loader)
    

    return dice_tot, dice_classes_tot


def main(args):

    pl.seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    filename = f'{args.name}_{args.pred}'    
    dir_checkpoint = os.path.join('checkpoints/', filename)
    os.makedirs(dir_checkpoint, exist_ok=True)

    labels, n_classes = get_labels(args.pred)

    # MMWHS Dataset
    if args.data_type == "MMWHS":
        dataset_type = MMWHS_single
        in_channels = 1
    elif args.data_type == "RetinalVessel":
        dataset_type = Retinal_Vessel_single
        in_channels = 3
    else:
        raise ValueError(f"Data type {args.data_type} not supported")
    
    cases = range(0,20)
    kf = KFold(n_splits=args.k_folds, shuffle=True)
    fold = 0
    test_dice_tot, test_dice_classes = [], []

    for fold_train, fold_test in kf.split(cases):
        print(f"Fold {fold}")
        print(f"TRAINING ON CASES: ", fold_train)
        dir_checkpoint_fold = os.path.join(dir_checkpoint, f'fold_{fold}')            
        os.makedirs(dir_checkpoint_fold, exist_ok=True)

        train_val = np.split(np.array(fold_train), [13, 3])
        fold_train = list(train_val[0])
        fold_val = list(train_val[1])
        dataset_train = dataset_type(args, fold_train, labels)
        train_loader = DataLoader(dataset_train, batch_size=args.bs, shuffle=True, num_workers=4)
        dataset_val = dataset_type(args, fold_val, labels) 
        val_loader = DataLoader(dataset_val, batch_size=args.bs, num_workers=4)

        train(args, dir_checkpoint_fold, device, n_classes, train_loader, val_loader, in_channels)

        dataset_test = dataset_type(args, fold_test, labels, train=False) 
        test_loader = DataLoader(dataset_test, batch_size=args.bs, num_workers=4)
        pretrained_filename = glob.glob(os.path.join(dir_checkpoint_fold, "*.pth"))

        dice_tot, dice_classes_tot = test(pretrained_filename[0], test_loader, n_classes, in_channels, device, args.model)
        fold += 1
        test_dice_tot.append(dice_tot)
        test_dice_classes.append(dice_classes_tot)
    

    test_dice_tot = np.array(test_dice_tot)
    test_dice_classes = np.array(test_dice_classes)

    print("Test results total: ", test_dice_tot)
    print(f"Mean test dice total: {np.mean(test_dice_tot)}")
    print(f"Std test dice total: {np.std(test_dice_tot)}")

    for item in range(n_classes):
        print(f"Test results {item}: ", test_dice_classes[:, item])
        print(f"Mean test dice {item}: {np.mean(test_dice_classes[:, item])}")
        print(f"Std test dice {item}: {np.std(test_dice_classes[:, item])}")

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Unet model on the MM-WHS dataset')

    # Other hyperparameters
    parser.add_argument('--data_dir', default='../data/other/CT_withGT_proc/annotated', type=str,
                        help='Directory where to look for the data. For jobs on Lisa, this should be $TMPDIR.')

    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate')
    
    parser.add_argument('--bs', default=4, type=int,
                        help='batch_size')
    
    parser.add_argument('--data_type', default='MMWHS', type=str,
                    help='Baseline used') 
    
    parser.add_argument('--name', default='trial', type=str,
                    help='Baseline used') 
    
    parser.add_argument('--pred', default='MYO', type=str,
                        help='Prediction of which label') # MYO, LV, RV, MYO_RV, MYO_LV_RV

    parser.add_argument('--mode', default='train', type=str,
                        help='train or test')
    

    parser.add_argument('--model', default='ResUnet', type=str,
                        help='ResUnet or Unet')
    
     # Add k-folds argument
    parser.add_argument('--k_folds', default=6, type=int, help='Number of folds for K-Fold Cross-Validation')


    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)

    main(args)