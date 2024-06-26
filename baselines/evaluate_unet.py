import os
import torch
import argparse
import sys
import glob
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from dataloaders import MMWHS, CHAOS

import numpy as np

from monai.networks.nets import UNet
from sklearn.model_selection import KFold 
from helpers import *
import numpy as np
from UNet import UNet_model

def test(model_file, test_loader, n_classes, device, model_type):

    # Get UNet or ResUNet
    if model_type == "ResUNet":
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

    # Load trained model
    model.to(device)
    print(f"Testing model {model_file}")
    model.load_state_dict(torch.load(model_file))
    model.eval()

    dice_classes_tot = np.zeros(n_classes)
    assd_classes_tot = np.zeros(n_classes)

    # Skips images with only background
    true_dice_class = 0 
    assd_len = len(test_loader)
    dsc_len = len(test_loader)
    
    with torch.no_grad():
        
        for it, (image, label) in enumerate(test_loader):
            image = image.to(device)
            label = label.to(device)

            # Forward pass for testing
            outputs = model(image)

            # COmpute test metrics
            dice_classes = dice(label, outputs, n_classes)
            assd_classes = assd(label, outputs, n_classes, pixdim=image.meta["pixdim"][1])

            # skip when either label or pred is 0
            if not np.isinf(assd_classes).any():
                assd_classes_tot += assd_classes
            else:
                assd_len -= 1
            
            # Skip disce class when no label is there. Background is always there
            if torch.all(label==0):
                dsc_len -= 1
            else:
                true_dice_class += dice_classes[1]

            dice_classes_tot += dice_classes

        dice_classes_tot /= len(test_loader)
        assd_classes_tot /= assd_len
        true_dice_class /= dsc_len
    
    return dice_classes_tot, assd_classes_tot, true_dice_class


def main(args):
    pl.seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    filename = f'{args.model}_{args.name}_{args.pred}'  
    dir_checkpoint = os.path.join('checkpoints/', filename)
    os.makedirs(dir_checkpoint, exist_ok=True)

    labels, n_classes = get_labels(args.pred)

    # Get Dataset
    if args.data_type == "MMWHS":
        dataset_type = MMWHS
    elif args.data_type == "CHAOS":
        dataset_type = CHAOS
    else:
        raise ValueError(f"Data type {args.data_type} not supported")
    
    cases = range(0,20)
    kf = KFold(n_splits=args.k_folds, shuffle=True)
    fold = 0
    test_dice_classes, test_assd_classes, test_true_dice_class = [], [], []

    # Test each model for all folds
    for fold_train, fold_test in kf.split(cases):

        print(f"Fold {fold}")
        dir_checkpoint_fold = os.path.join(dir_checkpoint, f'fold_{fold}')            
        os.makedirs(dir_checkpoint_fold, exist_ok=True)

        # Get dataset and pretrained file name
        dataset_test = dataset_type(args.data_dir_test, fold_test, labels) 
        test_loader = DataLoader(dataset_test, batch_size=args.bs, num_workers=4)
        pretrained_filename = glob.glob(os.path.join(dir_checkpoint_fold, "*.pth"))

        # Test
        dice_classes, assd_classes, true_dice_class = test(pretrained_filename[0], test_loader, n_classes, device, args.model)
        test_dice_classes.append(dice_classes)
        test_assd_classes.append(assd_classes)
        test_true_dice_class.append(true_dice_class)
        fold += 1
       
    test_dice_classes = np.array(test_dice_classes)
    test_assd_classes = np.array(test_assd_classes)
    test_true_dice_class = np.array(test_true_dice_class)
    print(test_dice_classes.shape)
    print(test_assd_classes.shape)
    print(test_true_dice_class.shape)

    for item in range(n_classes):
        print(f"DSC RESULTS")
        print(f"All test results {item}: ", test_dice_classes[:, item])
        print(f"Mean test dice {item}: {np.mean(test_dice_classes[:, item])}")
        print(f"Std test dice {item}: {np.std(test_dice_classes[:, item])}")

    
    # ASSD is not computed for the background
    for item in range(1, n_classes):
        print(f"ASSD RESULTS")
        print(f"All test results {item}: ", test_assd_classes[:, item])
        print(f"Mean test dice {item}: {np.mean(test_assd_classes[:, item])}")
        print(f"Std test dice {item}: {np.std(test_assd_classes[:, item])}")
    
    # DSC with correction for when label is all zero
    print(f"True DSC RESULTS")
    print(f"All test results: ", test_true_dice_class)
    print(f"Mean test dice: {np.mean(test_true_dice_class)}")
    print(f"Std test dice: {np.std(test_true_dice_class)}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Unet model on the MM-WHS dataset')

    # Other hyperparameters
    parser.add_argument('--data_dir_test', default='../data/other/CT_withGT_proc/annotated', type=str,
                        help='Directory where to look for the data. For jobs on Lisa, this should be $TMPDIR.')
    parser.add_argument('--seed', default=42, type=int, help='Seed to use for reproducing results')
    
    parser.add_argument('--bs', default=1, type=int, help='batch_size')
    
    parser.add_argument('--data_type', default='MMWHS', type=str, help='Baseline used') 
    
    parser.add_argument('--name', default='trial', type=str, help='Baseline used') 
    
    parser.add_argument('--pred', default='MYO', type=str, help='Prediction of which label') # MYO, LV, RV, MYO_RV, MYO_LV_RV, Liver    

    parser.add_argument('--model', default='UNet', type=str, help='ResUNet or UNet')
    
     # Add k-folds argument
    parser.add_argument('--k_folds', default=5, type=int, help='Number of folds for K-Fold Cross-Validation')

    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)

    main(args)