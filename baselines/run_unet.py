import os
import torch
import argparse
import sys
import glob

from unet_pl import UNetL

import pytorch_lightning as pl
from monai.transforms import (
    LoadImaged,
    Compose,
    EnsureChannelFirstd,
    SqueezeDimd,
    MapLabelValued,
)

from data import MMWHS_single # , MMWHS_double

from sklearn.model_selection import KFold 

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

def train(args, dir_checkpoint, model_class, dataset, k_folds, device):
    test_results = []
    test_loader = dataset.test_dataloader()
    for fold in range(k_folds):
        print(f"Training fold {fold}")
        logger = TensorBoardLogger(dir_checkpoint, name=f'fold_{fold}')
        dataset.set_current_fold(fold)
        trainer = pl.Trainer(default_root_dir=dir_checkpoint, max_epochs=args.epochs, accelerator="gpu" if str(device).startswith("cuda") else "cpu", devices=1, logger=logger,
                            callbacks=ModelCheckpoint(
                            save_weights_only=True,
                            dirpath=dir_checkpoint,
                            filename = "{fold}_{epoch}-{Validation Mean Dice:.4f}",
                            monitor="Validation Mean Dice",
                            mode="max",
                            save_top_k=1,
                        ))
        model = model_class(args.bs, args.epochs, args.loss, args.lr, args.modality, args.pred)

        train_loader, val_loader = dataset.get_fold_dataloaders(fold)
        
        model.train_dataloader = lambda: train_loader
        model.val_dataloader = lambda: val_loader
        model.test_dataloader = lambda: test_loader

        trainer.fit(model)

        # run test set
        result = trainer.test(model)
        test_results.append(result[0]["Test Mean Dice"])
        # for now only one fold
        # break

    print("Test results:")
    print(test_results)
    final_res = sum(test_results)/len(test_results)
    print("Final test result:")
    print(final_res)


def test(trainer, models, model_class, dataset, k):
    test_results = []

    for pretrained_model in models:
        print(f"Testing model {pretrained_model}")
        model = model_class.load_from_checkpoint(pretrained_model) 
        test_loader = dataset.test_dataloader() 
        model.test_dataloader = lambda: test_loader
        result = trainer.test(model, dataloaders=test_loader)
        print("hier?")
        test_results.append(result[0]["Test Mean Dice"])

    print("Test results:")
    print(test_results)
    final_res = sum(test_results)/len(test_results)
    print("Final test result:")
    print(final_res)

def main(args):

    pl.seed_everything(args.seed)

    filename = f'Name_{args.name}_LR_{args.lr}_BS_{args.bs}_modality_{args.modality}_epochs_{args.epochs}_label_{args.pred}_loss_{args.loss}_model_{args.model}_folds_{args.k_folds}'    
    # filename = f'LR_{args.lr}_BS_{args.bs}_modality_{args.modality}_epochs_{args.epochs}_label_{args.pred}_loss_{args.loss}_model_{args.model}'    
    
    dir_checkpoint = os.path.join('checkpoints/', filename)
    os.makedirs(dir_checkpoint, exist_ok=True)
    logger = TensorBoardLogger('checkpoints/', name=filename)

    if args.pred == "MYO":
        labels = [1, 0, 0, 0, 0, 0, 0]
    else:
        labels = [1, 1, 1, 1, 1, 1, 1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.model == 'unet':
        model_class = UNetL
        dataset = MMWHS_single(target = labels, data_dir = args.data_dir, batch_size=args.bs, k_folds=args.k_folds, test_data_dir=args.test_data_dir)
        dataset.setup()
    else:
        print("Model not implemented")


    # K-Fold Cross-Validation
    if args.mode == "train":
        train(args, dir_checkpoint, model_class, dataset, args.k_folds, device)
    
    elif args.mode == "test":
        # Check whether pretrained model exists. If yes, load it and skip training
        pretrained_filenames = glob.glob(os.path.join(dir_checkpoint, "*.ckpt"))
        
        # TO DO: adjust for k-fold cross-validation
        if os.path.isfile(pretrained_filenames[0]):
            print(f"Found pretrained model, loading...")
            trainer = pl.Trainer(default_root_dir=dir_checkpoint, max_epochs=args.epochs, accelerator="gpu" if str(device).startswith("cuda") else "cpu", devices=1, logger=logger,
                            callbacks=ModelCheckpoint(
                            save_weights_only=True,
                            dirpath=dir_checkpoint,
                            filename = "{epoch}-{Validation Mean Dice:.4f}",
                            monitor="Validation Mean Dice",
                            mode="max",
                            save_top_k=1,
                        ))
            test(trainer, pretrained_filenames, model_class, dataset, args.k_folds)
        else:
            print("No pretrained model found, training from scratch...")
            train(args, dir_checkpoint, model_class, dataset, args.k_folds, device, logger)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Unet model on the MM-WHS dataset')

    # Other hyperparameters
    parser.add_argument('--data_dir', default='../data/other/CT_withGT_proc/annotated', type=str,
                        help='Directory where to look for the data. For jobs on Lisa, this should be $TMPDIR.')
    parser.add_argument('--test_data_dir', default='../data/other/CT_withGT_proc/annotated', type=str,
                        help='Directory where to look for the test data.')
    
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate')
    parser.add_argument('--bs', default=4, type=int,
                        help='batch_size')
    parser.add_argument('--modality', default="CT", type=str,
                        help='Learning rate')
    parser.add_argument('--model', default='unet', type=str,
                    help='Baseline used') # other option is drit_unet
    parser.add_argument('--name', default='trial', type=str,
                    help='Baseline used') # other option is drit_unet

    parser.add_argument('--loss', default="BCE", type=str,
                        help='Loss used during training') # BCE, Dice or DiceBCE
    
    parser.add_argument('--pred', default='MYO', type=str,
                        help='Prediction of which label') # MYO, LV, RV, MYO_RV, MYO_LV_RV

    parser.add_argument('--mode', default='train', type=str,
                        help='train or test')
    
     # Add k-folds argument
    parser.add_argument('--k_folds', default=6, type=int, help='Number of folds for K-Fold Cross-Validation')


    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)

    main(args)