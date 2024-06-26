import torch
import os
import argparse
from tqdm import tqdm
import logging
import glob
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold

from data.mmwhs_dataloader import MMWHS
from data.chaos_dataloader import CHAOS
from models.crosscompcsd import CrossCSD
from eval import eval_vmfnet_mm
from utils import *



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type= int, default=200, help='Number of epochs')
    parser.add_argument('--seed', default=42, type=int,help='Seed to use for reproducing results')
    parser.add_argument('--bs', type= int, default=4, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default='0.0001', help='The learning rate for model training')
    parser.add_argument('--weight_init', type=str, default="xavier", help='Weight initialization method')
    parser.add_argument('--k2', type=int,  default=10, help='Check decay learning')
    parser.add_argument('--vc_num', type=int,  default=10, help='Kernel/distributions amount')
    parser.add_argument('--k_folds', type= int, default=5, help='Cross validation')
    parser.add_argument('--pred', type=str, default='MYO', help='Segmentation task')


    parser.add_argument('--data_dir_s',  type=str, default='../data/other/MR_withGT_proc/', help='The name of the source data dir.')
    parser.add_argument('--data_dir_t',  type=str, default='../data/other/CT_withGT_proc/', help='The name of the target data dir.')
    parser.add_argument('--cp', type=str, default='checkpoints/', help='The name of the checkpoints.')
    parser.add_argument('--name', type=str, default='test', help='The name of this run.')
    parser.add_argument('--data_type', type=str, default="MMWHS or CHAOS")
    
    return parser.parse_args()

# Train procedure per fold
def train_net(train_loader, val_loader, fold, device, args, len_train_data, num_classes, save_dir, writer):
    # Get model
    model = CrossCSD(args, device, 1, num_classes, vMF_kappa=30, fold_nr=fold)
    model.to(device)

    best_score = 0
    global_step = 0
    print("Training fold: ", fold)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        with tqdm(total=len_train_data, desc=f'Epoch {epoch + 1}/{args.epochs}', unit='img') as pbar:
            model.train()
            for img_s, label_s, img_t, label_t in train_loader:
                img_s = img_s.to(device)
                img_t = img_t.to(device)
                label_s = label_s.to(device)

                # We only use the source labels for training
                losses = model.update(img_s, label_s, img_t)
                pbar.set_postfix(**{'loss (batch)': losses["loss/total"]})
                pbar.update(img_t.shape[0])

                for key, value in losses.items():
                    writer.add_scalar(f'{key}', value, global_step)

                global_step += 1
    
            # Validation
            if epoch % 5 == 0:
                # Validation is also only done on the source labels
                metrics_dict, images_dict, visuals_dict, lpips_metric = eval_vmfnet_mm(model, val_loader, device)

                # Compute validation metric
                new_metric = (1-lpips_metric) + metrics_dict["Target/DSC_fake"]

                # Tensorboard logging
                writer.add_scalar(f'Val_metrics/lpips_dscf', new_metric , epoch)
                writer.add_scalar(f'Val_metrics/lpips_target', lpips_metric, epoch)
                for key, value in metrics_dict.items():
                    writer.add_scalar(f'Val_metrics/{key}', value, epoch)
                
                if epoch % 20 == 0:
                    for key, value in images_dict.items():
                        writer.add_images(f'Val_images/{key}', value, epoch, dataformats='NCHW')
                    
                    for key, value in visuals_dict.items():
                        for i in range(args.vc_num):
                            writer.add_images(f'Val_visuals/{key}_{i+1}', value[:,i,:,:].unsqueeze(1), epoch, dataformats='NCHW')
                
                # Learning rate scheduler
                model.scheduler_source.step(new_metric)
                model.scheduler_target.step(new_metric)

                # Save best model during training based on validation score
                if new_metric > best_score:
                    best_score = new_metric
                    print("Epoch checkpoint")
                    
                    print(f"--- Remove old model before saving new one ---")
                    existing_model_files = glob.glob(f"{save_dir}/*.pth")
                    for file_path in existing_model_files:
                        try:
                            os.remove(file_path)
                            print(f"Deleted old model file: {file_path}")
                        except OSError as e:
                            print(f"Error deleting file {file_path}: {e}")
                    
                    torch.save(model.state_dict(), os.path.join(save_dir, f'CP_epoch_{epoch}_model_{best_score}.pth'))

                    logging.info('Checkpoint saved !')
                    
            
    writer.close()


def train_k_folds(args, labels, num_classes, device, dataset_type):
    cases = range(0,20)
    kf = KFold(n_splits=args.k_folds, shuffle=True)
    fold = 0

    # Train and validate model on all folds
    for fold_train_val, fold_val in kf.split(cases):
        print("Training fold: ", fold)
        dir_checkpoint = os.path.join(args.cp, args.name)
        save_dir = os.path.join(dir_checkpoint, f'fold_{fold}')
        os.makedirs(save_dir, exist_ok=True)
        log_dir = os.path.join('logs', os.path.join(args.name, 'fold_{}'.format(fold)))
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        
        print("Loading train data")
        dataset_train = dataset_type(args, labels, fold_train_val)
        train_loader = DataLoader(dataset_train, batch_size=args.bs, shuffle=True, num_workers=4)
        print("Loading val data")
        dataset_val = dataset_type(args, labels, fold_val) 
        val_loader = DataLoader(dataset_val, batch_size=1, num_workers=4)
        len_train_data = dataset_train.__len__()

        train_net(train_loader, val_loader, fold, device, args, len_train_data, num_classes, save_dir, writer) 
        fold += 1

    print("Training complete!")

        
def main(args):
    set_seed(args.seed)
    labels, num_classes = get_labels(args.pred)

    # Get Dataset
    if args.data_type == "MMWHS":
        dataset_type = MMWHS
    elif args.data_type == "chaos":
        dataset_type = CHAOS
    else:
        raise ValueError(f"Data type {args.data_type} not supported")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # cross-validation
    train_k_folds(args, labels, num_classes, device, dataset_type)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    main(args)
