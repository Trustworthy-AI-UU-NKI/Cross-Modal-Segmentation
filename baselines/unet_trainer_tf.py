# imports 
import numpy as np
import os
import monai
from monai.data import Dataset, DataLoader, list_data_collate, decollate_batch
from monai.networks.nets import UNet
from monai.transforms import (
    LoadImaged,
    Compose,
    EnsureChannelFirstd,
    SqueezeDimd,
    MapLabelValued,
    MapLabelValue,
    Activations,
    AsDiscrete,
    ScaleIntensityd,
)
import logging 
import sys
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.visualize import plot_2d_or_3d_image

from tf_data_proc import create_dataset

import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
import argparse

import random
import glob

from monai.losses import DiceLoss
from monai.utils import set_determinism
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm



def evaluate_unet(val_loader, model, device, epoch, dice_metric, best_metric, dir_checkpoint, writer):
    # post transforms?
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    model.eval()
    with torch.no_grad():
        for val_data in val_loader:
            val_images, val_labels = val_data["img"].to(device), val_data["seg"].to(device)
            # SLiding windonw inferer?? sliding_window_inference?
            val_outputs = model(val_images)
            if torch.any((val_labels != 0) & (val_labels != 1)):
                print("val labels are not binary")
            # compute metric for current iteration
            val_outputs_onehot = [post_trans(i) for i in decollate_batch(val_outputs)]
            dice_metric(y_pred=val_outputs_onehot, y=val_labels)

        # aggregate the final mean dice result
        metric = dice_metric.aggregate().item()
        # reset the status for next validation round
        

        if metric > best_metric:
            best_metric = metric
            torch.save(model.state_dict(), os.path.join(dir_checkpoint, "best_metric_model_unet.pth"))
            print(f"saved new best metric model at {epoch}")


        writer.add_scalar("val_mean_dice", metric, epoch + 1)
        plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
        plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label") # val_labels_onehot
        plot_2d_or_3d_image(val_outputs_onehot, epoch + 1, writer, index=0, tag="output")
        dice_metric.reset()

    return best_metric
            

def main(args):
    """
    Function for training and testing a simple no adaption Unet.
    Inputs:
        args - Namespace object from the argument parser
    """
    #set_determinism(seed=args.seed)

    filename = f'LR_{args.lr}_BS_{args.bs}_modality_{args.modality}_epochs_{args.epochs}_tf'
    log_dir = os.path.join(args.log_dir, filename)
    os.makedirs(args.log_dir, exist_ok=True)

    writer = SummaryWriter(comment=filename, log_dir=log_dir)

    logging.info(f'''Starting training Unet:
        Epochs:          {args.epochs}
        Batch size:      {args.bs}
        Learning rate:   {args.lr}
        Modality:        {args.modality}
        Checkpoints at filename:     {filename}
    ''')

    # Example usage
    train_filenames =  sorted(glob.glob(os.path.join(args.data_dir, "ct_train_tfs/*.tfrecords")))

    # Create the dataset
    train_files = create_dataset(train_filenames)

    # Convert the dataset to a MONAI dictionary dataset
    transforms = Compose([  # Add your MONAI transforms here
        # Example: Convert to NumPy array
        lambda data: {'img': np.array(data['img']),
                    'seg': np.array(data['seg'])},
        ScaleIntensityd(keys=['img']),
        MapLabelValued(keys=["seg"], orig_labels=[1, 3], target_labels=[1, 1]),

    ])
    # Create a validation set
    val_filenames = sorted(glob.glob(os.path.join(args.data_dir, "ct_val_tfs/ct_val_slice*.tfrecords")))
    val_files = create_dataset(val_filenames)


    dir_checkpoint = os.path.join('checkpoints/', filename) 
    os.makedirs(dir_checkpoint, exist_ok = True)

    # create a training data loader
    train_ds = Dataset(data=train_files, transform=transforms, num_workers=4, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=4)

    val_ds = Dataset(data=val_files, transform=transforms)
    val_loader = DataLoader(val_ds, batch_size=args.bs, num_workers=4, pin_memory=torch.cuda.is_available())
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    # create UNet, DiceLoss and Adam optimizer
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    loss_function = DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    global_step = 0
    best_metric = -1
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        step = 0
        with tqdm(total=len(train_filenames), desc=f'Epoch {epoch + 1}/{args.epochs}') as pbar:
            for batch in train_loader:
                optimizer.zero_grad()
                inputs, labels = batch["img"].to(device), batch["seg"].to(device)
        
                # Forward pass
                outputs = model(inputs)
                loss = loss_function(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                step += 1

        best_metric = evaluate_unet(val_loader, model, device, epoch, dice_metric, best_metric, dir_checkpoint, writer)
            
        average_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {average_loss}")
        # Log loss to TensorBoard
        writer.add_scalar("Training Loss", average_loss, epoch+1)


    # Close the TensorBoard writer
    writer.close()




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train and evaluate a Unet model on the MM-WHS dataset')


    # Other hyperparameters
    parser.add_argument('--data_dir', default='../PnpAda_release_data/train_val/', type=str,
                        help='Directory where to look for the data. For jobs on Lisa, this should be $TMPDIR.')
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
    parser.add_argument('--log_dir', default='UNET_logs/', type=str,
                        help='Directory where the PyTorch Lightning logs should be created.')

    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)

    main(args)
