import os
import torch
from tqdm import tqdm
import glob
import argparse
import sys

from unet import UNetL
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from monai.data import Dataset, list_data_collate
from monai.transforms import (
    LoadImaged,
    Compose,
    EnsureChannelFirstd,
    SqueezeDimd,
    MapLabelValued,
)

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

def train(trainer, model, transforms, bs):
    train_images = sorted(glob.glob(os.path.join(args.data_dir, "train/images/*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(args.data_dir, "train/labels/*.nii.gz")))
    train_files = [{"img": img, "seg": seg} for img, seg in zip(train_images, train_labels)]

    val_images = sorted(glob.glob(os.path.join(args.data_dir, "val/images/*.nii.gz")))
    val_labels = sorted(glob.glob(os.path.join(args.data_dir, "val/labels/*.nii.gz")))
    val_files = [{"img": img, "seg": seg} for img, seg in zip(val_images, val_labels)]

     # Create a training data loader
    train_ds = Dataset(data=train_files, transform=transforms)
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=4, collate_fn=list_data_collate, pin_memory=torch.cuda.is_available())

    # Create a validation data loader
    val_ds = Dataset(data=val_files, transform=transforms)
    val_loader = DataLoader(val_ds, batch_size=bs, num_workers=4, collate_fn=list_data_collate, pin_memory=torch.cuda.is_available())

    model.train_dataloader = lambda: train_loader
    model.val_dataloader = lambda: val_loader

    trainer.fit(model)


def test(trainer, model, transforms,bs):
    test_images = sorted(glob.glob(os.path.join(args.data_dir, "test/images/*.nii.gz")))
    test_labels = sorted(glob.glob(os.path.join(args.data_dir, "test/labels/*.nii.gz")))
    test_files = [{"img": img, "seg": seg} for img, seg in zip(test_images, test_labels)]

    # Create a test data loader
    test_ds = Dataset(data=test_files, transform=transforms)
    test_loader = DataLoader(test_ds, batch_size=bs, num_workers=4, collate_fn=list_data_collate, pin_memory=torch.cuda.is_available())

    model.test_dataloader = lambda: test_loader

    trainer.test(model)


def main(args):

    pl.seed_everything(args.seed)

    filename = f'LR_{args.lr}_BS_{args.bs}_modality_{args.modality}_epochs_{args.epochs}_label_{args.pred}_loss_{args.loss}_model_{args.model}'    
    dir_checkpoint = os.path.join('checkpoints/', filename)
    os.makedirs(dir_checkpoint, exist_ok=True)
    logger = TensorBoardLogger('checkpoints/', name=filename)

    if args.model == 'unet':
        model_class = UNetL
        model = model_class(args.bs, args.epochs, args.loss, args.lr, args.modality, args.pred)
    else:
        print("Model not implemented")

    if args.pred == "MYO":
        labels = [1, 0, 0, 0, 0, 0, 0]
    else:
        labels = [1, 1, 1, 1, 1, 1, 1]

    transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            SqueezeDimd(keys=["img", "seg"], dim=-1),
            MapLabelValued(keys=["seg"], orig_labels=[1, 2, 3, 4, 5 , 6 , 7], target_labels=labels),
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = pl.Trainer(default_root_dir=dir_checkpoint, max_epochs=args.epochs, accelerator="gpu" if str(device).startswith("cuda") else "cpu", devices=1, logger=logger,
                            callbacks=ModelCheckpoint(
                            save_weights_only=True,
                            dirpath=dir_checkpoint,
                            filename = "{epoch}-{Validation Mean Dice:.4f}",
                            monitor="Validation Mean Dice",
                            mode="max",
                            save_top_k=1,
                        ))

    if args.mode == "train":
        train(trainer, model, transforms, args.bs)
    elif args.mode == "test":
        # Check whether pretrained model exists. If yes, load it and skip training

        pretrained_filename = glob.glob(os.path.join(dir_checkpoint, "*.ckpt"))[0]
        if os.path.isfile(pretrained_filename):
            print(f"Found pretrained model at {pretrained_filename}, loading...")
            model = model_class.load_from_checkpoint(pretrained_filename) 
            test(trainer, model, transforms, args.bs)
        else:
            print("No pretrained model found, training from scratch...")
            # train(trainer, model, transforms, args.bs)
            # test(trainer, model, transforms, args.bs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Unet model on the MM-WHS dataset')

    # Other hyperparameters
    parser.add_argument('--data_dir', default='../data/preprocessed/CT', type=str,
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
    parser.add_argument('--model', default='unet', type=str,
                    help='Baseline used')

    parser.add_argument('--loss', default="BCE", type=str,
                        help='Loss used during training')
    
    parser.add_argument('--pred', default='MYO', type=str,
                        help='Prediction of which label')

    parser.add_argument('--mode', default='train', type=str,
                        help='train or test')

    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)

    main(args)