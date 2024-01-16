from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from sklearn.model_selection import KFold
import glob
import os

import torch

from monai.transforms import (
    LoadImaged,
    Compose,
    EnsureChannelFirstd,
    SqueezeDimd,
    MapLabelValued,
)

from monai.data import Dataset, list_data_collate

import itertools

class MMWHS_single(pl.LightningDataModule):
    def __init__(self, target, data_dir, batch_size=4, k_folds=6):
        super().__init__()
        self.bs = batch_size
        self.k_folds = k_folds
        self.current_fold = 0
        self.fold_datasets = []
        self.test_dataset = None
        self.target_labels = target
        self.data_dir = data_dir

    def setup(self):
        # Transform
        transforms = Compose(
            [
                LoadImaged(keys=["img", "seg"]),
                EnsureChannelFirstd(keys=["img", "seg"]),
                SqueezeDimd(keys=["img", "seg"], dim=-1),
                MapLabelValued(keys=["seg"], orig_labels=[1, 2, 3, 4, 5 , 6 , 7], target_labels=self.target_labels),
            ]
        )

        all_images = sorted(glob.glob(os.path.join(self.data_dir, "images/case_10*")))
        all_labels = sorted(glob.glob(os.path.join(self.data_dir, "labels/case_10*")))
        train_cases = range(0,18)
    
        # Splitting into folds
        kf = KFold(n_splits=self.k_folds, shuffle=True)
        for train_idx, val_idx in kf.split(train_cases):

            train_images = [glob.glob(all_images[idx]+ "/*.nii.gz") for idx in train_idx]
            train_images = sorted(list(itertools.chain.from_iterable(train_images)))
            train_labels = [glob.glob(all_labels[idx]+ "/*.nii.gz") for idx in train_idx]
            train_labels = sorted(list(itertools.chain.from_iterable(train_labels)))
            train_files = [{"img": img, "seg": seg} for img, seg in zip(train_images, train_labels)]


            val_images = [glob.glob(all_images[idx]+ "/*.nii.gz") for idx in val_idx]
            val_images = sorted(list(itertools.chain.from_iterable(val_images)))
            val_labels = [glob.glob(all_labels[idx]+ "/*.nii.gz") for idx in val_idx]
            val_labels = sorted(list(itertools.chain.from_iterable(val_labels)))
            val_files = [{"img": img, "seg": seg} for img, seg in zip(val_images, val_labels)]

            train_ds = Dataset(data=train_files, transform=transforms)
            val_ds = Dataset(data=val_files, transform=transforms)

            self.fold_datasets.append((train_ds, val_ds))


        # Test dataset remains the same
        test_images = sorted(glob.glob(os.path.join(self.data_dir, "images/case_1019/*.nii.gz")))
        test_images2 = sorted(glob.glob(os.path.join(self.data_dir, "images/case_1020/*.nii.gz")))
        test_images = test_images + test_images2

        test_labels = sorted(glob.glob(os.path.join(self.data_dir, "labels/case_1019/*.nii.gz")))
        test_labels2 = sorted(glob.glob(os.path.join(self.data_dir, "labels/case_1020/*.nii.gz")))
        test_labels = test_labels + test_labels2

        test_files = [{"img": img, "seg": seg} for img, seg in zip(test_images, test_labels)]

        # Create a test data set
        self.test_dataset = Dataset(data=test_files, transform=transforms)
    
    def set_current_fold(self, fold):
        self.current_fold = fold

    def get_fold_dataloaders(self, fold):
        train_subset, val_subset = self.fold_datasets[fold]
        train_loader = DataLoader(train_subset, batch_size=self.bs, shuffle=True, num_workers=4, collate_fn=list_data_collate, pin_memory=torch.cuda.is_available())
        val_loader = DataLoader(val_subset, batch_size=self.bs, num_workers=4, collate_fn=list_data_collate, pin_memory=torch.cuda.is_available())
        return train_loader, val_loader

    def train_dataloader(self):
        train_subset, _ = self.fold_datasets[self.current_fold]
        return DataLoader(train_subset, batch_size=self.bs, num_workers=4, shuffle=True, collate_fn=list_data_collate, pin_memory=torch.cuda.is_available())

    def val_dataloader(self):
        _, val_subset = self.fold_datasets[self.current_fold]
        return DataLoader(val_subset, batch_size=self.bs, num_workers=4, collate_fn=list_data_collate, pin_memory=torch.cuda.is_available())

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.bs, num_workers=4, collate_fn=list_data_collate, pin_memory=torch.cuda.is_available())


class MMWHS_double(pl.LightningDataModule):
    def __init__(self, target, data_dir="../data/preprocessed/", batch_size=4, k_folds=6, mod='CT', FT=False):
        super().__init__()
        self.bs = batch_size
        self.k_folds = k_folds
        self.current_fold = 0
        self.fold_datasets = []
        self.test_dataset = None
        self.target_labels = target
        self.ft = FT
        if mod == 'CT':
            self.data_dir_true = os.path.join(data_dir, 'CT/annotated/')
            self.data_dir_fake = os.path.join(data_dir, 'MRI_fake/annotated/')
            self.data_dir_test = os.path.join(data_dir, 'MRI/annotated/')
        else:   
            self.data_dir_true = os.path.join(data_dir, 'MRI/annotated/')
            self.data_dir_fake = os.path.join(data_dir, 'CT_fake/annotated/')
            self.data_dir_test = os.path.join(data_dir, 'CT/annotated/')

    def setup(self):
        # Transform
        transforms = Compose(
            [
                LoadImaged(keys=["img", "seg"]),
                EnsureChannelFirstd(keys=["img", "seg"]),
                SqueezeDimd(keys=["img", "seg"], dim=-1),
                MapLabelValued(keys=["seg"], orig_labels=[1, 2, 3, 4, 5 , 6 , 7], target_labels=self.target_labels),
            ]
        )

        all_images = sorted(glob.glob(os.path.join(self.data_dir_true, "images/case_10*"))) + sorted(glob.glob(os.path.join(self.data_dir_fake, "images/case_10*")))
        all_labels = sorted(glob.glob(os.path.join(self.data_dir_true, "labels/case_10*"))) + sorted(glob.glob(os.path.join(self.data_dir_fake, "labels/case_10*")))
        
        #TODO: add funetuning aproach
        if self.ft:
            train_cases = [*range(0,18), *range(20, 38)] 
        else:
            # take only the fake images OR 1-18 of the fake images
            train_cases = range(20,38) # range(20, 40)
    
        # Splitting into folds
        kf = KFold(n_splits=self.k_folds, shuffle=True)
        for train_idx, val_idx in kf.split(train_cases):

            train_images = [glob.glob(all_images[idx]+ "/*.nii.gz") for idx in train_idx]
            train_images = sorted(list(itertools.chain.from_iterable(train_images)))
            train_labels = [glob.glob(all_labels[idx]+ "/*.nii.gz") for idx in train_idx]
            train_labels = sorted(list(itertools.chain.from_iterable(train_labels)))
            train_files = [{"img": img, "seg": seg} for img, seg in zip(train_images, train_labels)]


            val_images = [glob.glob(all_images[idx]+ "/*.nii.gz") for idx in val_idx]
            val_images = sorted(list(itertools.chain.from_iterable(val_images)))
            val_labels = [glob.glob(all_labels[idx]+ "/*.nii.gz") for idx in val_idx]
            val_labels = sorted(list(itertools.chain.from_iterable(val_labels)))
            val_files = [{"img": img, "seg": seg} for img, seg in zip(val_images, val_labels)]

            train_ds = Dataset(data=train_files, transform=transforms)
            val_ds = Dataset(data=val_files, transform=transforms)

            self.fold_datasets.append((train_ds, val_ds))


        # Test dataset remains the same
        test_images_fake = sorted(glob.glob(os.path.join(self.data_dir_fake, "images/case_1019/*.nii.gz"))) + sorted(glob.glob(os.path.join(self.data_dir_fake, "images/case_1020/*.nii.gz")))
        test_labels_fake = sorted(glob.glob(os.path.join(self.data_dir_fake, "labels/case_1019/*.nii.gz"))) + sorted(glob.glob(os.path.join(self.data_dir_fake, "labels/case_1020/*.nii.gz")))
       

        test_files = [{"img": img, "seg": seg} for img, seg in zip(test_images_fake, test_labels_fake)]

        # Create a test data set
        self.test_dataset = Dataset(data=test_files, transform=transforms)
    
    def set_current_fold(self, fold):
        self.current_fold = fold

    def get_fold_dataloaders(self, fold):
        train_subset, val_subset = self.fold_datasets[fold]
        train_loader = DataLoader(train_subset, batch_size=self.bs, shuffle=True, num_workers=4, collate_fn=list_data_collate, pin_memory=torch.cuda.is_available())
        val_loader = DataLoader(val_subset, batch_size=self.bs, num_workers=4, collate_fn=list_data_collate, pin_memory=torch.cuda.is_available())
        return train_loader, val_loader

    def train_dataloader(self):
        train_subset, _ = self.fold_datasets[self.current_fold]
        return DataLoader(train_subset, batch_size=self.bs, num_workers=4, shuffle=True, collate_fn=list_data_collate, pin_memory=torch.cuda.is_available())

    def val_dataloader(self):
        _, val_subset = self.fold_datasets[self.current_fold]
        return DataLoader(val_subset, batch_size=self.bs, num_workers=4, collate_fn=list_data_collate, pin_memory=torch.cuda.is_available())

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.bs, num_workers=4, collate_fn=list_data_collate, pin_memory=torch.cuda.is_available())

