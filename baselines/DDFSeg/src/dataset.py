from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import Dataset
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

# Data class for annotated source images (mod) and target images
class MMWHS_double(pl.LightningDataModule):
    def __init__(self, target, data_dir, batch_size=4, k_folds=6, mod='CT'):
        super().__init__()
        self.bs = batch_size
        self.k_folds = k_folds
        self.current_fold = 0
        self.fold_datasets = []
        self.test_dataset = None
        self.target_labels = target
        self.annotated_mod = mod

        if mod == 'CT':
            self.data_dir_an = os.path.join(data_dir, 'CT/annotated/')
            self.data_dir_unan = os.path.join(data_dir, 'MRI/unannotated/')
            self.data_dir_test = os.path.join(data_dir, 'MRI/annotated/')
        else:   
            self.data_dir_an = os.path.join(data_dir, 'MRI/annotated/')
            self.data_dir_unan = os.path.join(data_dir, 'CT/unannotated/')
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

        all_images = sorted(glob.glob(os.path.join(self.data_dir_an, "images/case_10*"))) + sorted(glob.glob(os.path.join(self.data_dir_unan, "images/case_10*")))
        all_labels = sorted(glob.glob(os.path.join(self.data_dir_an, "labels/case_10*"))) + sorted(glob.glob(os.path.join(self.data_dir_unan, "labels/case_10*")))
        

        train_cases = [*range(0,18), *range(20, 38)] 

    
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

            train_ds = dataset_double(data=train_files, transform=transforms)
            val_ds = dataset_double(data=val_files, transform=transforms)

            self.fold_datasets.append((train_ds, val_ds))


        # Test dataset remains the same
        test_images = sorted(glob.glob(os.path.join(self.data_dir_test, "images/case_1019/*.nii.gz"))) + sorted(glob.glob(os.path.join(self.data_dir_test, "images/case_1020/*.nii.gz")))
        test_labels = sorted(glob.glob(os.path.join(self.data_dir_test, "labels/case_1019/*.nii.gz"))) + sorted(glob.glob(os.path.join(self.data_dir_test, "labels/case_1020/*.nii.gz")))
       

        test_files = [{"img": img, "seg": seg} for img, seg in zip(test_images, test_labels)]

        # Create a test data set
        self.test_dataset = dataset_double(data=test_files, transform=transforms)
    
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


class dataset_double(Dataset):
  def __init__(self, data, transform):
    # preprocessed/ folder
    self.data = data
    self.transforms = transform
    self.source_im = data["img"]
    self.source_seg = data["seg"]
    self.target_im = data["img_target"]
    # A
    self.A = sorted(glob.glob(os.path.join(self.data_1, "images/case_*/slice_*.nii.gz"))) 
    del self.A[::2]
    # B
    self.B = sorted(glob.glob(os.path.join(self.data_2, "images/case_*/slice_*.nii.gz")))
    del self.B[::2]

    self.A_size = len(self.A)
    self.B_size = len(self.B)
    self.dataset_size = max(self.A_size, self.B_size)



  def __getitem__(self, index):
    if self.dataset_size == self.A_size:
      data_A = self.load_img(self.A[index], self.input_dim_A)
      data_B = self.load_img(self.B[random.randint(0, self.B_size - 1)], self.input_dim_B)
    else:
      data_A = self.load_img(self.A[random.randint(0, self.A_size - 1)], self.input_dim_A)
      data_B = self.load_img(self.B[index], self.input_dim_B)

    data_A = self.transforms(data_A)
    data_B = self.transforms(data_B)
    return data_A, data_B



  def __len__(self):
    return self.dataset_size
