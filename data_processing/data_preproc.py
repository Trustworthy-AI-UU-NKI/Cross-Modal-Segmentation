# imports 
import numpy as np
import os
import monai
from monai.data import CacheDataset, DataLoader, GridPatchDataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Resized,
    ScaleIntensityd,
    SqueezeDimd,
)

import matplotlib.pyplot as plt
import glob
import torch
import argparse
import sys



def create_data(data_dir, modality, output_folder):
    if modality == "CT":
        data_dir = os.path.join(data_dir, "ct_train")
        output_folder_train = os.path.join(output_folder, "ct_train")
        output_folder_val = os.path.join(output_folder, "ct_val")
        output_folder_test = os.path.join(output_folder, "ct_test")

    elif modality == "MRI":
        data_dir = os.path.join(data_dir, "mr_train")
        output_folder_train = os.path.join(output_folder, "mr_train")
        output_folder_val = os.path.join(output_folder, "mr_val")
        output_folder_test = os.path.join(output_folder, "mr_val")

    # Get al the files
    images = sorted(glob.glob(os.path.join(data_dir, "ct_train_?_image.nii.gz")))
    labels = sorted(glob.glob(os.path.join(data_dir, "ct_train_?_label.nii.gz")))
    dataset_files = [{"img": img, "seg": seg} for img, seg in zip(images, labels)]  

    volume_transforms = Compose(
    [
        LoadImaged(keys=["img", "seg"]),
        EnsureChannelFirstd(keys=["img", "seg"]),
        ScaleIntensityd(keys="img"), # normalization between 0 and 1 # segmentation masl also?
        EnsureTyped(keys=["img", "seg"]),
    ])  
    
    # TO DO: Experiment with last two parameters
    volume_ds = CacheDataset(data=dataset_files, transform=volume_transforms, cache_rate=1.0, num_workers=4)
    
    # Create patches (i.e. images)
    patch_func = monai.data.PatchIterd(
        keys=["img", "seg"],
        patch_size=(None, None, 1),  # dynamic first two dimensions
        start_pos=(0, 0, 0)
    )

    patch_transform = Compose(
        [
            SqueezeDimd(keys=["img", "seg"], dim=-1),  # squeeze the last dim
            Resized(keys=["img", "seg"], spatial_size=[224, 224]), # resize the spatial size
        ]
    )

example_patch_ds = GridPatchDataset(data=volume_ds_person1, patch_iter=patch_func, transform=patch_transform)






if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create data for cardiac segmentation of the MM-WHS dataset')

    parser.add_argument('--data_dir', type=str, default="../MMWHS_Dataset/",
                        help='Path to the scene object')
    # CT or MRI
    parser.add_argument('--modality', type=str, default="CT",
                        help='name of the scene') 
    parser.add_argument('--output_folder', type=str, default="processed/",
                        help='Path to the store directory.')

    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)

    create_data(args.data_dir, args.modality, args.output_folder)


       