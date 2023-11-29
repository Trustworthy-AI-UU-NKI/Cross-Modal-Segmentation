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
    MapLabelValued,
    Flipd,
    Rotated,
)

import matplotlib.pyplot as plt
import glob
import torch
import argparse
import sys
import SimpleITK as sitk
import random

def create_ct_data(ds, output_dir_image, output_dir_label):   
    # Create patches (i.e. images)
    patch_func = monai.data.PatchIterd(
        keys=["img", "seg"],
        patch_size=(None, None, 1),  # dynamic first two dimensions
        start_pos=(0, 0, 0)
    )

    patch_transform = Compose(
        [
            SqueezeDimd(keys=["img", "seg"], dim=-1),  # squeeze the last dim
            #Rotated(keys=["img", "seg"], angle=math.radians(180)),
            Flipd(keys=["img", "seg"], spatial_axis=1),
            MapLabelValued(keys=["seg"], orig_labels=[205, 420, 500, 550, 600, 820, 850], target_labels=[1, 2, 3, 4, 5 , 6 , 7]),
        ]
    )
    

    example_patch_ds = GridPatchDataset(data=ds, patch_iter=patch_func, transform=patch_transform)
    patch_data_loader = DataLoader(example_patch_ds, batch_size=1, num_workers=2, pin_memory=torch.cuda.is_available())

    i = 0
    for batch in patch_data_loader:
        image, label = batch[0]["img"], batch[0]["seg"]
        if torch.any(label.data != 0).item():
            # Convert the torch tensor to a SimpleITK image

            slice_image = sitk.GetImageFromArray(image.squeeze(0))
            slice_label = sitk.GetImageFromArray(label.squeeze(0))
            # # Save the 2D slice as a NIfTI file
            sitk.WriteImage(slice_image, os.path.join(output_dir_image, f"slice_{i}.nii.gz"))
            sitk.WriteImage(slice_label, os.path.join(output_dir_label, f"slice_{i}.nii.gz"))
            i += 1

def create_mr_data(ds, output_dir_image, output_dir_label):
    # Create patches (i.e. images)
    patch_func = monai.data.PatchIterd(
        keys=["img", "seg"],
        patch_size=(None, 1, None),  # dynamic first two dimensions
        start_pos=(0, 0, 0)
    )

    patch_transform = Compose(
        [
            SqueezeDimd(keys=["img", "seg"], dim=0),  # squeeze the last dims
            MapLabelValued(keys=["seg"], orig_labels=[205, 420, 500, 550, 600, 820, 850], target_labels=[1, 2, 3, 4, 5 , 6 , 7]),
        ]
    )
    

    example_patch_ds = GridPatchDataset(data=ds, patch_iter=patch_func, transform=patch_transform)
    patch_data_loader = DataLoader(example_patch_ds, batch_size=1, num_workers=2, pin_memory=torch.cuda.is_available())

    i = 0
    for batch in patch_data_loader:
        image, label = batch[0]["img"], batch[0]["seg"]
        if torch.any(label.data != 0).item():
            # Convert the torch tensor to a SimpleITK image

            slice_image = sitk.GetImageFromArray(image.squeeze(2))
            slice_label = sitk.GetImageFromArray(label.squeeze(2))
            # # Save the 2D slice as a NIfTI file
            sitk.WriteImage(slice_image, os.path.join(output_dir_image, f"slice_{i}.nii.gz"))
            sitk.WriteImage(slice_label, os.path.join(output_dir_label, f"slice_{i}.nii.gz"))
            i += 1



def create_data_helper(modality,images, labels, output_dir):
    output_dir_image = os.path.join(output_dir, "images")
    output_dir_label = os.path.join(output_dir, "labels")
    os.makedirs(output_dir_image, exist_ok=True)
    os.makedirs(output_dir_label, exist_ok=True)    

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

    if modality == "CT":
        create_ct_data(volume_ds, output_dir_image, output_dir_label)
    elif modality == "MRI":
        create_mr_data(volume_ds, output_dir_image, output_dir_label)

    

def create_data(data_dir, modality, output_folder):
    # get directories
    if modality == "MRI":
        data_dir = os.path.join(data_dir, "mr_train")
        images = sorted(glob.glob(os.path.join(data_dir, "mr_train_*_image.nii.gz"))) 
        labels = sorted(glob.glob(os.path.join(data_dir, "mr_train_*_label.nii.gz")))
    elif modality == "CT":
        data_dir = os.path.join(data_dir, "ct_train")
        images = sorted(glob.glob(os.path.join(data_dir, "ct_train_*_image.nii.gz"))) 
        labels = sorted(glob.glob(os.path.join(data_dir, "ct_train_*_label.nii.gz")))
    else:
        print("Modality not supported")
        return
    
    # random shuffle the list?
    random.shuffle(images)  
    random.shuffle(labels)
    
    train_images = images[:16]
    train_labels = labels[:16]
    val_images = images[16:18]
    val_labels = labels[16:18]
    test_images = images[18:]
    test_labels = labels[18:]

    create_data_helper(modality, train_images, train_labels, os.path.join(output_folder, "train"))
    create_data_helper(modality, val_images, val_labels, os.path.join(output_folder, "val"))
    create_data_helper(modality, test_images, test_labels, os.path.join(output_folder, "test"))
            


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create data for cardiac segmentation of the MM-WHS dataset')

    parser.add_argument('--data_dir', type=str, default="../MMWHS_Dataset/",
                        help='Path to the scene object')
    # CT or MRI
    parser.add_argument('--modality', type=str, default="CT",
                        help='name of the scene') 
    parser.add_argument('--output_folder', type=str, default="preprocessed/",
                        help='Path to the store directory.')

    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)

    create_data(args.data_dir, args.modality, args.output_folder)
