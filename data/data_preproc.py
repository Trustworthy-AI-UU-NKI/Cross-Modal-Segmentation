# imports 
import numpy as np
import os
import monai
from monai.data import Dataset, DataLoader, GridPatchDataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Resized,
    ScaleIntensityd,
    SqueezeDimd,
    MapLabelValued,
    LoadImage,
    Rotated,
    Flipd,
    SpatialPadd,
    SpatialResampled,
    Spacingd,
    Orientationd,
)

from transforms import CropAroundMaskd
import matplotlib.pyplot as plt
import glob
import torch
import argparse
import sys
import SimpleITK as sitk
import random

def create_ct_data(dataset_files, output_dir_image, output_dir_label, spatial_size):  
    volume_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            ScaleIntensityd(keys="img"), # normalization between 0 and 1 # segmentation masl also?
            EnsureTyped(keys=["img", "seg"]),
        ])  

    sample = 1001
    i = 0
    
    for file in dataset_files:
        file_name = f"case_{sample}"
        output_dir_image_file = os.path.join(output_dir_image, file_name)
        output_dir_label_file = os.path.join(output_dir_label, file_name)
        os.makedirs(output_dir_image_file, exist_ok=True)
        os.makedirs(output_dir_label_file, exist_ok=True)   
        ds = Dataset(data=[file], transform=volume_transforms)


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
                Resized(keys=["img"], spatial_size=[256,256], mode="bilinear"),
                Resized(keys=["seg"], spatial_size=[256,256], mode="nearest"),
                MapLabelValued(keys=["seg"], orig_labels=[205, 420, 500, 550, 600, 820, 850], target_labels=[1, 2, 3, 4, 5 , 6 , 7]), 
                SqueezeDimd(keys=["img", "seg"], dim=0),
            ]
        )

        example_patch_ds = GridPatchDataset(data=ds, patch_iter=patch_func, transform=patch_transform)
        patch_data_loader = DataLoader(example_patch_ds, batch_size=1)

        for batch in patch_data_loader:
            image, label = batch[0]["img"], batch[0]["seg"]
            # Convert the torch tensor to a SimpleITK image
            slice_image = sitk.GetImageFromArray(image)
            slice_label = sitk.GetImageFromArray(label)
            # # Save the 2D slice as a NIfTI file
            sitk.WriteImage(slice_image, os.path.join(output_dir_image_file, f"slice_{i}.nii.gz"))
            sitk.WriteImage(slice_label, os.path.join(output_dir_label_file, f"slice_{i}.nii.gz"))
            i += 1
        
        sample += 1

def create_mr_data(dataset_files, output_dir_image, output_dir_label, spatial_size):
    volume_transforms = Compose(
            [
                LoadImaged(keys=["img", "seg"]),
                EnsureChannelFirstd(keys=["img", "seg"]),
                ScaleIntensityd(keys="img"), # normalization between 0 and 1 # segmentation masl also?
                EnsureTyped(keys=["img", "seg"]),
                Spacingd(keys=["img"], pixdim=(1, 1, 1), mode = 'bilinear'),
                Spacingd(keys=["seg"], pixdim=(1, 1, 1), mode = 'nearest'),
                Orientationd(keys=["img", "seg"], axcodes="LPI"),
                CropAroundMaskd(keys=["img", "seg"], extra_spacing=1.3),
            ]
        ) 
    
    sample = 1001
    i = 0
    
    for file in dataset_files:
        file_name = f"case_{sample}"
        output_dir_image_file = os.path.join(output_dir_image, file_name)
        output_dir_label_file = os.path.join(output_dir_label, file_name)
        os.makedirs(output_dir_image_file, exist_ok=True)
        os.makedirs(output_dir_label_file, exist_ok=True)   
        ds = Dataset(data=[file], transform=volume_transforms)


        # Create patches (i.e. images)
        patch_func = monai.data.PatchIterd(
            keys=["img", "seg"],
            patch_size=(None, None, 1),  # dynamic first two dimensions
            start_pos=(0, 0, 0)
        )

        patch_transform = Compose(
            [
                Resized(keys=["img"], spatial_size=[spatial_size,spatial_size,1], mode = "bilinear"),
                Resized(keys=["seg"], spatial_size=[spatial_size,spatial_size,1], mode = "nearest"),
                MapLabelValued(keys=["seg"], orig_labels=[205, 420, 500, 550, 600, 820, 850, 421], target_labels=[1, 2, 3, 4, 5 , 6 , 7, 0]),
                SqueezeDimd(keys=["img", "seg"], dim=0),  # squeeze the first dim
            ]
        )

        example_patch_ds = GridPatchDataset(data=ds, patch_iter=patch_func, transform=patch_transform)
        patch_data_loader = DataLoader(example_patch_ds, batch_size=1)

        for batch in patch_data_loader:
            image, label = batch[0]["img"], batch[0]["seg"]
            # Convert the torch tensor to a SimpleITK image

            slice_image = sitk.GetImageFromArray(image)
            slice_label = sitk.GetImageFromArray(label)
            # # Save the 2D slice as a NIfTI file
            sitk.WriteImage(slice_image, os.path.join(output_dir_image_file, f"slice_{i}.nii.gz"))
            sitk.WriteImage(slice_label, os.path.join(output_dir_label_file, f"slice_{i}.nii.gz"))
            i += 1
        
        sample += 1



def create_data_helper(modality,images, labels, output_dir, spatial_size):
    output_dir_image = os.path.join(output_dir, "images")
    output_dir_label = os.path.join(output_dir, "labels")
    os.makedirs(output_dir_image, exist_ok=True)
    os.makedirs(output_dir_label, exist_ok=True)    

    dataset_files = [{"img": img, "seg": seg} for img, seg in zip(images, labels)]  
    
    if modality == "CT":
        create_ct_data(dataset_files, output_dir_image, output_dir_label, spatial_size)
    elif modality == "MRI":
        create_mr_data(dataset_files, output_dir_image, output_dir_label, spatial_size)

    

def create_data(data_dir, modality, output_folder, spatial_size):
    # get directories
    if modality == "MRI":
        data_dir = os.path.join(data_dir, "mr_train")
        images = sorted(glob.glob(os.path.join(data_dir, "mr_train_*_image.nii.gz"))) 
        labels = sorted(glob.glob(os.path.join(data_dir, "mr_train_*_label.nii.gz")))
        output_folder = os.path.join(output_folder, "MRI")
    elif modality == "CT":
        data_dir = os.path.join(data_dir, "ct_train")
        images = sorted(glob.glob(os.path.join(data_dir, "ct_train_*_image.nii.gz"))) 
        labels = sorted(glob.glob(os.path.join(data_dir, "ct_train_*_label.nii.gz")))
        output_folder = os.path.join(output_folder, "CT")
    else:
        print("Modality not supported")
        return


    create_data_helper(modality, images, labels, os.path.join(output_folder, "annotated"), spatial_size)
           


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create data for cardiac segmentation of the MM-WHS dataset')

    parser.add_argument('--data_dir', type=str, default="../MMWHS_Dataset/",
                        help='Path to the scene object')
    # CT or MRI
    parser.add_argument('--modality', type=str, default="CT",
                        help='name of the scene') 
    parser.add_argument('--output_folder', type=str, default="preprocessed/",
                        help='Path to the store directory.')

    parser.add_argument('--spatial_size', type=int, default=256,
                        help='Spatial size of images and labels.')

    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)

    create_data(args.data_dir, args.modality, args.output_folder, args.spatial_size)
