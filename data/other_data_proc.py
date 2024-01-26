# imports 
import numpy as np
import monai
from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityd,
    Resized,
    EnsureChannelFirstd,
    SqueezeDimd,
)


from monai.data import Dataset, DataLoader
import itertools
import os

from sklearn.model_selection import KFold

from monai.networks.nets import UNet

import math
import matplotlib.pyplot as plt
import glob
import torch
import SimpleITK as sitk
from transforms import *


def main ():
    transforms = Compose(
                [
                    LoadImaged(keys=["image", "label"]),
                    #Spacing(pixdim=(0.5, 0.5), mode="bilinear"),
                    #ResizeWithPadOrCrop(spatial_size=[256, 256]),
                    #CenterSpatialCrop(roi_size=[200,200]),
                    EnsureChannelFirstd(keys=["image", "label"]),
                    SqueezeDimd(keys=["image", "label"], dim=0),
                    CropAroundMaskd2d(keys=["image", "label"]),#  = 256),
                    Resized(keys=["image", "label"], spatial_size=[256, 256]),
                    #AsChannelLastd(keys=["image", "label"], channel_dim=0),
                    ScaleIntensityd(keys=["image"]),
                ]
            )
    
    data_dir = "other/MR_withGT"
    output_dir = "other/MR_withGT_proc/annotated/"
    output_dir_image = os.path.join(output_dir, "images")
    output_dir_label = os.path.join(output_dir, "labels")

    os.makedirs(output_dir_image, exist_ok=True)
    os.makedirs(output_dir_label, exist_ok=True)

    slices = 0
    sample = 1001
    # CT: 33,53
    for i in range(1, 20):
        file_name = f"case_{sample}"
        output_dir_image_file = os.path.join(output_dir_image, file_name)
        os.makedirs(output_dir_image_file, exist_ok=True)
        output_dir_label_file = os.path.join(output_dir_label, file_name)
        os.makedirs(output_dir_label_file, exist_ok=True)

        images = sorted(glob.glob(os.path.join(data_dir, f"img{i}_slice*.nii.gz")))
        labels = sorted(glob.glob(os.path.join(data_dir, f"lab{i}_slice*.nii.gz")))

        example_dataset = [{"image": img, "label": seg} for img, seg in zip(images, labels)]
        print(len(example_dataset))
        
        example_patch_ds = Dataset(data=example_dataset, transform=transforms)
        patch_data_loader = DataLoader(example_patch_ds, batch_size=1)

        for batch_data in patch_data_loader:
            img = batch_data["image"][0]
            seg = batch_data["label"][0]
            print(img.shape)

            # Modify the permute indices as per your data dimensions
            img_reordered = img.permute(2, 1, 0)  # Change this based on your data's shape
            seg_reordered = seg.permute(2, 1, 0)  # Change this based on your data's shape

            # Convert to SimpleITK image
            slice_image = sitk.GetImageFromArray(img_reordered.numpy())
            slice_label = sitk.GetImageFromArray(seg_reordered.numpy())
            sitk.WriteImage(slice_image, os.path.join(output_dir_image_file, f"slice_{slices}.nii.gz"))
            sitk.WriteImage(slice_label, os.path.join(output_dir_label_file, f"slice_{slices}.nii.gz"))
            slices += 1
        quit()
        sample += 1




if __name__ == "__main__":
    main()