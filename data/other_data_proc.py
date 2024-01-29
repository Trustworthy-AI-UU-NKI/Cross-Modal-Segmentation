# imports 

from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityd,
    Resized,
    EnsureChannelFirstd,
    SqueezeDimd,
    MapLabelValued
)


from monai.data import Dataset, DataLoader
import os
import glob
import SimpleITK as sitk
from transforms import *


def preprocess_data(transforms, data_dir, output_dir, r1, r2, starting_case = 1001):
    output_dir_image = os.path.join(output_dir, "images")
    output_dir_label = os.path.join(output_dir, "labels")

    os.makedirs(output_dir_image, exist_ok=True)
    os.makedirs(output_dir_label, exist_ok=True)

    slices = 0
    sample = starting_case
    
    for i in range(r1, r2):
        file_name = f"case_{sample}"
        print(file_name)

        output_dir_image_file = os.path.join(output_dir_image, file_name)
        os.makedirs(output_dir_image_file, exist_ok=True)
        output_dir_label_file = os.path.join(output_dir_label, file_name)
        os.makedirs(output_dir_label_file, exist_ok=True)

        images = sorted(glob.glob(os.path.join(data_dir, f"img{i}_slice*.nii.gz")))
        labels = sorted(glob.glob(os.path.join(data_dir, f"lab{i}_slice*.nii.gz")))

        example_dataset = [{"image": img, "label": seg} for img, seg in zip(images, labels)]
        
        example_patch_ds = Dataset(data=example_dataset, transform=transforms)
        patch_data_loader = DataLoader(example_patch_ds, batch_size=1)

        for batch_data in patch_data_loader:
            img = batch_data["image"][0]
            seg = batch_data["label"][0]

            # Modify the permute indices as per your data dimensions
            img_reordered = img.permute(2, 1, 0)  # Change this based on your data's shape
            seg_reordered = seg.permute(2, 1, 0)  # Change this based on your data's shape

            # Convert to SimpleITK image
            slice_image = sitk.GetImageFromArray(img_reordered.numpy())
            slice_label = sitk.GetImageFromArray(seg_reordered.numpy())
            sitk.WriteImage(slice_image, os.path.join(output_dir_image_file, f"slice_{slices}.nii.gz"))
            sitk.WriteImage(slice_label, os.path.join(output_dir_label_file, f"slice_{slices}.nii.gz"))
            slices += 1
    
        sample += 1


def main ():
    transforms = Compose(
                [
                    LoadImaged(keys=["image", "label"]),
                    EnsureChannelFirstd(keys=["image", "label"]),
                    SqueezeDimd(keys=["image", "label"], dim=0),
                    CropAroundMaskd2d(keys=["image", "label"]),
                    Resized(keys=["image"], spatial_size=[256, 256], mode="bilinear"),
                    Resized(keys=["label"], spatial_size=[256, 256], mode="nearest"),
                    ScaleIntensityd(keys=["image"]),
                    MapLabelValued(keys=["label"], orig_labels=[205, 420, 500, 550, 600], target_labels=[1, 2, 3, 4, 5]), 
                ]
            )
    
    # MRI with GT
    print("MRI")
    data_dir = "other/MR_withGT"    
    output_dir = "other/MR_withGT_proc/annotated/"
    r1 = 20# 1
    r2 = 21
    preprocess_data(transforms, data_dir, output_dir, r1, r2, starting_case = 1020)

    # # CT with GT
    # print("CT")
    # data_dir = "other/CT_withGT"
    # output_dir = "other/CT_withGT_proc/annotated/"
    # r1 = 33
    # r2 = 53
    # preprocess_data(transforms, data_dir, output_dir, r1, r2)

    # MRI with 'fake' GT
    print("MRI with fake GT")
    data_dir = "other/MR_woGT"    
    output_dir = "other/MR_withGT_proc/annotated/"
    r1 = 21
    r2 = 47
    starting_case = 1021
    preprocess_data(transforms, data_dir, output_dir, r1, r2, starting_case = starting_case)

    # CT with 'fake' GT
    print("CT with fake GT")
    data_dir = "other/CT_woGT"
    output_dir = "other/CT_withGT_proc/annotated/"
    r1 = 1
    r2 = 33
    starting_case = 1021
    preprocess_data(transforms, data_dir, output_dir, r1, r2, starting_case = starting_case)


if __name__ == "__main__":
    main()