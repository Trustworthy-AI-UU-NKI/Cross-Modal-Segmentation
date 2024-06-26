# imports 
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    CenterSpatialCropd,
    Spacingd,
    ScaleIntensityd,
    MapLabelValued,
    SqueezeDimd,
    Transposed
)

import numpy as np
from monai.data import Dataset, DataLoader
import os
import glob
from transforms import *
import nibabel as nib


def get_roi_size(case, mod):
    if mod == 0:
        if case in [0, 1, 8, 10, 11, 12, 14, 17, 18]:
            return 200
        if case in [2, 3, 4, 6, 15, 16]:
            return 220
        if case in [9]:
            return 230
        if case in [5]:
            return 240
        if case in [7, 13]:
            return 180
        else: 
            return 256
    
    if mod == 1:
        if case in [0, 3, 8, 11, 12, 14, 15]:
            return 220
        if case in [1, 10, 17, 18]:
            return 210
        if case in [7, 13]:
            return 200
        if case in [2, 4, 6, 9, 16]:
            return 240
        else: 
            return 256
  

def preprocess_data(mod, data_dir_image, data_dir_label, output_dir):
    slices = 0
    cas = 0
    output_dir_image = os.path.join(output_dir, "images")
    output_dir_label = os.path.join(output_dir, "labels")

    os.makedirs(output_dir_image, exist_ok=True)
    os.makedirs(output_dir_label, exist_ok=True)

    file_cases = sorted(glob.glob(data_dir_image))
    label_cases = sorted(glob.glob(data_dir_label)) 

    for im_case, lab_case in zip(file_cases, label_cases):
        file_name = f"case_{cas}"
        size = get_roi_size(cas, mod)
        print(file_name)
        transforms = Compose(
                [
                    LoadImaged(keys=["image", "label"], image_only=False),
                    EnsureChannelFirstd(keys=["label"]),
                    EnsureChannelFirstd(keys=["image"], channel_dim=-1),
                    CenterSpatialCropd(keys=["image", "label"], roi_size = (size, size)),
                    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
                    AdaptedResize(keys=["image", "label"], spat_size=256),
                    ScaleIntensityd(keys=["image"]),
                    MapLabelValued(keys=["label"], orig_labels=[0, 63, 126, 189, 252], target_labels=[0, 1, 2, 3, 4]),
                    Transposed(keys=["image", "label"], indices=(0, 2, 1)),
                    SqueezeDimd(keys=["image", "label"], dim=0),
                ]
            )

        output_dir_image_file = os.path.join(output_dir_image, file_name)
        os.makedirs(output_dir_image_file, exist_ok=True)
        output_dir_label_file = os.path.join(output_dir_label, file_name)
        os.makedirs(output_dir_label_file, exist_ok=True)

        images = sorted(glob.glob(os.path.join(im_case, f"IMG-*.dcm")))
        labels = sorted(glob.glob(os.path.join(lab_case, f"IMG-*.png")))

        example_dataset = [{"image": img, "label": seg} for img, seg in zip(images, labels)]
        
        example_patch_ds = Dataset(data=example_dataset, transform=transforms)
        patch_data_loader = DataLoader(example_patch_ds, batch_size=1)

        for batch_data in patch_data_loader:
            img = batch_data["image"][0]
            seg = batch_data["label"][0]
            
            # Create the NIfTI image
            nifti_image = nib.Nifti1Image(img.numpy(), img.meta["affine"].numpy().astype(np.float32))
            nifti_label = nib.Nifti1Image(seg.numpy(), img.meta["affine"].numpy().astype(np.float32))

            # Save the NIfTI image
            nib.save(nifti_image, os.path.join(output_dir_image_file, f"slice_{slices}.nii.gz"))
            nib.save(nifti_label, os.path.join(output_dir_label_file, f"slice_{slices}.nii.gz"))
            slices += 1
        
        cas += 1


def main ():
    # First modality
    print("MRI T1")
    data_dir = "../CHAOS_MR/*/T1DUAL/"
    data_dir_image = os.path.join(data_dir, "DICOM_anon/InPhase")
    data_dir_label = os.path.join(data_dir, "Ground")

    output_dir = "preprocessed_chaos/T1/"
    preprocess_data(0, data_dir_image, data_dir_label, output_dir)

    # Second modality
    print("MRI T2")
    data_dir = "../CHAOS_MR/*/T2SPIR/"  
    data_dir_image = os.path.join(data_dir, "DICOM_anon")
    data_dir_label = os.path.join(data_dir, "Ground") 
    output_dir = "preprocessed_chaos/T2/"

    preprocess_data(1, data_dir_image, data_dir_label, output_dir)



if __name__ == "__main__":
    main()