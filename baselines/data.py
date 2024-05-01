import glob
import os

import torch

from monai.transforms import (
    LoadImage,
    Compose,
    MapLabelValue,
    LoadImaged,
    MapLabelValued, 
    RandFlipd,
    RandRotated
)
from torch.utils.data import Dataset
import itertools


class MMWHS_single(Dataset):
    def __init__(self, args, fold, labels = [1, 0, 0, 0, 0, 0, 0], train=True):
        self.data_dir = args.data_dir
        self.target_labels = labels
        print(self.data_dir)

        self.all_images = sorted(glob.glob(os.path.join(self.data_dir, "images/case_10*")))
        self.all_labels = sorted(glob.glob(os.path.join(self.data_dir, "labels/case_10*")))

        print(self.all_images)
        print(self.all_labels)

        images = [glob.glob(self.all_images[idx]+ "/*.nii.gz") for idx in fold]
        self.imgs = sorted(list(itertools.chain.from_iterable(images)))
        labels = [glob.glob(self.all_labels[idx]+ "/*.nii.gz") for idx in fold]
        self.labs = sorted(list(itertools.chain.from_iterable(labels)))
        
        self.dataset_size = len(self.imgs)

        self.transforms_seg = Compose(
                [
                LoadImage(),
                MapLabelValue(orig_labels=[1, 2, 3, 4, 5, 6, 7], target_labels=self.target_labels),
                MapLabelValue(orig_labels=[421], target_labels=[0]), 

                ])
        
        self.transform_img = Compose(
                [
                LoadImage(),
                ])

        # if train:
        #     self.transform_dict = Compose(
        #         [
        #             LoadImaged(keys=["img", "seg"]),
        #             MapLabelValued(keys=["seg"], orig_labels=[1, 2, 3, 4, 5, 6, 7], target_labels=self.target_labels),
        #             MapLabelValued(keys=["seg"], orig_labels=[421], target_labels=[0]),
        #             RandFlipd(keys=["img", "seg"], prob=0.5, spatial_axis=0),
        #             RandFlipd(keys=["img", "seg"], prob=0.5, spatial_axis=1),
        #             RandRotated(keys=["img", "seg"], prob=0.5, range_x=[0.4, 0.4], mode=["bilinear", "nearest"])
        #         ]
        #     )
        # else:
        self.transform_dict = Compose(
            [
                LoadImaged(keys=["img", "seg"]),
                MapLabelValued(keys=["seg"], orig_labels=[1, 2, 3, 4, 5, 6, 7], target_labels=self.target_labels),
                MapLabelValued(keys=["seg"], orig_labels=[421], target_labels=[0]),
            ]
        )
        
        self.data = [{"img": img, "seg": seg} for img, seg in zip(self.imgs, self.labs)] 



    def __getitem__(self, index):
        # image = self.transform_img(self.imgs[index])
        # label = self.transforms_seg(self.labs[index])

        data_point = self.transform_dict(self.data[index])
        image = data_point["img"]
        label = data_point["seg"]    
        return image, label

    def __len__(self):
        return self.dataset_size
    

# TO IMPLEMENT
class Retinal_Vessel_single(Dataset):
    def __init__(self, args, fold, labels = [1, 0, 0, 0, 0, 0, 0]):
        self.data_dir = args.data_dir
        self.target_labels = labels

        self.all_images = sorted(glob.glob(os.path.join(self.data_dir, "images/case_10*")))
        self.all_labels = sorted(glob.glob(os.path.join(self.data_dir, "labels/case_10*")))

        images = [glob.glob(self.all_images[idx]+ "/*.nii.gz") for idx in fold]
        self.imgs = sorted(list(itertools.chain.from_iterable(images)))
        labels = [glob.glob(self.all_labels[idx]+ "/*.nii.gz") for idx in fold]
        self.labs = sorted(list(itertools.chain.from_iterable(labels)))
        
        self.dataset_size = len(self.imgs)

        self.transforms_seg = Compose(
                [
                LoadImage(),
                MapLabelValue(orig_labels=[1, 2, 3, 4, 5, 6, 7], target_labels=self.target_labels),
                MapLabelValue(orig_labels=[421], target_labels=[0])
                ])
        
        self.transform_img = Compose(
                [
                LoadImage(),
                ])


    def __getitem__(self, index):
        image = self.transform_img(self.imgs[index])
        label = self.transforms_seg(self.labs[index])

        return image, label

    def __len__(self):
        return self.dataset_size
    