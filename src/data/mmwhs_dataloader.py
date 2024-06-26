import glob
import os
import random
import itertools

from torch.utils.data import Dataset
from monai.transforms import Compose, LoadImage, MapLabelValue, LoadImaged, MapLabelValued

# For two modalities
class MMWHS(Dataset):
  def __init__(self, args, labels, fold):
    self.data_dirs = args.data_dir_s # source data
    self.data_dirt = args.data_dir_t # target data
    self.target_labels = labels

    self.all_source_images = sorted(glob.glob(os.path.join(self.data_dirs, "images/case_10*")))
    self.all_source_labels = sorted(glob.glob(os.path.join(self.data_dirs, "labels/case_10*")))
    self.all_target_images = sorted(glob.glob(os.path.join(self.data_dirt, "images/case_10*")))
    self.all_target_labels = sorted(glob.glob(os.path.join(self.data_dirt, "labels/case_10*")))

    # Get al cases from this fold
    images_source = [glob.glob(self.all_source_images[idx]+ "/*.nii.gz") for idx in fold]
    self.images_source = sorted(list(itertools.chain.from_iterable(images_source)))
    labels_source = [glob.glob(self.all_source_labels[idx]+ "/*.nii.gz") for idx in fold]
    self.labels_source = sorted(list(itertools.chain.from_iterable(labels_source)))
    images_target = [glob.glob(self.all_target_images[idx]+ "/*.nii.gz") for idx in fold]
    self.images_target = sorted(list(itertools.chain.from_iterable(images_target)))
    labels_target = [glob.glob(self.all_target_labels[idx]+ "/*.nii.gz") for idx in fold]
    self.labels_target = sorted(list(itertools.chain.from_iterable(labels_target)))
        
    self.source_size = len(self.images_source)
    self.target_size = len(self.images_target)
    self.dataset_size = max(self.source_size, self.target_size)


    self.transforms_seg = Compose(
            [
              LoadImage(),
              MapLabelValue(orig_labels=[1, 2, 3, 4, 5], target_labels=self.target_labels),
              MapLabelValue(orig_labels=[421], target_labels=[0])
            ])
  
    self.transform_img = Compose(
            [
              LoadImage(),
            ])


  def __getitem__(self, index):
    # If there are more source images than target images, we randomly select a target image
    if self.source_size > self.target_size:
      random_index = random.randint(0, self.target_size - 1)
      image_s = self.transform_img(self.images_source[index])
      label_s = self.transforms_seg(self.labels_source[index])
      image_t = self.transform_img(self.images_target[random_index])
      label_t = self.transforms_seg(self.labels_target[random_index])
    elif self.target_size > self.source_size:
    # If there are more target images than source images, we randomly select a source image
      random_index = random.randint(0, self.source_size - 1)
      image_s = self.transform_img(self.images_source[random_index])
      label_s = self.transforms_seg(self.labels_source[random_index])
      image_t = self.transform_img(self.images_target[index])
      label_t = self.transforms_seg(self.labels_target[index])
    else:
      image_s = self.transform_img(self.images_source[index])
      label_s = self.transforms_seg(self.labels_source[index])
      image_t = self.transform_img(self.images_target[index])
      label_t = self.transforms_seg(self.labels_target[index])
  
    return image_s, label_s, image_t, label_t

  def __len__(self):
    return self.dataset_size
  
# For one modality
class MMWHS_single(Dataset):
  def __init__(self, data_dir, fold, labels = [1, 0, 0, 0, 0]):
    self.data_dir = data_dir
    self.target_labels = labels

    self.all_images = sorted(glob.glob(os.path.join(self.data_dir, "images/case_10*")))
    self.all_labels = sorted(glob.glob(os.path.join(self.data_dir, "labels/case_10*")))

    # Get all cases for this fold
    images = [glob.glob(self.all_images[idx]+ "/*.nii.gz") for idx in fold]
    self.images = sorted(list(itertools.chain.from_iterable(images)))
    labels = [glob.glob(self.all_labels[idx]+ "/*.nii.gz") for idx in fold]
    self.labels = sorted(list(itertools.chain.from_iterable(labels)))

    
    self.dataset_size = len(self.images)

    self.transform_dict = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            MapLabelValued(keys=["seg"], orig_labels=[1, 2, 3, 4, 5], target_labels=self.target_labels),
            MapLabelValued(keys=["seg"], orig_labels=[421], target_labels=[0]),
        ]
    )
        
    self.data = [{"img": img, "seg": seg} for img, seg in zip(self.images, self.labels)] 


  def __getitem__(self, index):
    data_point = self.transform_dict(self.data[index])
    image = data_point["img"]
    label = data_point["seg"] 

    return image, label

  def __len__(self):
    return self.dataset_size