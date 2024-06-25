import glob
import os
import random
import itertools

from torch.utils.data import Dataset
from monai.transforms import Compose, LoadImage, MapLabelValue, LoadImaged, MapLabelValued

# For two modalities
class CHAOS(Dataset):
  def __init__(self, args, labels, fold):
    self.data_dirs = args.data_dir_s # source data
    self.data_dirt = args.data_dir_t # target data
    self.target_labels = labels

    self.all_source_images = sorted(glob.glob(os.path.join(self.data_dirs, "images/case_*")))
    self.all_source_labels = sorted(glob.glob(os.path.join(self.data_dirs, "labels/case_*")))
    self.all_target_images = sorted(glob.glob(os.path.join(self.data_dirt, "images/case_*")))
    self.all_target_labels = sorted(glob.glob(os.path.join(self.data_dirt, "labels/case_*")))

    # Get all the cases in this fold
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
              MapLabelValue(orig_labels=[1, 2, 3, 4], target_labels=self.target_labels),
            ])
  
    self.transform_img = Compose(
            [
              LoadImage(),
            ])


  def __getitem__(self, index):
    if self.source_size > self.target_size:
      random_index = random.randint(0, self.target_size - 1)
      image_s = self.transform_img(self.images_source[index])
      label_s = self.transforms_seg(self.labels_source[index])
      image_t = self.transform_img(self.images_target[random_index])
      label_t = self.transforms_seg(self.labels_target[random_index])
    elif self.target_size > self.source_size:
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
  
    return image_s.unsqueeze(0), label_s.unsqueeze(0), image_t.unsqueeze(0), label_t.unsqueeze(0)

  def __len__(self):
    return self.dataset_size
  
# For one modality
class CHAOS_single(Dataset):
  def __init__(self, data_dir, fold, labels = [1, 0, 0, 0]):
    self.data_dir = data_dir
    self.target_labels = labels

    self.all_images = sorted(glob.glob(os.path.join(self.data_dir, "images/case_*")))
    self.all_labels = sorted(glob.glob(os.path.join(self.data_dir, "labels/case_*")))

    # Get all the cases in this fold
    images = [glob.glob(self.all_images[idx]+ "/*.nii.gz") for idx in fold]
    self.images = sorted(list(itertools.chain.from_iterable(images)))
    labels = [glob.glob(self.all_labels[idx]+ "/*.nii.gz") for idx in fold]
    self.labels = sorted(list(itertools.chain.from_iterable(labels)))
    
    self.dataset_size = len(self.images)

    self.transform_dict = Compose(
        [
            LoadImaged(keys=["img", "seg"],),
            MapLabelValued(keys=["seg"], orig_labels=[1, 2, 3, 4], target_labels=self.target_labels)
        ]
    )
  
    self.data = [{"img": img, "seg": seg} for img, seg in zip(self.images, self.labels)] 


  def __getitem__(self, index):
    data_point = self.transform_dict(self.data[index])
    image = data_point["img"]
    label = data_point["seg"] 

    return image.unsqueeze(0), label.unsqueeze(0)

  def __len__(self):
    return self.dataset_size