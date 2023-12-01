from monai.config import KeysCollection
import torch
import numpy as np
from monai.transforms import MapTransform, SpatialCropd


class CropAroundMaskd(MapTransform):
    def __init__(self, keys, spatial_size):
        super().__init__(keys)
        self.resize = spatial_size

    def __call__(self, data):
        
        image, seg = data[self.keys[0]].squeeze, data[self.keys[1]]
        
        # Get bounding box from the segmentation mask
        seg_tensor = seg.squeeze(0)
        spat_size = seg_tensor.shape[0]
        rest_size = seg_tensor.shape[2]

        min_x, min_y, min_z = np.min(np.where(seg_tensor > 0), axis = 1)
        max_x, max_y, max_z = np.max(np.where(seg_tensor > 0), axis = 1)

        center_x = (min_x + max_x) / 2
        center_z = (min_z + max_z) / 2

        center = (center_x, spat_size/2, center_z)

        # find the size of the bounding box
        diff_x = 2*np.max(np.array([max_x-center_x, center_x-min_x])).item()
        diff_x += diff_x / 6

        if self.resize > diff_x:
            diff_x = self.resize

        diff_z = diff_x

        if diff_z > rest_size:
            diff_z = rest_size

        size = (diff_x, spat_size, diff_z) # minimal difference such that the whole segmentation is obtained

        # Create spatial crop transform
        cropper = SpatialCropd(keys =self.keys, roi_center = center, roi_size = size)

        # Apply the crop to the image and segmentation mask
        cropped_data = cropper(data)

        # Update the data dictionary
        data.update(cropped_data)

        
        return data
