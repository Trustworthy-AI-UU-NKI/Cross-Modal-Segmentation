from monai.config import KeysCollection
import torch
import numpy as np
from monai.transforms import MapTransform, SpatialCropd, SpatialPadd


class CropAroundMaskd(MapTransform):
    def __init__(self, keys, spatial_size):
        super().__init__(keys)
        self.resize = spatial_size

    def __call__(self, data):
        
        seg = data[self.keys[1]]
        
        # Get bounding box from the segmentation mask
        seg_tensor = seg.squeeze(0)
        spat_size = seg_tensor.shape[0]
        rest_size = seg_tensor.shape[2]

        nonzero_indices = torch.nonzero(seg_tensor > 0, as_tuple=True)

        # Extract the minimum indices along each dimension
        min_x = torch.min(nonzero_indices[0]).item()  # x-axis
        min_z = torch.min(nonzero_indices[2]).item()  # z-axis

        max_x = torch.max(nonzero_indices[0]).item()  # x-axis
        max_z = torch.max(nonzero_indices[2]).item() # z-axis

        # Calculate the center of the bounding box
        center_x = (min_x + max_x) / 2
        center_z = (min_z + max_z) / 2

        center = (center_x, spat_size/2, center_z)

        # find the size of the bounding box
        diff_x = (max_x-min_x) * 1.1 # add some extra space
        diff_z = (max_z-min_z) * 1.1 # add some extray space
    
        padding = False 
    
        assert spat_size >= rest_size, "resolution is to small to crop around the segmentation mask"

        if diff_x <= rest_size and diff_x >= diff_z:
            diff_z = diff_x
        elif diff_x <= rest_size and diff_z <= rest_size:
            diff_x = diff_z
        elif diff_x <= rest_size:
            diff_x = rest_size
            diff_z = rest_size
        else:
            # We have to pad later on to make the image square
            diff_z = rest_size
            padding = True

        size = (round(diff_x), spat_size, round(diff_z)) # minimal difference such that the whole segmentation is obtained
       
        # Create spatial crop transform
        cropper = SpatialCropd(keys =self.keys, roi_center = center, roi_size = size)

        # Apply the crop to the image and segmentation mask
        cropped_data = cropper(data)

        if padding:
            # Pad the image and segmentation mask to make it square
            padder = SpatialPadd(keys = self.keys, spatial_size = diff_x)
            cropped_data = padder(cropped_data) 

        # Update the data dictionary
        data.update(cropped_data)

        return data
