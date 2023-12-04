from monai.config import KeysCollection
import torch
import numpy as np
from monai.transforms import MapTransform, SpatialCropd, SpatialPadd


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

        # THis is not working, incorrect --> fix this!!
        min_x, min_y, min_z = np.min(np.where(seg_tensor > 0), axis = 1)
        max_x, max_y, max_z = np.max(np.where(seg_tensor > 0), axis = 1)

        print(min_x, min_y, min_z)
        print(max_x, max_y, max_z)

        center_x = (min_x + max_x) / 2
        center_z = (min_z + max_z) / 2

        center = (center_x, spat_size/2, center_z)

        # find the size of the bounding box
        diff_x = (max_x-min_x) * 1.05 # add some extra space
        diff_z = (max_z-min_z) * 1.05
        #opt_diff_x = diff_x + diff_x / 6
        print("diff_x: ", diff_x)
        print("diff_z: ", diff_z)
    
        padding = False 
        
        assert spat_size >= rest_size, "resolution is to small to crop around the segmentation mask"

        if diff_x <= rest_size and diff_x >= diff_z:
            print('first case')
            diff_z = diff_x
        elif diff_x <= rest_size:
            print('second case')
            diff_x = diff_z
        else:
            print('third case')
            # We have to pad later on to make the image square
            diff_z = rest_size
            padding = True

        size = (diff_x, spat_size, diff_z) # minimal difference such that the whole segmentation is obtained
        print("size: ", size)

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
