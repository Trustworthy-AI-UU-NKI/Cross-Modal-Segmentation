from monai.config import KeysCollection
import torch
import numpy as np
from monai.transforms import MapTransform, SpatialCropd, SpatialPadd



class CropAroundMaskd2d(MapTransform):
    def __init__(self, keys, extra_spacing = 1.3):
        super().__init__(keys)
        self.marge = extra_spacing

    def __call__(self, data):
        seg = data[self.keys[1]]
        # print("here?")
        # print(seg.shape)
        # Get bounding box from the segmentation mask
        # 
        x_dim_size = seg.shape[1] # left right of the image
        y_dim_size = seg.shape[2] # front back of the image
        z_dim_size = seg.shape[0] # top bottom of the image --> in this axis we slice

        nonzero_indices = torch.nonzero(seg > 0, as_tuple=True)

        # Extract the minimum indices along each dimension
        min_x = torch.min(nonzero_indices[1]).item()  # x-axis
        min_y = torch.min(nonzero_indices[2]).item()  # y-axis

        max_x = torch.max(nonzero_indices[1]).item()  # x-axis
        max_y = torch.max(nonzero_indices[2]).item() # y-axis

        # Calculate the center of the bounding box
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        center = (center_x, center_y)

        # find the size of the bounding box
        diff_x = (max_x-min_x) * self.marge # add some extra space: default 1.4
        diff_y = (max_y-min_y) * self.marge # add some extra space: default 1.4
    
        padding = False 
    
        if diff_x > x_dim_size:
            diff_x = x_dim_size
        
        if diff_y > y_dim_size:
            diff_y = y_dim_size
        
        if diff_x > diff_y and diff_x <= y_dim_size:
            diff_y = diff_x
        
        elif diff_y > diff_x and diff_y <= x_dim_size:
            diff_x = diff_y
    
        else:
            padding = True

        # print("before cropping", data["image"].shape)
        size = (round(diff_x), round(diff_y)) # minimal difference such that the whole segmentation is obtained
        # print("size", size)
        # print("center", center)
        # Create spatial crop transform
        cropper = SpatialCropd(keys = self.keys, roi_center = center, roi_size = size)

        # Apply the crop to the image and segmentation mask
        cropped_data = cropper(data)
        # print("after cropping?", cropped_data["image"].shape)
        #print(cropped_data)

        if padding:
            # Pad the image and segmentation mask to make it square
            padder = SpatialPadd(keys = self.keys, spatial_size = round(max([diff_x, diff_y])))
            cropped_data = padder(cropped_data) 
            # print("padding")

        # Update the data dictionary
        data.update(cropped_data)

        return data



class CropAroundMaskd(MapTransform):
    def __init__(self, keys, extra_spacing = 1.4):
        super().__init__(keys)
        self.marge = extra_spacing

    def __call__(self, data):
        
        seg = data[self.keys[1]]
        
        # Get bounding box from the segmentation mask
        seg_tensor = seg.squeeze(0)
        # 
        x_dim_size = seg_tensor.shape[0] # left right of the image
        y_dim_size = seg_tensor.shape[1] # front back of the image
        z_dim_size = seg_tensor.shape[2] # top bottom of the image --> in this axis we slice

        nonzero_indices = torch.nonzero(seg_tensor > 0, as_tuple=True)

        # Extract the minimum indices along each dimension
        min_x = torch.min(nonzero_indices[0]).item()  # x-axis
        min_y = torch.min(nonzero_indices[1]).item()  # y-axis

        max_x = torch.max(nonzero_indices[0]).item()  # x-axis
        max_y = torch.max(nonzero_indices[1]).item() # y-axis

        # Calculate the center of the bounding box
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        center = (center_x, center_y, z_dim_size/2)

        # find the size of the bounding box
        diff_x = (max_x-min_x) * self.marge # add some extra space: default 1.4
        diff_y = (max_y-min_y) * self.marge # add some extra space: default 1.4
    
        padding = False 
    
        if diff_x > x_dim_size:
            diff_x = x_dim_size
        
        if diff_y > y_dim_size:
            diff_y = y_dim_size
        
        if diff_x > diff_y and diff_x <= y_dim_size:
            diff_y = diff_x
        
        elif diff_y > diff_x and diff_y <= x_dim_size:
            diff_x = diff_y
    
        else:
            padding = True


        size = (round(diff_x), round(diff_y), z_dim_size) # minimal difference such that the whole segmentation is obtained
       
        # Create spatial crop transform
        cropper = SpatialCropd(keys = self.keys, roi_center = center, roi_size = size)

        # Apply the crop to the image and segmentation mask
        cropped_data = cropper(data)

        if padding:
            # Pad the image and segmentation mask to make it square
            padder = SpatialPadd(keys = self.keys, spatial_size = round(max([diff_x, diff_y])))
            cropped_data = padder(cropped_data) 

        # Update the data dictionary
        data.update(cropped_data)

        return data
    
class CropAroundHeartd(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        # here we only have the image
        # Calculate the center of the bounding box
        img = data[self.keys[0]]
        
        # Get bounding box from the segmentation mask
        img_tensor = img.squeeze(0)

        min_dim_size = min(img_tensor.shape[0], img_tensor.shape[1])
        center_x = img_tensor.shape[0] / 2
        center_y = img_tensor.shape[1] / 2
        center_z = img_tensor.shape[2] / 2

        center = (center_x, center_y, center_z)
        size = (min_dim_size, min_dim_size, img_tensor.shape[2])

        # Create spatial crop transform
        cropper = SpatialCropd(keys = self.keys, roi_center = center, roi_size = size)

        # Apply the crop to the image and segmentation mask
        cropped_data = cropper(data)

        # Update the data dictionary
        data.update(cropped_data)

        return data
        




