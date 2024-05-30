from monai.config import KeysCollection
import torch
import numpy as np
from monai.transforms import MapTransform, SpatialCropd, SpatialPadd, Resized



class CropAroundMaskd2dResize(MapTransform):
    def __init__(self, keys, spatial_size=256, extra_spacing = 1.3):
        super().__init__(keys)
        self.marge = extra_spacing
        self.spat = spatial_size

    def __call__(self, data):
        seg = data[self.keys[1]]
        # print("here?")
        # print(seg.shape)
        # Get bounding box from the segmentation mask
        # 
        x_dim_size = seg.shape[1] # left right of the image
        y_dim_size = seg.shape[2] # front back of the image
    
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
        # print("SIZE: ", size)
        # print(data[self.keys[0]]["metadata"]["pixdim"])
        # print(data[self.keys[0]].meta["pixdim"])
        

        # Apply the crop to the image and segmentation mask
        cropped_data = cropper(data)
        # print("after cropping?", cropped_data["image"].shape)
        #print(cropped_data)

        if padding:
            # Pad the ima÷ge and segmentation mask to make it square
            # print("WE PAD?????")
            max_dim = round(max([diff_x, diff_y]))
            padder = SpatialPadd(keys = self.keys, spatial_size = max_dim)
            cropped_data = padder(cropped_data) 
            size = (max_dim, max_dim)
            # print("padding")

        resizer = Resized(keys=["image", "label"], spatial_size=[256, 256], mode=["bilinear", "nearest"])
        res_cropped_data = resizer(cropped_data)

        # Update the data dictionary
        data.update(res_cropped_data)
        new_pix_dim = (size[0] / 256) * 1.0
        data[self.keys[0]].meta["pixdim"][2] = new_pix_dim
        data[self.keys[0]].meta["pixdim"][3] = new_pix_dim
        # print(data[self.keys[0]].meta["pixdim"])
        # data["metadata"]["pixdim"] = new_pixel_dimensions


        return data


class AdaptedResize(MapTransform):
    def __init__(self, keys, spat_size=256):
        super().__init__(keys)
        self.spatial_size = spat_size

    def __call__(self, data):
        size = data[self.keys[0]].shape[1]
        new_pix_dim = (size / 256) * 1.0
        sizer = Resized(keys=["image", "label"], spatial_size=[256, 256], mode=["bilinear", "nearest"])
        resized_data = sizer(data)
        data.update(resized_data)

        data[self.keys[0]].meta["spacing"][0] = new_pix_dim
        data[self.keys[0]].meta["spacing"][1] = new_pix_dim

        return data