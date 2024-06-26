import torch
from tqdm import tqdm
import numpy as np
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity 

from monai.transforms import ScaleIntensity

def eval_vmfnet_mm(model, loader, device):
    """Evaluation with reconstruction performance"""
    model.eval()
    n_val = len(loader)  # the number of batches
    display_itr = 0

    first = True
    lpips_list_target = []
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="squeeze", normalize=True).to(device)
    fake_dsc = 0
    n_val_dsc = n_val

    # For all validation images
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for img_s, label_s, img_t, label_t in loader:
            img_s = img_s.to(device)
            label_s = label_s.to(device)

            # Get all metrics, images and compositional representations for evaluation
            with torch.no_grad():
                metrics_dict, images_dict, visuals_dict  = model.forward_eval(img_s, label_s)
            
            if first:
                metrics_dict_total = metrics_dict
                first = False
            else:
                for key, value in metrics_dict.items():
                    metrics_dict_total[key] += value
            
            # Skip dice class when no label is there. Background is always there
            if torch.all(label_s==0):
                n_val_dsc -= 1
            else:
                fake_dsc += metrics_dict["Target/DSC_fake"]

            # Get LPIPS
            img_t3 = torch.cat((img_t, img_t, img_t), dim=1)
            fake_img = ScaleIntensity()(images_dict['Target/fake'])
            fake_img_t3 = torch.cat((fake_img, fake_img, fake_img), dim=1)
            lpips_list_target.append(lpips(fake_img_t3, img_t3))

            # Choose images to show in TB. 5 is chosen randomly
            if display_itr == 5:
                image_dict_show = images_dict
                visual_dict_show = visuals_dict
                
            display_itr += 1
            pbar.update()

    for key, value in metrics_dict.items():
        metrics_dict_total[key] /= n_val

    metrics_dict_total["Target/DSC_fake"] = fake_dsc / n_val
    return metrics_dict_total, image_dict_show, visual_dict_show, torch.mean(torch.stack(lpips_list_target))


def test_vmfnet_mm(model, loader, device):
    model.eval()
    n_val = len(loader)  # the number of batches
    display_itr = 0

    assd = 0
    dsc_0 = 0
    dsc_1 = 0
    n_val_assd = n_val
    n_val_dsc = n_val

    # For every test image
    with tqdm(total=n_val, desc='Test round', unit='batch', leave=False) as pbar:
        for img_t, label_t in loader:
            img_t = img_t.to(device)
            label_t = label_t.to(device)

            # Get all metrics, images and compositional representations for evaluation
            with torch.no_grad():
                metrics_dict, images_dict, visuals_dict  = model.test(img_t, label_t)
                        
            # Get ASSD
            if np.isinf(metrics_dict["Target/assd"]).any():
                # print("inf value in assd")
                n_val_assd -= 1
            else:
                assd += metrics_dict["Target/assd"]
            
            # Skip dice of class 1 when no label is there. Background (class 0) is always there
            if torch.all(label_t==0):
                n_val_dsc -= 1
            else:
                dsc_1 += metrics_dict["Target/DSC_1"]

            dsc_0 += metrics_dict["Target/DSC_0"]    

            # Choose images to show in TB. 5 is chosen randomly
            if display_itr == 5:
                image_dict_show = images_dict
                visual_dict_show = visuals_dict
                
            display_itr += 1
            pbar.update()

    
    return assd/n_val_assd, dsc_0/n_val, dsc_1/n_val_dsc, image_dict_show, visual_dict_show
