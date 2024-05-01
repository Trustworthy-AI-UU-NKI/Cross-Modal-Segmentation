import os
import torchvision
from tensorboardX import SummaryWriter
import numpy as np
from PIL import Image
import SimpleITK as sitk
import torch 
import glob

class Saver():
  def __init__(self, fold, args, num_classes=2):
    self.name = os.path.join(args.name, f"fold_{fold}")   
    self.display_dir = os.path.join(args.display_dir, self.name)
    self.model_dir = os.path.join(args.result_dir, self.name)   
    self.img_save_freq = args.img_save_freq

    # make directories
    os.makedirs(self.display_dir, exist_ok=True)
    os.makedirs(self.model_dir, exist_ok=True)

    # create tensorboard writer
    self.writer = SummaryWriter(logdir=self.display_dir)
    self.save_hparams(args, num_classes)
  
  def save_hparams(self, args, num_classes):
    hparams = {
      "lr"        : args.lr,
      "lr seg"    : args.lr67,
      "lr5"       : args.lr5,
      "lrA"       : args.lr_A,
      "lrB"       : args.lr_B,
      "bs"        : args.bs,
      "resolution" : args.resolution,
      "num_classes" : num_classes,
      "keep_rate" : args.keep_rate,
      "epochs"    : args.epochs,
      "pred"      : args.pred,
      "source modality"  : args.modality,
      "keep_rate" : args.keep_rate, 
      "skip"      : args.skip

    }
    self.writer.add_hparams(hparam_dict=hparams, metric_dict={})


  def write_val_dsc(self, ep, new_val, new_val_false):
     self.writer.add_scalar("Validation Dice", new_val, ep)
     self.writer.add_scalar("Validation Dice False", new_val_false, ep)


  # write losses and images to tensorboard
  def write_display(self, ep, model):
    members = [attr for attr in dir(model) if not callable(getattr(model, attr)) and 'loss' in attr and 'item' in attr]
   
    for m in members:
       self.writer.add_scalar(m, getattr(model, m), ep)
    
  # save result images
  def write_images(self, images_b, labels_b, pred_mask_b, ep):
  
    if (ep + 1) % self.img_save_freq == 0:
      image_display = torch.cat((images_b, labels_b, pred_mask_b), dim=0)
      image_dis = torchvision.utils.make_grid(image_display, nrow=1)
      self.writer.add_image('Validation samples', torch.Tensor(image_dis), ep)


  # save model
  def write_model(self, ep, model, val_dcs, val_dcs_false):
    print(f"--- Remove old model before saving new one ---")
    # Define the pattern to search for existing model files in the directory
    existing_model_files = glob.glob(f"{self.model_dir}/*.pth")
    # Delete each found model file
    for file_path in existing_model_files:
        try:
            os.remove(file_path)
            print(f"Deleted old model file: {file_path}")
        except OSError as e:
            print(f"Error deleting file {file_path}: {e}")
  

    print(f"--- save the model @ ep {ep} with dcs {val_dcs} ---")
    filename = f"{self.model_dir}/ep:{ep}_val:{val_dcs}_valFalse{val_dcs_false}_.pth"
    model.save(filename, ep)
   