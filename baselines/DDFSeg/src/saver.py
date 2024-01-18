import os
import torchvision
from tensorboardX import SummaryWriter
import numpy as np
from PIL import Image
import SimpleITK as sitk
import torch 

class Saver():
  def __init__(self, fold, args):
    self.name = os.path.join(args.name, f"fold_{fold}")   
    self.display_dir = os.path.join(args.display_dir, self.name)
    self.model_dir = os.path.join(args.result_dir, self.name)   
    self.display_freq = args.display_freq
    self.model_save_freq = args.model_save_freq

    # make directories
    os.makedirs(self.display_dir, exist_ok=True)
    os.makedirs(self.model_dir, exist_ok=True)

    # create tensorboard writer
    self.writer = SummaryWriter(logdir=self.display_dir)

  # write losses and images to tensorboard
  def write_display(self, total_it, model):
    if (total_it + 1) % self.display_freq == 0:
      # write loss
      members = [attr for attr in dir(model) if not callable(getattr(model, attr)) and not attr.startswith("__") and 'loss' in attr]
      for m in members:
        self.writer.add_scalar(m, getattr(model, m), total_it)
      # write img
      image_dis = torchvision.utils.make_grid(model.image_display, nrow=model.image_display.size(0)//2)/2 + 0.5
      self.writer.add_image('Image', torch.Tensor(image_dis), total_it)

  # save result images
  def write_img(self, ep, model):
    if (ep + 1) % self.img_save_freq == 0:
      assembled_images = model.assemble_outputs()
      img_filename = '%s/gen_%05d.jpg' % (self.image_dir, ep)
      torchvision.utils.save_image(assembled_images / 2 + 0.5, img_filename, nrow=1)
    elif ep == -1:
      assembled_images = model.assemble_outputs()
      img_filename = '%s/gen_last.jpg' % (self.image_dir, ep)
      torchvision.utils.save_image(assembled_images / 2 + 0.5, img_filename, nrow=1)

  # save model
  def write_model(self, ep, total_it, model):
    if (ep + 1) % self.model_save_freq == 0:
      print('--- save the model @ ep %d ---' % (ep))
      model.save('%s/%05d.pth' % (self.model_dir, ep), ep, total_it)
    elif ep == -1:
      model.save('%s/last.pth' % self.model_dir, ep, total_it)