import torch
import pytorch_lightning as pl
from torch.nn import BCEWithLogitsLoss
from monai.data import  decollate_batch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (  
    Activations,
    AsDiscrete,
    Compose,
)

from types import SimpleNamespace
from monai.losses import DiceLoss

class DiceBCELoss(torch.nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()
        self.dice_loss = DiceLoss(sigmoid=True)
        self.bce_loss = BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        dice_loss = self.dice_loss(inputs, targets)
        bce_loss = self.bce_loss(inputs, targets)
        return dice_loss + bce_loss


class UNetL(pl.LightningModule):
    def __init__(self, bs, epochs, loss, lr, modality, pred):
        super().__init__()
        self.dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        self.model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256), #(64, 128, 256, 512, 1024)
            strides=(2, 2, 2, 2),
            num_res_units=2, # 0
            # act=RELU
        )
        self.save_hyperparameters()
        self.lr = lr
        self.bs = bs
        self.epochs = epochs
        self.modality = modality
        self.pred = pred
        self.post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

        if loss == "BCE":
            self.loss_function = BCEWithLogitsLoss()
        elif loss == "Dice":
            self.loss_function = DiceLoss(sigmoid=True)
        elif loss == "DiceBCE":
            self.loss_function = DiceBCELoss()
        else:
            raise ValueError("Loss not supported")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch["img"].to(self.device), batch["seg"].to(self.device)
        outputs = self.model(inputs)
        loss = self.loss_function(outputs, labels)
        self.log("Train loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch["img"].to(self.device), batch["seg"].to(self.device)
        val_outputs = self.model(inputs)

        val_outputs_onehot = [self.post_trans(i) for i in decollate_batch(val_outputs)]
        self.dice_metric(y_pred=val_outputs_onehot, y=labels)

        loss = self.loss_function(val_outputs, labels)

        # Log only the last validation images to TensorBoard
        if batch_idx == len(self.val_dataloader()) - 1:
            self.log_images(inputs, labels, val_outputs_onehot, "val", self.global_step)
        
        self.log("Validation loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        metric = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        self.log("Validation Mean Dice", metric, on_step=False, on_epoch=True, prog_bar=True)
        return {"Validation Mean Dice": metric}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def log_images(self, inputs, labels, outputs, tag, step):
        for i, (input_img, label_img, output_img) in enumerate(zip(inputs, labels, outputs)):
            self.logger.experiment.add_image(f"{tag}_input_{i}", input_img, step)
            self.logger.experiment.add_image(f"{tag}_label_{i}", label_img, step)
            self.logger.experiment.add_image(f"{tag}_output_{i}", output_img, step)
            
    
    def test_step(self, batch, batch_idx):
        inputs, labels = batch["img"].to(self.device), batch["seg"].to(self.device)
        outputs = self.model(inputs)
        loss = self.loss_function(outputs, labels)

        outputs_onehot = [self.post_trans(i) for i in decollate_batch(outputs)]
        self.dice_metric(y_pred=outputs_onehot, y=labels)

        if batch_idx == len(self.test_dataloader()) - 1:
            self.log_images(inputs, labels, outputs_onehot, "test", self.global_step)

        # Add any additional testing/evaluation logic here
        return loss

    def on_test_epoch_end(self):
        metric = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        self.log("Test Mean Dice", metric)
        return {"Test Mean Dice": metric}




