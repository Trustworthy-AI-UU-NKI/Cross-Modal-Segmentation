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
import torch.nn as nn


class DDFNetL(pl.LightningModule):
    def __init__(self, bs, epochs, loss, lr, modality, pred):
        super().__init__()
        self.dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        self.model = DDFNet(
            
        )
        self.save_hyperparameters()
        self.lr = lr
        self.bs = bs
        self.epochs = epochs
        self.modality = modality
        self.pred = pred
        self.post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

        
        self.loss_function = BCEWithLogitsLoss()
        

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


class DDFNet(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Sequence[int] | int = 3,
        up_kernel_size: Sequence[int] | int = 3,
        num_res_units: int = 0,
        act: tuple | str = Act.PRELU,
        norm: tuple | str = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
    ) -> None:
        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if isinstance(kernel_size, Sequence) and len(kernel_size) != spatial_dims:
            raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence) and len(up_kernel_size) != spatial_dims:
            raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x


