import torch
from monai.metrics import DiceMetric
import torch.nn.functional as F
import numpy as np

# evaluation functions heres

def dice(compact_pred, labels, n_class):
    # Initialize the DiceMetric object
    # set reduction to 'none' to get the score for each class separately
    dice_metric = DiceMetric(reduction="mean")

    compact_pred_oh = F.one_hot(compact_pred.squeeze(1), n_class).permute(0, 3, 1, 2)
    labels_oh = F.one_hot(labels.squeeze(1), n_class).permute(0, 3, 1, 2)

    # Compute the Dice score
    dice_metric(y_pred=compact_pred_oh, y=labels_oh)
    metric = dice_metric.aggregate().item()
    dice_metric.reset()

    return metric