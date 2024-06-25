import torch
from monai.metrics import DiceMetric, SurfaceDistanceMetric
import torch.nn.functional as F


def dice(labels, pred, n_class):
    dice_metric = DiceMetric(reduction="mean_batch")
    compact_pred = torch.argmax(pred, dim=1).unsqueeze(1)
    compact_pred_oh = F.one_hot(compact_pred.long().squeeze(1), n_class).permute(0, 3, 1, 2)

    labels_oh = F.one_hot(labels.long().squeeze(1), n_class).permute(0, 3, 1, 2)
    
    dice_metric(y_pred=compact_pred_oh, y=labels_oh)
    metric = dice_metric.aggregate()
    dice_metric.reset()

    return metric.detach().cpu().numpy()

def assd(labels, pred, n_class, pixdim):
    
    assd_metric = SurfaceDistanceMetric(reduction="mean_batch", symmetric=True)

    compact_pred = torch.argmax(pred, dim=1).unsqueeze(1)
    compact_pred_oh = F.one_hot(compact_pred.long().squeeze(1), n_class).permute(0, 3, 1, 2)

    labels_oh = F.one_hot(labels.long().squeeze(1), n_class).permute(0, 3, 1, 2)

    assd_metric(y_pred=compact_pred_oh, y=labels_oh)
    metric = assd_metric.aggregate()
    assd_metric.reset()

    return metric.detach().cpu().numpy() * pixdim 

def get_labels(pred):
    match pred:
        case "MYO":
            labels = [1, 0, 0, 0, 0, 0, 0]
            n_classes = 2
        case "LV":
            labels = [0, 0, 1, 0, 0, 0, 0]
            n_classes = 2
        case "RV":
            labels = [0, 0, 0, 0, 1, 0, 0]
            n_classes = 2
        case "Liver":
            labels = [1, 0, 0, 0]
            n_classes = 2
        case "MYO_LV_RV":
            labels = [1, 0, 2, 0, 3, 0, 0]
            n_classes = 4
        case _:
            labels = [1, 1, 1, 1, 1, 0, 0]
            n_classes = 6
    
    return labels, n_classes