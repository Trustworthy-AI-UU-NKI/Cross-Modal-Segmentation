from monai.metrics import DiceMetric, SurfaceDistanceMetric, HausdorffDistanceMetric
import torch
import torch.nn.functional as F
import numpy as np


def dice(labels, pred, n_class):
    # Initialize the DiceMetric object
    # set reduction to 'none' to get the score for each class separately
    dice_metric = DiceMetric(reduction="mean_batch")
    compact_pred = torch.argmax(pred, dim=1).unsqueeze(1)
    compact_pred_oh = F.one_hot(compact_pred.long().squeeze(1), n_class).permute(0, 3, 1, 2)

    labels_oh = F.one_hot(labels.long().squeeze(1), n_class).permute(0, 3, 1, 2)
    # print(labels_oh.shape)

    # Compute the Dice score
    dice_metric(y_pred=compact_pred_oh, y=labels_oh)
    metric = dice_metric.aggregate()
    dice_metric.reset()

    return metric

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def assd(labels, pred, n_class, pix_dim):
    
    assd_metric = SurfaceDistanceMetric(reduction="mean_batch", symmetric=True)#_batch")
    # print(compact_pred.shape)
    # print(labels.shape)
    compact_pred = torch.argmax(pred, dim=1).unsqueeze(1)
    compact_pred_oh = F.one_hot(compact_pred.long().squeeze(1), n_class).permute(0, 3, 1, 2)

    labels_oh = F.one_hot(labels.long().squeeze(1), n_class).permute(0, 3, 1, 2)
    # print(labels_oh.shape)

    # Compute the hd score
    assd_metric(y_pred=compact_pred_oh, y=labels_oh)
    metric = assd_metric.aggregate()
    assd_metric.reset()

    return metric * pix_dim


def get_labels(pred):
    # match case statement
    match pred:
        case "MYO":
            labels = [1, 0, 0, 0, 0]
            n_classes = 2
        case "LV":
            labels = [0, 0, 1, 0, 0]
            n_classes = 2
        case "RV":
            labels = [0, 0, 0, 0, 1]
            n_classes = 2
        case "liver":
            labels = [1, 0, 0, 0]
            n_classes = 2
        case "RK":
            labels = [0, 1, 0, 0]
            n_classes = 2
        case "LK":
            labels = [0, 0, 1, 0]
            n_classes = 2
        case "Spleen":
            labels = [0, 0, 0, 1]
            n_classes = 2
        case _:
            print("Class prediction not implemented yet")
    
    return labels, n_classes