from monai.metrics import DiceMetric, SurfaceDistanceMetric
import torch
import torch.nn.functional as F
import numpy as np

# Set seed for reproducible results
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Compute DSC
def dice(labels, pred, n_class):
    # Initialize the DiceMetric object with mean batch to keep seperate dsc for classes
    dice_metric = DiceMetric(reduction="mean_batch")

    # Predictions are in logit form (of size [B, C, H, W]) where C = nr of classes
    compact_pred = torch.argmax(pred, dim=1).unsqueeze(1)
    # Get onehot encoding (oh)
    compact_pred_oh = F.one_hot(compact_pred.long().squeeze(1), n_class).permute(0, 3, 1, 2)
    labels_oh = F.one_hot(labels.long().squeeze(1), n_class).permute(0, 3, 1, 2)

    # Compute the DSC
    dice_metric(y_pred=compact_pred_oh, y=labels_oh)
    metric = dice_metric.aggregate()
    dice_metric.reset()

    return metric

# COmpute ASSD
def assd(labels, pred, n_class, pix_dim):
    # Initialize the ASSD object
    assd_metric = SurfaceDistanceMetric(reduction="mean_batch", symmetric=True)

    # Predictions are in logit form (of size [B, C, H, W]) where C = nr of classes
    compact_pred = torch.argmax(pred, dim=1).unsqueeze(1)
    # Get onehot encoding (oh)
    compact_pred_oh = F.one_hot(compact_pred.long().squeeze(1), n_class).permute(0, 3, 1, 2)
    labels_oh = F.one_hot(labels.long().squeeze(1), n_class).permute(0, 3, 1, 2)

    # Compute the assd 
    assd_metric(y_pred=compact_pred_oh, y=labels_oh)
    metric = assd_metric.aggregate()
    assd_metric.reset()

    # assd is in pixels --> need to convert to mm
    # NB: Can only when B = 1!!! (During testing!)
    return metric * pix_dim

# Get labels and nr_classes according to the task
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
        case "Liver":
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