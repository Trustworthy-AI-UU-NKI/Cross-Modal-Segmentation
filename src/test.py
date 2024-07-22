import torch
import os
import argparse
import logging
from torch.utils.data import DataLoader
from data.mmwhs_dataloader import MMWHS_single
from data.chaos_dataloader import CHAOS_single
from models.crosscompcsd import CrossCSD
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
import pytorch_lightning as pl
import glob
import numpy as np

from eval import test_vmfnet_mm
from utils import *




def get_args():
    usage_text = ()
    parser = argparse.ArgumentParser(description=usage_text)
    parser.add_argument('--seed', default=42, type=int,help='Seed to use for reproducing results') # --> their default was 14
    parser.add_argument('--bs', type= int, default=1, help='Batsh size')
    parser.add_argument('--cp', type=str, default='checkpoints/', help='The name of the checkpoints.')
    parser.add_argument('--name', type=str, default='test', help='The name of this run.')
    parser.add_argument('--learning_rate', type=float, default='0.0001', help='The learning rate for model training')
    parser.add_argument('--weight_init', type=str, default="xavier", help='Weight initialization method')
    parser.add_argument('--k2', type=int,  default=10, help='Check decay learning')
    parser.add_argument('--vc_num', type=int,  default=10, help='Kernel/distributions amount')
    parser.add_argument('--data_dir',  type=str, default='../data/MMWHS/CT_withGT_proc/', help='The name of the target data dir.')
    parser.add_argument('--k_folds', type= int, default=5, help='Cross validation')
    parser.add_argument('--pred', type=str, default='MYO', help='Segmentation task')
    parser.add_argument('--data_type', type=str, default="MMWHS") #MMWHS, RetinalVessel

    return parser.parse_args()


def test_net(save_dir, test_loader, writer, device, num_classes, fold):
    pretrained_model = glob.glob(os.path.join(save_dir, "*.pth"))

    if pretrained_model == []:
        print("no pretrained model found!")
        quit()
    else:
        model_file = pretrained_model[0]

    # Get pretrained model
    print("Loading model: ", model_file)
    model = CrossCSD(args, device, 1, num_classes, vMF_kappa=30, fold_nr=fold)
    model.to(device)
    model.resume(model_file)
    model.eval()

    # Evaluate model
    assd, dsc_0, dsc_1, images_dict, visuals_dict = test_vmfnet_mm(model, test_loader, device)

    # Show test metrics and images in Tensorboard
    writer.add_scalar(f'Test_metrics/assd', assd, 0)
    writer.add_scalar(f'Test_metrics/dsc_0', dsc_0, 0)
    writer.add_scalar(f'Test_metrics/dsc_1', dsc_1, 0)
    
    for key, value in images_dict.items():
        writer.add_images(f'Test_images/{key}', value, 0, dataformats='NCHW')
    
    for key, value in visuals_dict.items():
        for i in range(args.vc_num):
            writer.add_images(f'Test_visuals/{key}_{i+1}', value[:,i,:,:].unsqueeze(1), 0, dataformats='NCHW')

    return assd, dsc_0, dsc_1


def test_k_folds(args, labels, num_classes, device, dataset_type):
    cases = range(0,20)
    kf = KFold(n_splits=args.k_folds, shuffle=True)
    fold = 0
    dsc_scores_BG = []
    dsc_scores = []
    assd_scores = []

    # Test model on all folds
    for fold_train_val, fold_test in kf.split(cases):
        dir_checkpoint = os.path.join(args.cp, args.name)
        save_dir = os.path.join(dir_checkpoint, f'fold_{fold}')
        os.makedirs(save_dir, exist_ok=True)
        log_dir = os.path.join('logs', os.path.join(args.name, 'fold_{}'.format(fold)))
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
    
        print("Loading test data")
        dataset_test = dataset_type(args.data_dir, fold_test, labels) 
        test_loader = DataLoader(dataset_test, batch_size=1, num_workers=4)
        assd, dsc_0, dsc_1 = test_net(save_dir, test_loader, writer, device, num_classes, fold)
        fold += 1

        dsc_scores.append(dsc_1.cpu())
        dsc_scores_BG.append(dsc_0.cpu())
        assd_scores.append(assd)

    # Print all test metrics 
    dsc_scores_BG = np.array(dsc_scores_BG)
    mean_dsc_BG = np.mean(dsc_scores_BG)
    std_dsc_BG = np.std(dsc_scores_BG)
    print("FINAL RESULTS Background")
    print("DSC_0: ", dsc_scores_BG)
    print(f"Mean DSC_0: {mean_dsc_BG}, Std DSC_0: {std_dsc_BG}")

    dsc_scores = np.array(dsc_scores)
    mean_dsc = np.mean(dsc_scores)
    std_dsc = np.std(dsc_scores)
    print("FINAL RESULTS TRUE DSC")
    print("DSC_1: ", dsc_scores)
    print(f"Mean DSC_1: {mean_dsc}, Std DSC_1: {std_dsc}")

    assd_scores = np.array(assd_scores)
    mean_assd = np.mean(assd_scores)
    std_assd = np.std(assd_scores)
    print("ASSD: ", assd_scores)
    print(f"Mean ASSD: {mean_assd}, Std ASSD: {std_assd}")
        
def main(args):
    set_seed(args.seed)
    labels, num_classes = get_labels(args.pred)

    # Set right dataset
    if args.data_type == "MMWHS":
        dataset_type = MMWHS_single
    elif args.data_type == "CHAOS":
        dataset_type = CHAOS_single
    else:
        raise ValueError(f"Data type {args.data_type} not supported")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Test
    test_k_folds(args, labels, num_classes, device, dataset_type)



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    main(args)
