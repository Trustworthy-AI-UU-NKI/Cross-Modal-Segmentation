import torch
import os
import argparse
from tqdm import tqdm
import logging
from torch.utils.data import DataLoader
from mmwhs_dataloader import MMWHS
from chaos_dataloader import CHAOS
from models.crosscompcsd import CrossCSD
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
import pytorch_lightning as pl
import glob
import numpy as np

from eval import eval_vmfnet_mm
from utils import *



def get_args():
    usage_text = (
        "vMFNet Pytorch Implementation"
        "Usage:  python train.py [options],"
        "   with [options]:"
    )
    parser = argparse.ArgumentParser(description=usage_text)
    #training details
    parser.add_argument('--epochs', type= int, default=200, help='Number of epochs')
    parser.add_argument('--seed', default=42, type=int,help='Seed to use for reproducing results') # --> their default was 14
    parser.add_argument('--bs', type= int, default=4, help='Number of inputs per batch')
    parser.add_argument('--cp', type=str, default='checkpoints/', help='The name of the checkpoints.')


    parser.add_argument('--name', type=str, default='test_MM_KLD', help='The name of this train/test. Used when storing information.')
    parser.add_argument('-lr','--learning_rate', type=float, default='0.0001', help='The learning rate for model training')
    parser.add_argument('-wi','--weight_init', type=str, default="xavier", help='Weight initialization method, or path to weights file (for fine-tuning or continuing training)')
    # parser.add_argument('--k1', type=int,  default=40, help='When the learning rate starts decaying')
    parser.add_argument('--k2', type=int,  default=10, help='Check decay learning')
    parser.add_argument('--layer', type=int,  default=8, help='layer from which the deepf eatures are obtained')
    parser.add_argument('--vc_num', type=int,  default=12, help='Kernel/distributions amount')
    parser.add_argument('--data_dir_t',  type=str, default='../data/other/CT_withGT_proc/', help='The name of the data dir.')
    parser.add_argument('--data_dir_s',  type=str, default='../data/other/MR_withGT_proc/', help='The name of the data dir.')
    parser.add_argument('--k_folds', type= int, default=5, help='Cross validation')
    parser.add_argument('--pred', type=str, default='MYO', help='Segmentation task')
   
    parser.add_argument('--vc_num_seg', type=int,  default=12, help='Kernel/distributions amount as input for the segmentation model')
    parser.add_argument('--init', type=str, default='pretrain', help='Initialization method') # pretrain (original), xavier, cross.
    parser.add_argument('--norm', type=str, default="Batch")
    parser.add_argument('--encoder_type', type=str, default="unet")

    parser.add_argument('--true_clu_loss', action='store_true')  # default is optimization with fake loss
    parser.add_argument('--pretrain', action='store_true')  # default is optimization with fake loss

    parser.add_argument('--content_disc', action='store_true')  # with or without content discriminator
    parser.add_argument('--data_type', type=str, default="MMWHS") #MMWHS, RetinalVessel
    

    return parser.parse_args()


def test_net(save_dir, val_loader, writer, device, num_classes, fold):
    pretrained_model = glob.glob(os.path.join(save_dir, "*.pth"))

    if pretrained_model == []:
        print("no pretrained model found!")
        quit()
    else:
        model_file = pretrained_model[0]

    print("Loading model: ", model_file)
    model = CrossCSD(args, device, 1, num_classes, vMF_kappa=30, fold_nr=fold)
    model.to(device)
    model.resume(model_file)
    model.eval()

    metrics_dict, images_dict, visuals_dict, lpips_metric = eval_vmfnet_mm(model, val_loader, device)

    writer.add_scalar(f'Test_metrics/lpips_target', lpips_metric, 0)

    for key, value in metrics_dict.items():
        writer.add_scalar(f'Test_metrics/{key}', value, 0)
    
    for key, value in images_dict.items():
        writer.add_images(f'Test_images/{key}', value, 0, dataformats='NCHW')
    
    for key, value in visuals_dict.items():
        for i in range(args.vc_num):
            writer.add_images(f'Test_visuals/{key}_{i+1}', value[:,i,:,:].unsqueeze(1), 0, dataformats='NCHW')

    return metrics_dict



def train_net(train_loader, val_loader, fold, device, args, len_train_data, num_classes, save_dir, writer):

    model = CrossCSD(args, device, 1, num_classes, vMF_kappa=30, fold_nr=fold)
    model.to(device)

    best_score = 0

    global_step = 0
    print("Training fold: ", fold)

    for epoch in range(args.epochs):
        model.train()
        with tqdm(total=len_train_data, desc=f'Epoch {epoch + 1}/{args.epochs}', unit='img') as pbar:
            model.train()
            for img_s, label_s, img_t, label_t in train_loader:
                img_s = img_s.to(device)
                img_t = img_t.to(device)
                label_s = label_s.to(device)

                losses = model.update(img_s, label_s, img_t, epoch)
                pbar.set_postfix(**{'loss (batch)': losses["loss/total"]})
                pbar.update(img_t.shape[0])

                for key, value in losses.items():
                    writer.add_scalar(f'{key}', value, global_step)

                global_step += 1
    
            
            if epoch % 5 == 0:
                metrics_dict, images_dict, visuals_dict, lpips_metric = eval_vmfnet_mm(model, val_loader, device)

                writer.add_scalar(f'Val_metrics/lpips_target', lpips_metric, epoch)
                new_metric = (1-lpips_metric) + metrics_dict["Target/DSC_fake"]
                writer.add_scalar(f'Val_metrics/lpips_dscf', new_metric , epoch)
                model.scheduler_source.step(new_metric)
                model.scheduler_target.step(new_metric)

                for key, value in metrics_dict.items():
                    writer.add_scalar(f'Val_metrics/{key}', value, epoch)
                
                if epoch % 20 == 0:
                    for key, value in images_dict.items():
                        writer.add_images(f'Val_images/{key}', value, epoch, dataformats='NCHW')
                    
                    for key, value in visuals_dict.items():
                        for i in range(args.vc_num):
                            writer.add_images(f'Val_visuals/{key}_{i+1}', value[:,i,:,:].unsqueeze(1), epoch, dataformats='NCHW')

                #on which parameter are we going to choose the best model?? 
                if new_metric > best_score:
                    best_score = new_metric
                    print("Epoch checkpoint")
                    
                    print(f"--- Remove old model before saving new one ---")
                    existing_model_files = glob.glob(f"{save_dir}/*.pth")
                    for file_path in existing_model_files:
                        try:
                            os.remove(file_path)
                            print(f"Deleted old model file: {file_path}")
                        except OSError as e:
                            print(f"Error deleting file {file_path}: {e}")
                    
                    torch.save(model.state_dict(), os.path.join(save_dir, f'CP_epoch_{epoch}_model_{best_score}.pth'))

                    logging.info('Checkpoint saved !')
                    
            
    writer.close()


def train_k_folds(args, labels, num_classes, device, dataset_type):
    cases = range(0,20)
    kf = KFold(n_splits=args.k_folds, shuffle=True)
    fold = 0
    dsc_scores_BG = []
    dsc_scores = []
    dsc_scores_prev = []
    assd_scores = []

    # for fold_train, fold_test_val in kf.split(cases):
    for fold_train_val, fold_test in kf.split(cases):
        print("Training fold: ", fold)
        dir_checkpoint = os.path.join(args.cp, args.name)
        save_dir = os.path.join(dir_checkpoint, f'fold_{fold}')
        os.makedirs(save_dir, exist_ok=True)
        log_dir = os.path.join('logs', os.path.join(args.name, 'fold_{}'.format(fold)))
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        
        print('Train fold:', fold_train_val)
        print('Val fold:', fold_test)
        print("loading train data")
        dataset_train = dataset_type(args, labels, fold_train_val)
        train_loader = DataLoader(dataset_train, batch_size=args.bs, shuffle=True, num_workers=4)
        print("loading val data")
        dataset_val = dataset_type(args, labels, fold_test) 
        val_loader = DataLoader(dataset_val, batch_size=1, num_workers=4)
        len_train_data = dataset_train.__len__()

        train_net(train_loader, val_loader, fold, device, args, len_train_data, num_classes, save_dir, writer) 
        print("loading test data")
        dataset_test = dataset_type(args, labels, fold_test) 
        test_loader = DataLoader(dataset_test, batch_size=1, num_workers=4)
        metrics_fold = test_net(save_dir, test_loader, writer, device, num_classes, fold)
        fold += 1

        dsc_scores.append(metrics_fold['Target/DSC'].cpu())
        dsc_scores_BG.append(metrics_fold['Target/DSC_0'].cpu())
        dsc_scores_prev.append(metrics_fold['Target/DSC_1'].cpu())
        assd_scores.append(metrics_fold['Target/assd'])

    dsc_scores_BG = np.array(dsc_scores_BG)
    mean_dsc_BG = np.mean(dsc_scores_BG)
    std_dsc_BG = np.std(dsc_scores_BG)
    print("FINAL RESULTS BG")
    print("DSC_0: ", dsc_scores_BG)
    print(f"Mean DSC_0: {mean_dsc_BG}, Std DSC_0: {std_dsc_BG}")

    dsc_scores = np.array(dsc_scores)
    mean_dsc = np.mean(dsc_scores)
    std_dsc = np.std(dsc_scores)
    print("FINAL RESULTS TRUE DSC")
    print("DSC_1: ", dsc_scores)
    print(f"Mean DSC_1: {mean_dsc}, Std DSC_1: {std_dsc}")

    dsc_scores_prev = np.array(dsc_scores_prev)
    mean_dsc_pre = np.mean(dsc_scores_prev)
    std_dsc_pre = np.std(dsc_scores_prev)
    print("FINAL RESULTS DSC WITHOUT ADJUSTMENT")
    print("DSC_1: ", dsc_scores_prev)
    print(f"Mean DSC_1: {mean_dsc_pre}, Std DSC_1: {std_dsc_pre}")

    assd_scores = np.array(assd_scores)
    mean_assd = np.mean(assd_scores)
    std_assd = np.std(assd_scores)
    print("ASSD: ", assd_scores)
    print(f"Mean ASSD: {mean_assd}, Std ASSD: {std_assd}")
        
def main(args):
    #pl.seed_everything(args.seed)
    set_seed(args.seed)
    labels, num_classes = get_labels(args.pred)
    # MMWHS Dataset
    if args.data_type == "MMWHS":
        dataset_type = MMWHS
    elif args.data_type == "chaos":
        dataset_type = CHAOS
    else:
        raise ValueError(f"Data type {args.data_type} not supported")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    train_k_folds(args, labels, num_classes, device, dataset_type)



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    main(args)
