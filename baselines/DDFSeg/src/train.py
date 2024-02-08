import torch
from torch.utils.data import DataLoader
from dataset import MMWHS
from saver import Saver
from model import DDFSeg
import argparse
import sys
import pytorch_lightning as pl
import numpy as np

from sklearn.model_selection import KFold



def train(train_loader, val_loader, fold, device, args):
    # model
    print('\n--- load model ---')
    model = DDFSeg(args)
    model.setgpu(device)

    # how with different folds? 
    # if args.resume

    # saver for display and output
    saver = Saver(fold, args)


    # train
    print('\n--- train ---')




    total_it = 0
    max_it = 500000 # aanpassen
    for ep in range(args.epochs):
        print("In the epoch ", ep)

        if ep < 5:
            loss_f_weight_value = 0.0
        elif ep < 7:
            loss_f_weight_value = 0.1 * (ep - 4.0) / (7.0 - 4.0)
        else:
            loss_f_weight_value = 0.1

        for it, (images_a, labels_a, images_b, ) in enumerate(train_loader):

            # input data
            images_a = images_a.to(device).detach()
            images_b = images_b.to(device).detach()
            labels_a = labels_a.to(device).detach()

            # update model
            # fake_B_temp is from A and fake_A_temp is from B
            model.update(images_a, images_b, labels_a, loss_f_weight_value)

            # I think this is updating the learning rates --> scheduler
            # summary_str_gan, summary_str_seg, summary_str_lossf = sess.run([self.lr_gan_summ, self.lr_seg_summ, self.loss_f_weight_summ],
            #                  feed_dict={
            #                      self.learning_rate: curr_lr,
            #                      self.learning_rate_seg: curr_lr_seg,
            #                      self.loss_f_weight: loss_f_weight_value,
            #                  })

            #         writer.add_summary(summary_str_gan, epoch * max_inter + i)
            #         writer.add_summary(summary_str_seg, epoch * max_inter + i)
            #         writer.add_summary(summary_str_lossf, epoch * max_inter + i)
            #         writer.flush()
        
            model.update_num_fake_inputs()

            # save losses to display file
            saver.write_display(total_it, model)

            print('total_it: %d (ep %d, it %d), lr %08f' % (total_it, ep, it, model.gen_opt.param_groups[0]['lr']))
            total_it += 1
            if total_it >= max_it:
                saver.write_model(-1, model)
            break
    
    #     #save the best val model
    #     print("Processing val {}".format(i))
    #     max_inter_val = np.uint16(np.floor(len(rows_t_val) / BATCH_SIZE))
    #     val_dice_list = []
    #     for m in range(0, max_inter_val):
    #         np_b_mean = sess.run([self.dice_b_mean], feed_dict = {self.input_b: inputs_val['images_j_val'],self.gt_b:inputs_val['gts_j_val'], self.is_training: False, self.keep_rate: 1.0})
    #         val_dice_list.append(np_b_mean)
    #     val_dice_mean=np.mean(val_dice_list)
    #     print(len(val_dice_list), val_dice_mean)
    #     if(val_dice_mean > val_dice):
    #         val_dice = val_dice_mean
    #         saver.save(sess, os.path.join(self._output_dir, "pch"), global_step=cnt)

    #     sess.run(tf.assign(self.global_step, epoch + 1))

    # decay learning rate
    if args.n_ep_decay > -1:
        model.update_lr()

    # save result image
    saver.write_img(ep, model)

    # Save network weights
    saver.write_model(ep, total_it, model)

    # test here()


def main(args):
    pl.seed_everything(args.seed)

    if args.pred == "MYO":
        labels = [1, 0, 0, 0, 0, 0, 0]
    else:
        labels = [1, 1, 1, 1, 1, 1, 1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_cases = range(0,20)
    target_cases = range(0,18)
    kf = KFold(n_splits=args.k_folds, shuffle=True)
    fold = 0
    for fold_source_train, fold_source_val, fold_target_train, fold_target_val  in zip(kf.split(source_cases), kf.split(target_cases)):
        dataset_train = MMWHS(args, labels, fold_target_train, fold_source_train)
        train_loader = DataLoader(dataset_train, batch_size=args.bs, shuffle=True, num_workers=4)
        dataset_val = MMWHS(args, labels, fold_target_val, fold_source_val) 
        val_loader = DataLoader(dataset_val, batch_size=args.bs, num_workers=4)
        
        train(train_loader, val_loader, fold, device, args) 
        fold += 1
        


  
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train the DDFSeg model on the MM-WHS dataset')
    
    # data loader related
    parser.add_argument('--data_dir', type=str, default='../../../data/preprocessed', help='path of data to domain 1')
    parser.add_argument('--phase', type=str, default='train', help='phase for dataloading')
    parser.add_argument('--resume', type=str, default=None, help='specified the dir of saved models for resume the training')

    # ouptput related
    parser.add_argument('--name', type=str, default='trial', help='folder name to save outputs')
    parser.add_argument('--display_dir', type=str, default='../logs', help='path for saving display results')
    parser.add_argument('--result_dir', type=str, default='../results', help='path for saving result images and models')
    parser.add_argument('--display_freq', type=int, default=100, help='freq (iteration) of display on tensorboard')
    parser.add_argument('--model_save_freq', type=int, default=5, help='freq (epoch) of saving models')

    # training related --> Aanpassen
    parser.add_argument('--pred', default='MYO', type=str,help='Prediction of which label') # MYO, LV, RV, MYO_RV, MYO_LV_RV
    parser.add_argument('--modality', default="CT", type=str, help='modality that is annotated - source domain') # or MRI
    parser.add_argument('--k_folds', default=6, type=int, help='Number of folds for K-Fold Cross-Validation')
    parser.add_argument('--epochs', default=10, type=int, help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,help='Seed to use for reproducing results')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--lr67', default=1e-3, type=float, help='Learning rate for the segmentor')
    parser.add_argument('--lr5', default=1e-3, type=float, help='Learning rate for the zero_loss')   
    parser.add_argument('--lr_A', default=1e-3, type=float, help='Learning rate (lambda A)')
    parser.add_argument('--lr_B', default=1e-3, type=float, help='Learning rate (lambda B)')
    parser.add_argument('--bs', default=4, type=int,help='batch_size')
    parser.add_argument('--num_cls', default=4, type=int, help='Number of classes')
    parser.add_argument('--keep_rate', default=0.75, type=float, help='Keep rate for dropout')
    parser.add_argument('--resolution', default=256, type=int, help='Resolution of input images')
    parser.add_argument('--skip', default=True, type=bool, help='Skip connection in the generator')
    
    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)

    main(args)