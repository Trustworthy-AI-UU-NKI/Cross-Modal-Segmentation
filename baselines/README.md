# Baselines

This folder contains the implementation for all the baselines; No Adaptation UNet, Full Supervision UNet, vMFnet, DRIT++ with UNet and with ResUNet and DDFSeg. Below, we explain how to run all baselines. 

## Code structure
- The `checkpoints` folder contains all checkpoints and tensorboard logging files from the UNet-NA, UNet-FS and the second stage (after DRIT) UNet and ResUNet models.
- The `DDFSeg` folder contains all source code for the DDFSeg network. 
- The `DRIT` folder contains all source code for the DRIT model.
- The `vMFNet` folder contains all source code for the vMFNet network.

## UNet-NA
To train and test this baseline for $MRI \rightarrow CT$, run the code below. For the other direction, please change the arguments `--data_dir`, `--data_dir_test`, and `--name` accordingly. For another segmentation task, you can change `--pred` to LV or RV. 

```
source activate gpu_env
python run_unet.py -- --pred MYO --name UNet_trained_on_MRI --epochs 200 --bs 4 --lr 0.0001 --data_dir ../data/MMWHS/MR_withGT_proc/ --data_dir_test ../data/MMWHS/CT_withGT_proc/ --k_folds 5 --data_type MMWHS

python evaluate_unet.py -- --pred MYO --name UNet_trained_on_MRI --bs 1 --data_dir_test ../data/MMWHS/CT_withGT_proc/ --k_folds 5 --data_type MMWHS
```

If you want to run this baseline on the CHAOS dataset with $T1 \rightarrow T2$ for cross-modal liver segmentation,  run the code below. For the other direction, please change the arguments `--data_dir`, `--data_dir_test`, and `--name` accordingly.

```
source activate gpu_env
python run_unet.py -- --pred Liver --name UNet_trained_on_T1 --epochs 200 --bs 4 --lr 0.0001 --data_dir ../data/CHAOS/T1/ --data_dir_test ../data/CHAOS/T2/ --k_folds 5 --data_type CHAOS

python evaluate_unet.py -- --pred Liver --name UNet_trained_on_T1 --bs 1 --data_dir_test ../data/CHAOS/T2/ --k_folds 5 --data_type CHAOS
```

## UNet-FS
To train and test this baseline for $MRI \rightarrow CT$, run the code below. For the other direction, please change the arguments `--data_dir`, `--data_dir_test`, and `--name` accordingly. For another segmentation task, you can change `--pred` to LV or RV. 

```
source activate gpu_env
python run_unet.py -- --pred MYO --name UNet_trained_on_CT --epochs 200 --bs 4 --lr 0.0001 --data_dir ../data/MMWHS/CT_withGT_proc/ --data_dir_test ../data/MMWHS/CT_withGT_proc/ --k_folds 5 --data_type MMWHS

python evaluate_unet.py -- --pred MYO --name UNet_trained_on_CT --bs 1 --data_dir_test ../data/MMWHS/CT_withGT_proc/ --k_folds 5 --data_type MMWHS
```

If you want to run this baseline on the CHAOS dataset with $T1 \rightarrow T2$ for cross-modal liver segmentation,  run the code below. For the other direction, please change the arguments `--data_dir`, `--data_dir_test`, and `--name` accordingly.

```
source activate gpu_env
python run_unet.py -- --pred Liver --name UNet_trained_on_T2 --epochs 200 --bs 4 --lr 0.0001 --data_dir ../data/CHAOS/T2/ --data_dir_test ../data/CHAOS/T2/ --k_folds 5 --data_type CHAOS

python evaluate_unet.py -- --pred Liver --name UNet_trained_on_T2 --bs 1 --data_dir_test ../data/CHAOS/T2/ --k_folds 5 --data_type CHAOS
```


## vMFNet
To obtain this code, please clone [this repo](https://github.com/aeijpe/vMFNet) in the baseline folder.
To train and test this baseline for $MRI \rightarrow CT$, run the code below. For the other direction, please change the arguments `--data_dir` and `--name` accordingly. For another segmentation task, you can change `--pred` to LV or RV

```
source activate gpu_env
cd vMFNet
python train.py --name single_MRI_MYO --data_dir ../dta/MMWHS/MR_withGT_proc/ --pred MYO --data_type MMWHS
python test.py --name single_MRI_MYO --data_dir ../data/MMWHS/CT_withGT_proc/ --pred MYO --data_type MMWHS
```

To train and test this baseline for $T1 \rightarrow T2$, run the code below. For the other direction, please change the arguments `--data_dir` and `--name` accordingly.

```
source activate gpu_env
cd vMFNet
python train.py --name single_T1_Liver --data_dir ../data/CHAOS/T1/ --pred Liver --data_type CHAOS
python test.py --name single_T1_Liver --data_dir ../data/CHAOS/T2/ --pred Liver --data_type CHAOS
```



## DRIT++ with UNet and DRIT++ with Res UNet

To obtain the source code for the DRIT model, please clone [this repository](https://github.com/aeijpe/DRIT), into the `baselines` folder. For more information on the DRIT model, please check the README in the `DRIT` folder. These baselines (i.e., DRIT+UNet and DRIT+ResUNet), comprise two distinct models, an I2I translation model and a segmentation model. In the first stage, the I2I translation model DRIT++ is trained. Please run the following lines to train the DRIT model on the MMWHS or CHAOS dataset, respectively.

```
source activate gpu_env
cd DRIT/src
python train.py --name chaos_run_fold_0 --batch_size 2 --data_dir1 ../../../data/CHAOS/T1/ --data_dir2 ../../../data/CHAOS/T2/ --cases_folds 0 --data_type CHAOS
python train.py --name chaos_run_fold_1 --batch_size 2 --data_dir1 ../../../data/CHAOS/T1/ --data_dir2 ../../../data/CHAOS/T2/ --cases_folds 1 --data_type CHAOS 
python train.py --name chaos_run_fold_2 --batch_size 2 --data_dir1 ../../../data/CHAOS/T1/ --data_dir2 ../../../data/CHAOS/T2/ --cases_folds 2 --data_type CHAOS
python train.py --name chaos_run_fold_3 --batch_size 2 --data_dir1 ../../../data/CHAOS/T1/ --data_dir2 ../../../data/CHAOS/T2/ --cases_folds 3 --data_type CHAOS
python train.py --name chaos_run_fold_4 --batch_size 2 --data_dir1 ../../../data/CHAOS/T1/ --data_dir2 ../../../data/CHAOS/T2/ --cases_folds 4 --data_type CHAOS 
```

```
source activate gpu_env
cd DRIT/src
python train.py --name mmwhs_run_fold_0 --batch_size 2 --data_dir1 ../../../data/MMWHS/CT_withGT_proc/ --data_dir2 ../../../data/MMWHS/MR_withGT_proc/ --cases_folds 0 --data_type MMWHS
python train.py --name mmwhs_run_fold_1 --batch_size 2 --data_dir1 ../../../data/MMWHS/CT_withGT_proc/ --data_dir2 ../../../data/MMWHS/MR_withGT_proc/ --cases_folds 1 --data_type MMWHS 
python train.py --name mmwhs_run_fold_2 --batch_size 2 --data_dir1 ../../../data/MMWHS/CT_withGT_proc/ --data_dir2 ../../../data/MMWHS/MR_withGT_proc/ --cases_folds 2 --data_type MMWHS
python train.py --name mmwhs_run_fold_3 --batch_size 2 --data_dir1 ../../../data/MMWHS/CT_withGT_proc/ --data_dir2 ../../../data/MMWHS/MR_withGT_proc/ --cases_folds 3 --data_type MMWHS
python train.py --name mmwhs_run_fold_4 --batch_size 2 --data_dir1 ../../../data/MMWHS/CT_withGT_proc/ --data_dir2 ../../../data/MMWHS/MR_withGT_proc/ --cases_folds 4 --data_type MMWHS 
```

Then, we evaluated the model using the LPIPS metric, by running the following code for the MMWHS or CHAOS datasets, respectively.
```
python evaluate_model.py --data_type MMWHS --cases_fold 0 --data_dir1 ../../../data/MMWHS/CT_withGT_proc/ --data_dir2 ../../../data/MMWHS/MR_withGT_proc/
python evaluate_model.py --data_type MMWHS --cases_fold 1 --data_dir1 ../../../data/MMWHS/CT_withGT_proc/ --data_dir2 ../../../data/MMWHS/MR_withGT_proc/
python evaluate_model.py --data_type MMWHS --cases_fold 2 --data_dir1 ../../../data/MMWHS/CT_withGT_proc/ --data_dir2 ../../../data/MMWHS/MR_withGT_proc/
python evaluate_model.py --data_type MMWHS --cases_fold 3 --data_dir1 ../../../data/MMWHS/CT_withGT_proc/ --data_dir2 ../../../data/MMWHS/MR_withGT_proc/
python evaluate_model.py --data_type MMWHS --cases_fold 4 --data_dir1 ../../../data/MMWHS/CT_withGT_proc/ --data_dir2 ../../../data/MMWHS/MR_withGT_proc/
```

```
python evaluate_model.py --data_type CHAOS --cases_fold 0 --data_dir1 ../../../data/CHAOS/T1/ --data_dir2 ../../../data/CHAOS/T2/
python evaluate_model.py --data_type CHAOS --cases_fold 1 --data_dir1 ../../../data/CHAOS/T1/ --data_dir2 ../../../data/CHAOS/T2/
python evaluate_model.py --data_type CHAOS --cases_fold 2 --data_dir1 ../../../data/CHAOS/T1/ --data_dir2 ../../../data/CHAOS/T2/
python evaluate_model.py --data_type CHAOS --cases_fold 3 --data_dir1 ../../../data/CHAOS/T1/ --data_dir2 ../../../data/CHAOS/T2/
python evaluate_model.py --data_type CHAOS --cases_fold 4 --data_dir1 ../../../data/CHAOS/T1/ --data_dir2 ../../../data/CHAOS/T2/
```


After training and evaluation, the images of the source domain were translated to the target domain and coupled with the original ground truth annotations to create a synthetic, target dataset. For the translation, we choose the model per fold that had the lowest (==best) LPIPS. For more information on the evaluation, please see `../results/vis_results_DRIT.ipynb`. To create the synthetic MRI dataset, run the following code with the best model paths obtained from the evaluation in the `--resume` argument. To create the synthesic CT target data, change `--a2b` to `0` and the `--data_dir1`, `result_dir`, and `--resume` arguments accordingly. Similar, for the CHAOS dataset, where you also have to set `--data_type` to `CHAOS`.

```
python create_fake_img.py --resume ../results/run_fold_0/00699.pth --name run_fold_0 --a2b 1 --data_dir1 ../../../data/MMWHS/CT_withGT_proc/ --cases_folds 0 --result_dir ../../../data/other/fake_MR/ --data_type MMWHS
python create_fake_img.py --resume ../results/run_fold_1/00549.pth --name run_fold_1 --a2b 1 --data_dir1 ../../../data/MMWHS/CT_withGT_proc/ --cases_folds 1 --result_dir ../../../data/other/fake_MR/ --data_type MMWHS
python create_fake_img.py --resume ../results/run_fold_2/00499.pth --name run_fold_2 --a2b 1 --data_dir1 ../../../data/MMWHS/CT_withGT_proc/ --cases_folds 2 --result_dir ../../../data/other/fake_MR/ --data_type MMWHS
python create_fake_img.py --resume ../results/run_fold_3/00499.pth --name run_fold_3 --a2b 1 --data_dir1 ../../../data/MMWHS/CT_withGT_proc/ --cases_folds 3 --result_dir ../../../data/other/fake_MR/ --data_type MMWHS
python create_fake_img.py --resume ../results/run_fold_4/00549.pth --name run_fold_4 --a2b 1 --data_dir1 ../../../data/MMWHS/CT_withGT_proc/ --cases_folds 4 --result_dir ../../../data/other/fake_MR/ --data_type MMWHS
```

Then with those synthetic target data, we have two second stage segmentation models, UNet and ResUNet. The following code is an example of how to train and test a UNEt for the task $MRI \rightarrow CT$ for MYO segmentation.

```
python run_unet.py -- --pred MYO --name UNet_trained_on_fake_MRI --epochs 200 --bs 4 --lr 0.0001 --data_dir ../data/MMWHS/fake_MR/  --k_folds 5 --drit --data_dir_test ../data/MMWHS/MR_withGT_proc/ --model unet --data_type MMWHS
python evaluate_unet.py -- --pred MYO --name UNet_trained_on_fake_MRI --bs 1 --data_dir_test ../data/MMWHS/MR_withGT_proc/ --k_folds 5 --model unet --data_type MMWHS
```
- If you want to run the ResUNet, change `--model` to `ResUNet`.
- For the other segmentation tasks (i.e. other direction, structure or dataset) change the following arguments accordingly: `--pred`, `--name`, `--data_dir`, `--data_dir_test`, and `--data_type` (similar as with the UNet-NA and UNet-FS baselines).
- The final results of these baselines are also stored in the `checkpoints` folder.


# DDFSeg
For more information on this baseline, please check the README in the `DDFSeg` folder. 
To train and test this for baseline $MRI \rightarrow CT$, run the code below. For the other direction, please change the arguments `--data_dir1`, `--data_dir1`, `--test_data_dir`, `--name`, and `--resume` accordingly. For another segmentation task, you can change `--pred` to LV or RV

```
source activate gpu_env
cd DDFSeg/src

python train.py -- --name MYO_MMWHS_Target_CT --pred MYO --data_dir1 ../../../data/MMWHS/MR_withGT_proc/ --data_dir2 ../../../data/MMWHS/CT_withGT_proc/ --data_type MMWHS
python test.py -- --resume '../results/MYO_MMWHS_Target_CT/' --pred MYO --test_data_dir ../../../data/other/CT_withGT_proc/ --data_type MMWHS
```

To train and test this baseline for $T1 \rightarrow T2$, run the code below. For the other direction, please change the arguments `--data_dir1`, `--data_dir1`, `--test_data_dir`, `--name`, and `--resume` accordingly.
```
source activate gpu_env
cd DDFSeg/src
python train.py --name Liver_CHAOS_Target_T2 --pred Liver --data_dir1 ../../../data/preprocessed_chaos/T1/ --data_dir2 ../../../data/preprocessed_chaos/T2/ --data_type CHAOS
python test.py --resume '../results/Liver_CHAOS_Target_T2/' --pred Liver --test_data_dir ../../../data/CHAOS/T2/ --data_type CHAOS
```