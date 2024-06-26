# Baselines

This folder contains the implementation for all the baselines; No Adaptation UNet, Full Supervision UNet, vMFnet, DRIT++ with UNet and with ResUNet and DDFSeg. 

# Code structure
- The `data` folder contains the code for preprocessing the data, visualization and the actual preprocessed data.
- The `baselines` folder contains the code for the baselines; No Adaptation UNet, Full Supervision UNet, vMFnet, DRIT++ with UNet and with ResUNet and DDFSeg. 
- The `src` folder contains the code for our proposed method.
- The `data` folder contains the links to out preprocessed data, some notebooks to explore the data and our preprocessing steps.

## UNet-NA 

## UNet-FS

TO Train, run

The checkpoints for the saved models ()used for the final results) can be obtained from (), by putting htem in the checkpoints folder.

TO evaluate run

## vMFNet

This code is made with XXX. For more information, please see the REadme in the DDFSeg folder. 

To train, cd to the DDFSeg folder. 


Checkpiints are also saved and can be found in (put them in the DDFseg checkpoints file)

TO evalute, run

# DRIT++ with UNet and DRIT++ with Res UNet

The code for DRIT is taken from xxxx and slightly adapted to fit our data and pipeline setup. To run the drit model, run 



THe models weights are saved in (), and put them in the XXX folder

Then we find the optimal modelby running 


Then we create the fake data by 

Then we run the UNEt and resunet by

The final segmentation models for Res UNet and UNet can be foun in (). 

To evalute these models, run


# DDFSeg

This code is made with XXX. For more information, please see the REadme in the DDFSeg folder. 

To train, cd to the DDFSeg folder. 


Checkpiints are also saved and can be found in (put them in the DDFseg checkpoints file)

TO evalute, run



