# Baselines

This folder contains the code for all the baselines, except for the vMFNet. That can be found in the src folder.  

# UNet-NA and UNet-FS

TO Train, run

The checkpoints for the saved models ()used for the final results) can be obtained from (), by putting htem in the checkpoints folder.

TO evaluate run



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
