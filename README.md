# Enhancing Cross-Modal Medical Image Segmentation through Compositionality and Disentanglement

This repository contains the implementation of several disentangled representation learning models for cross-modal medical image segmentation for the Master Thesis Artificial Intelligence.
In particular, it contains the implementation of our proposed method, where we introduced compositionality into a cross-modal segmentation framework to enhance performance and interpretability, while reducing computational costs. 
A shorter version of this thesis is under review at the Deep Generative Models workshop @ MICCAI 2024. If accepted, we will make this repository public and upload all the models' checkpoints to Huggingface. 
Below, you can see some examples results.

![results](results/vis_MRI->CT_MYO_LV_RV.png)

# Code structure
- The `data` folder contains the code for preprocessing the data, visualization and the actual preprocessed data.
- The `baselines` folder contains the code for the baselines; No Adaptation UNet, Full Supervision UNet, vMFnet, DRIT++ with UNet and with resUNet and DDFSeg. 
- The `src` folder contains the code for our proposed method.
- The `data` folder contains the links to out preprocessed data, some notebooks to explore the data and our preprocessing steps.



# System Requirements
If you would like to run our implementation, please use the following code to create the conda environment that is used. 

```
conda env create -f environment.yml
conda activate gpu_env
```


# Acknowledgements

Parts of the code are based on [vMFNet](https://github.com/vios-s/vMFNet), [DRIT](https://github.com/HsinYingLee/DRIT) and [DDFSeg](https://github.com/Endless-Hao/DDFSeg).