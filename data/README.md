# Data preprocessing
This folder contains the code for the data preprocessing of the MM-WHS and CHAOS dataset and the visualization of the preprocessed data. 

- The `exploratory_notebooks` folder contains notebooks used for visualizing the preprocessed data. 
- In the `MMWHS` folder, you can download the preprocessed MMWHS data from [this google drive](https://drive.google.com/file/d/1shLpzuMr_PAtD1ruMpTz6sPl2mS7Ue1x/view?usp=share_link), and also the synthetic data created by the DRIT model
- In the `CHAOS` folder, you can download the preprocessed CHAOS data from [this google drive](https://drive.google.com/file/d/12VUqzlSbucH-9-YaxsffekeI5gF_0fKX/view?usp=share_link), and also the synthetic data created by the DRIT model

## MMWHS 
To preprocess the MMWHS data, first download the MMWHS data from [here](https://github.com/FupingWu90/CT_MR_2D_Dataset_DA), into the `MMWHS` folder.
Then run the following code from this directory (i.e. `data/`)

```
source activate gpu_env
python preproc_mmwhs.py
```

![data](example_ct.png)


## CHAOS
To preprocess the CHAOS data, first download the CHAOS data from [here](https://chaos.grand-challenge.org), into the main directory (i.e., `CrossModal-DRL`).
Then run the following code from this directory (i.e. `data/`)

```
source activate gpu_env
python preproc_chaos.py
```

![data](example_t1.png)