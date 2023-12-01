# Data preprocessing
This folder contains the code for the data preprocessing of the MM-WHS dataset and the visualization of the already preprocessed data of the Pnp-Adanet paper. 

- The `preprocessed` folder contains the preprocessed data, after running the code.
- The `exploratory_notebooks` folder contains notebooks used for visualizing the preprocessed data and for testing the process. 
- `transform.py` is a file which defines a data transform used by `data_preproc.py` to crop the 3D MRI data around the segmented anatomies.
- `data_preproc.py` is the main file for creating the preprocessed data.

`data_preproc.py` can be run with the following commandline to preprocess the 3D CT data of the MM-WHS dataset with the default settings:
```
python data_preproc.py -- --data_dir ../MMWHS_Dataset/ --modality CT --output_folder preprocessed/ --spatial_size 256
```