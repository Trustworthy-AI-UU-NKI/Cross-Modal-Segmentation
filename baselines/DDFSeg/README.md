# Disentangle domain features for cross-modality cardiac image segmentation

This is the translated code from Tensorflow from the [original repository](https://github.com/Endless-Hao/DDFSeg) to PyTorch. We attempted to keep as much as the code structure the same. However, we did make some modifications to the network:
- to ensure consistency with the baselines (setting seeds, 5-fold cross-validation, etc.), which was mainly done in the `train.py` and `test.py` files.
- to account for our dataset, for which adapted mainly in the `dataset.py` file.
- We changed the validation step. In the original implementation, the target images with the ground truth annotations are used as a validation set, which conflicts with the principle of not having access to these ground truth labels during training. Therefore, we modified this approach by validating with the source images and their respective labels, translating the source images to the target domain, and then predicting the segmentation labels from there.

Please refer to the README in the baselines folder to see how you can run this baseline. 

## Code structure

- In `results`, all checkpoints to the DDFSeg model for the different folds are stored.
- `src` contains the source code for the DDFSeg model.
