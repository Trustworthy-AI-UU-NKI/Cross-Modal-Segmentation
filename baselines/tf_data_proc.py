# imports 
import tensorflow as tf
import numpy as np
import os, sys
import monai
import PIL

import matplotlib.pyplot as plt


def decode_fn(example):
    # Decode string to bytes
    feature_description = {
        # image size, dimensions of 3 consecutive slices
        'dsize_dim0': tf.io.FixedLenFeature([], tf.int64), # 256
        'dsize_dim1': tf.io.FixedLenFeature([], tf.int64), # 256
        'dsize_dim2': tf.io.FixedLenFeature([], tf.int64), # 3
        # label size, dimension of the middle slice
        'lsize_dim0': tf.io.FixedLenFeature([], tf.int64), # 256
        'lsize_dim1': tf.io.FixedLenFeature([], tf.int64), # 256
        'lsize_dim2': tf.io.FixedLenFeature([], tf.int64), # 1
        # image slices of size [256, 256, 3]
        'data_vol': tf.io.FixedLenFeature([], tf.string),
        # label slice of size [256, 256, 1]
        'label_vol': tf.io.FixedLenFeature([], tf.string)}
    
    example = tf.io.parse_single_example(example, feature_description)

    d0 = tf.cast(example['dsize_dim0'], tf.int32)
    d1 = tf.cast(example['dsize_dim1'], tf.int32)
    d2 = tf.cast(example['dsize_dim2'], tf.int32)
    l0 = tf.cast(example['lsize_dim0'], tf.int32)
    l1 = tf.cast(example['lsize_dim1'], tf.int32)
    l2 = tf.cast(example['lsize_dim2'], tf.int32)
    data_vol = tf.io.decode_raw(example['data_vol'], tf.float32)
    label_vol = tf.io.decode_raw(example['label_vol'], tf.float32)

    # Reshape the data to the desired shape [256, 256, 3]
    data_vol = tf.reshape(data_vol, [example['dsize_dim0'], example['dsize_dim1'], example['dsize_dim2']])
    # Reshape the label to the desired shape [256, 256, 1]
    label_vol = tf.reshape(label_vol, [example['lsize_dim0'], example['lsize_dim1'], example['lsize_dim2']])



    return {'img': data_vol, 'seg': label_vol}


def create_dataset(tfrecords_filenames):
    datasets = []
    for tfrecords_filename in tfrecords_filenames:
        raw_dataset = tf.data.TFRecordDataset([tfrecords_filename])
        parsed_dataset = raw_dataset.map(decode_fn)
        datasets.append(parsed_dataset)
    
    combined_dataset = tf.data.Dataset.concatenate(*datasets)
    return combined_dataset