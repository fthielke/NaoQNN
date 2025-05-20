# SPDX-License-Identifier: MIT
import os
from functools import partial
import h5py
import numpy as np
import yaml
import tensorflow as tf

DATASET_H5_FILENAME = 'b-alls-2019.hdf5'
DATA_SPLIT_FILENAME = 'balls_split.yml'
TEST_RATIO = 0.1
VAL_RATIO = 0.05

def prepare_dataset():
    balls = h5py.File(DATASET_H5_FILENAME)

    if not os.path.exists(DATA_SPLIT_FILENAME):
        create_data_split(balls)

    if not all(os.path.exists(path) for path in ('balls_ds_training_pos', 'balls_ds_training_neg', 'balls_ds_validation', 'balls_ds_test')):
        convert_to_tf_datasets(balls)

def create_data_split(dataset):
    num_test_cases = int(len(dataset['positives/data']) * TEST_RATIO)
    num_val_cases = int(len(dataset['positives/data']) * VAL_RATIO)

    pos_indices = train_val_test_split(len(dataset['positives/data']), num_val_cases, num_test_cases)
    neg_indices = train_val_test_split(len(dataset['negatives/data']), num_val_cases, num_test_cases)

    with open(DATA_SPLIT_FILENAME, 'w', encoding='utf-8', newline='') as f:
        yaml.dump(
            {
                cat: dict(zip(('train', 'val', 'test'), map(lambda x: x.tolist(), indices)))
                for cat, indices in (('pos', pos_indices), ('neg', neg_indices))
            },
            f
        )

def train_val_test_split(num_total, num_val_cases, num_test_cases):
    indices = np.arange(num_total)
    np.random.shuffle(indices)
    return np.sort(indices[num_test_cases+num_val_cases:]), np.sort(indices[num_test_cases:num_test_cases+num_val_cases]), np.sort(indices[:num_test_cases])

def convert_to_tf_datasets(dataset):
    with open(DATA_SPLIT_FILENAME, 'r', encoding='utf-8', newline='') as f:
        indices = yaml.load(f, yaml.FullLoader)
    pos_indices = [indices['pos'][k] for k in ('train', 'val', 'test')]
    neg_indices = [indices['neg'][k] for k in ('train', 'val', 'test')]
    del indices

    pos_train, pos_val, pos_test = map(partial(load_pos_dataset, dataset), pos_indices)
    neg_train, neg_val, neg_test = map(partial(load_neg_dataset, dataset), neg_indices)

    val = pos_val.concatenate(neg_val)
    test = pos_test.concatenate(neg_test)

    pos_train.save('balls_ds_training_pos', compression='GZIP')
    neg_train.save('balls_ds_training_neg', compression='GZIP')
    val.save('balls_ds_validation', compression='GZIP')
    test.save('balls_ds_test', compression='GZIP')

def load_pos_dataset(source, indices):
    return tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(source['positives/data'][indices]),
        tf.data.Dataset.from_tensor_slices(source['positives/labels'][indices])
    ))

def load_neg_dataset(source, indices):
    return tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(source['negatives/data'][indices]),
        tf.data.Dataset.from_tensors(tf.zeros((4,), dtype=tf.int8)).repeat(len(indices))
    ))

if __name__ == '__main__':
    prepare_dataset()
