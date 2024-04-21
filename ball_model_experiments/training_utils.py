# SPDX-License-Identifier: MIT
from functools import partial
import tensorflow as tf
import numpy as np


def load_data():
    pos_train = tf.data.Dataset.load('balls_ds_training_pos', compression='GZIP')
    neg_train = tf.data.Dataset.load('balls_ds_training_neg', compression='GZIP')
    train = tf.data.Dataset.sample_from_datasets([pos_train.shuffle(len(pos_train), reshuffle_each_iteration=True), neg_train.shuffle(len(neg_train), reshuffle_each_iteration=True)], weights=[0.5, 0.5], rerandomize_each_iteration=True, stop_on_empty_dataset=True).repeat().batch(2048, num_parallel_calls=tf.data.AUTOTUNE)
    val = tf.data.Dataset.load('balls_ds_validation', compression='GZIP')
    test = tf.data.Dataset.load('balls_ds_test', compression='GZIP')
    return train, val, test

class CircleToMask(tf.keras.layers.Layer):
    def __init__(self, output_shape=(32,32), **kwargs):
        super().__init__(**kwargs)
        self.grid = tf.constant(np.transpose(np.mgrid[0:output_shape[0],0:output_shape[1]][::-1], axes=(1,2,0)), dtype=tf.float32)
    
    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], self.grid.shape[:2], 1])

    def call(self, inputs, *args, one_hot=False, **kwargs):
        score, center, radius = inputs[:,0], inputs[:,1:3], inputs[:,3]
        mask = score[:,None,None,None] * tf.cast(radius[:,None,None,None] >= tf.norm(self.grid[None,:,:,:] - center[:,None,None,:], axis=-1, keepdims=True), tf.float32)

        if one_hot:
            mask = tf.concat([1. - mask, mask], axis=-1)

        return mask

class ScaleLayer(tf.keras.layers.Layer):
    def __init__(self, input_ranges, output_ranges, **kwargs):
        super().__init__(**kwargs)
        self.factor = tf.constant([(out_max - out_min) / (in_max - in_min) for (in_min, in_max), (out_min, out_max) in zip(input_ranges, output_ranges)], dtype=tf.float32)
        self.offset = tf.constant([in_min * (out_max - out_min) / (in_max - in_min) + out_min for (in_min, in_max), (out_min, out_max) in zip(input_ranges, output_ranges)], dtype=tf.float32)

    def call(self, inputs, *args, **kwargs):
        return inputs * self.factor + self.offset

def dice_loss(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return tf.math.divide_no_nan(denominator - (2. * intersection), denominator)

def classification_loss(y_true, y_pred):
    y_true = tf.cast(y_true[:, :1], tf.float32)
    y_true = tf.concat([1. - y_true, y_true], axis=-1)
    return 0.5 * (tf.keras.losses.categorical_crossentropy(y_true, y_pred) + dice_loss(y_true[...,1], y_pred[...,1]))

def detection_loss_scaled(range_center, range_radius):
    scale_detection_outputs = ScaleLayer((range_center, range_center, range_radius), ((-1, 1), (-1, 1), (-1, 1)))
    return partial(detection_loss, scale_detection_outputs=scale_detection_outputs)

def detection_loss(y_true, y_pred, scale_detection_outputs):
    mask = y_true[:, 0] > 0
    y_true = scale_detection_outputs(tf.cast(tf.boolean_mask(y_true, mask, axis=0)[:, 1:], tf.float32))
    y_pred = scale_detection_outputs(tf.boolean_mask(y_pred, mask, axis=0))

    diff_center = tf.sqrt(tf.math.square(y_true[:, 0] - y_pred[:, 0]) + tf.math.square(y_true[:, 1] - y_pred[:, 1]))
    diff_radius = tf.math.abs(y_true[:, 2] - y_pred[:, 2])
    total_diff = 0.75 * diff_center + 0.25 * diff_radius

    return 0.5 * (tf.keras.losses.mean_absolute_error(0., total_diff) + tf.sqrt(tf.keras.losses.mean_squared_error(0., total_diff)))

class ClassificationF1(tf.keras.metrics.IoU):
    def __init__(self, **kwargs):
        super().__init__(num_classes=2, target_class_ids=(1,), **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true[:,:1], y_pred[:,1:] >= 0.5, sample_weight)

    def result(self):
        jaccard = super().result()
        return 2 * jaccard / (jaccard + 1)

class DetectionJaccard(tf.keras.metrics.IoU):
    def __init__(self, **kwargs):
        super().__init__(num_classes=2, target_class_ids=(1,), **kwargs)
        self._circle_to_mask = CircleToMask()

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(self._circle_to_mask(tf.cast(y_true, tf.float32)), self._circle_to_mask(tf.concat([tf.cast(y_true[:,:1], tf.float32), y_pred], axis=1)) >= 0.5, sample_weight)
