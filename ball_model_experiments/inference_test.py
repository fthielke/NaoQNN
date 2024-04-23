# SPDX-License-Identifier: MIT
import sys
import os
import numpy as np
import tensorflow as tf
from ball_model import BallModel
from training_utils import load_data
from train_ball_model import MODEL_CONFIGURATIONS, RANGE_CENTER, RANGE_RADIUS


def perform_inference(dataset, model_config):
    if os.path.exists('weights/' + '/'.join(str(c.value) for c in model_config) + '/results_test.npz'):
        return

    n_models = 1 + next(i for i in range(9, -1, -1) if os.path.exists('weights/' + '/'.join(str(c.value) for c in model_config) + f'/weights_{i}.index'))
    if n_models < 10:
        print(f'Warning: only {n_models} of 10 expected models found')

    tf.keras.backend.clear_session()

    encoder, classifier, detector = model_config
    model = BallModel(
        RANGE_CENTER,
        RANGE_RADIUS,
        encoder=encoder,
        classifier=classifier,
        detector=detector
    )

    predictions = np.empty((n_models, len(test)))
    circles = np.empty((n_models, len(test), 3))
    for i in range(n_models):
        model.load_weights('weights/' + '/'.join(str(c.value) for c in model_config) + f'/weights_{i}').expect_partial()
        outputs = model.predict(test.batch(128).prefetch(buffer_size=tf.data.AUTOTUNE))
        predictions[i] = outputs['scores'][:,1]
        circles[i] = outputs['circles']

    np.savez_compressed('weights/' + '/'.join(str(c.value) for c in model_config) + '/results_test.npz', predictions=predictions, circles=circles)

if __name__ == '__main__':
    train, val, test = load_data()
    del train
    del val

    for model_config in MODEL_CONFIGURATIONS:
        perform_inference(test, model_config)
