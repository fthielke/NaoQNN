# SPDX-License-Identifier: MIT
import sys
if __name__ == '__main__':
    sys.path.append('..')
import os
import tensorflow as tf

if __name__ == '__main__':
    try:
        tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
    except:
        print('Could not configure memory growth')

from ball_model import BallModel, BallModelEncoder, BallModelClassifier, BallModelDetector
from training_utils import load_data, classification_loss, detection_loss_scaled, ClassificationF1, DetectionJaccard

RANGE_CENTER = (-6, 38)
RANGE_RADIUS = (5, 19)
BATCH_SIZE = 128

MODEL_CONFIGURATIONS = (
    (BallModelEncoder.GERRIT_ORIGINAL, BallModelClassifier.GERRIT_ORIGINAL, BallModelDetector.GERRIT_ORIGINAL),
    (BallModelEncoder.GERRIT_VALIDPAD, BallModelClassifier.GERRIT_ORIGINAL, BallModelDetector.GERRIT_ORIGINAL),
    (BallModelEncoder.GERRIT_VALIDPAD_SPLITCONVS, BallModelClassifier.GERRIT_ORIGINAL, BallModelDetector.GERRIT_ORIGINAL),
    (BallModelEncoder.V4_FLOAT, BallModelClassifier.GERRIT_ORIGINAL, BallModelDetector.GERRIT_ORIGINAL),
    (BallModelEncoder.GERRIT_ORIGINAL_QUANTIZED, BallModelClassifier.GERRIT_ORIGINAL_QUANTIZED, BallModelDetector.GERRIT_ORIGINAL_QUANTIZED),
    (BallModelEncoder.GERRIT_VALIDPAD_QUANTIZED, BallModelClassifier.GERRIT_QUANTIZED, BallModelDetector.GERRIT_QUANTIZED),
    (BallModelEncoder.GERRIT_VALIDPAD_SPLITCONVS_QUANTIZED, BallModelClassifier.GERRIT_QUANTIZED, BallModelDetector.GERRIT_QUANTIZED),
    (BallModelEncoder.V4, BallModelClassifier.GERRIT_QUANTIZED, BallModelDetector.GERRIT_QUANTIZED),
)


def train_model(model_config, weights_filename, train, val):
    encoder, classifier, detector = model_config

    tf.keras.backend.clear_session()

    model = BallModel(
        RANGE_CENTER,
        RANGE_RADIUS,
        encoder=encoder,
        classifier=classifier,
        detector=detector
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=1e-2, decay_steps=100000, end_learning_rate=1e-6, power=.9)
        ),
        loss={
            'scores': classification_loss,
            'circles': detection_loss_scaled(RANGE_CENTER, RANGE_RADIUS)
        },
        metrics={
            'scores': ClassificationF1(name='f1'),
            'circles': DetectionJaccard(name='jaccard')
        },
        run_eagerly=False,
        jit_compile=True
    )

    history = model.fit(
        train,
        batch_size=BATCH_SIZE,
        steps_per_epoch=5000,
        epochs=1,
        validation_data=val
    )
    if history.history['val_scores_f1'][-1] < 0.85:
       return False

    history = model.fit(
        train,
        batch_size=BATCH_SIZE,
        steps_per_epoch=95000,
        epochs=1,
        validation_data=val
    )
    if history.history['val_scores_f1'][-1] < 0.85:
       return False

    model.save_weights(weights_filename, save_format='tf')
    return True


if __name__ == '__main__':
    train, val, test = load_data()
    del test
    train = train.rebatch(BATCH_SIZE).prefetch(buffer_size=1000)
    val = val.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('configuration', type=lambda x: MODEL_CONFIGURATIONS[int(x)], help=f'Config id in [0, {len(MODEL_CONFIGURATIONS)-1}]')
    model_config = parser.parse_args().configuration
    print(f'Chosen model config: {model_config}')

    count = 0
    while count < 10:
        if not os.path.exists('weights_new/' + '/'.join(str(c.value) for c in model_config) + f'/weights_{count}.index'):
            break
        count += 1

    while count < 10:
        success = train_model(
            model_config,
            'weights_new/' + '/'.join(str(c.value) for c in model_config) + f'/weights_{count}',
            train,
            val
        )
        if success:
            count += 1
