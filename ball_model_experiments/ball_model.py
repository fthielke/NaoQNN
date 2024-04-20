# SPDX-License-Identifier: MIT
from enum import Enum
import tensorflow as tf
from naoqnn import Sequential, QuantizedConv1x1BNRelu, QuantizedConvSpatialBNRelu, QuantizedConvStridedBNRelu, QuantizedConvBNRelu, QuantizedConv1x1Softmax, QuantizedConv1x1Scale, MaxPool2D, AvgPool2D, Flatten

class BallModelEncoder(Enum):
    V1 = 1
    V2 = 2
    V3 = 3
    V4 = 4
    V4_FLOAT = "4_float"
    GERRIT_ORIGINAL = 'gerrit_original'
    GERRIT_ORIGINAL_QUANTIZED = 'gerrit_original_quantized'
    GERRIT_VALIDPAD = 'gerrit_validpad'
    GERRIT_VALIDPAD_QUANTIZED = 'gerrit_validpad_quantized'
    GERRIT_VALIDPAD_SPLITCONVS = 'gerrit_validpad_splitconvs'
    GERRIT_VALIDPAD_SPLITCONVS_QUANTIZED = 'gerrit_validpad_splitconvs_quantized'

    def create(self) -> tf.keras.Model:
        layers = [tf.keras.Input((32, 32, 1))]

        if self == self.__class__.V1:
            layers.extend([
                QuantizedConv1x1BNRelu(16, kernel_quantization_bits=6),
                QuantizedConvSpatialBNRelu(kernel_quantization_bits=6),  # -> 30,30
                QuantizedConv1x1BNRelu(16, kernel_quantization_bits=6),

                QuantizedConv1x1BNRelu(16, kernel_quantization_bits=6),
                QuantizedConvSpatialBNRelu(kernel_quantization_bits=6),  # -> 28,28
                QuantizedConv1x1BNRelu(16, kernel_quantization_bits=6),

                MaxPool2D(),                                             # -> 14,14

                QuantizedConv1x1BNRelu(16, kernel_quantization_bits=6),
                QuantizedConvSpatialBNRelu(kernel_quantization_bits=6),  # -> 12,12
                QuantizedConv1x1BNRelu(16, kernel_quantization_bits=6),

                QuantizedConv1x1BNRelu(16, kernel_quantization_bits=6),
                QuantizedConvSpatialBNRelu(kernel_quantization_bits=6),  # -> 10,10
                QuantizedConv1x1BNRelu(16, kernel_quantization_bits=6),

                MaxPool2D(),                                             # ->  5, 5

                QuantizedConv1x1BNRelu(16, kernel_quantization_bits=6),
                QuantizedConvSpatialBNRelu(kernel_quantization_bits=6),  # ->  3, 3
                QuantizedConv1x1BNRelu(16, kernel_quantization_bits=6),

                QuantizedConv1x1BNRelu(16, kernel_quantization_bits=6),
                QuantizedConvSpatialBNRelu(kernel_quantization_bits=6),  # ->  1,1
                QuantizedConv1x1BNRelu(16, kernel_quantization_bits=6),
            ])
        elif self == self.__class__.V2:
            layers.extend([
                QuantizedConv1x1BNRelu(16, kernel_quantization_bits=6),

                MaxPool2D(),                                             # -> 16,16

                QuantizedConvSpatialBNRelu(kernel_quantization_bits=6),  # -> 14,14
                QuantizedConv1x1BNRelu(16, kernel_quantization_bits=6),

                QuantizedConv1x1BNRelu(16, kernel_quantization_bits=6),
                QuantizedConvSpatialBNRelu(kernel_quantization_bits=6),  # -> 12,12
                QuantizedConv1x1BNRelu(16, kernel_quantization_bits=6),

                MaxPool2D(),                                             # -> 6,6

                QuantizedConv1x1BNRelu(16, kernel_quantization_bits=6),
                QuantizedConvSpatialBNRelu(kernel_quantization_bits=6),  # -> 4,4
                QuantizedConv1x1BNRelu(16, kernel_quantization_bits=6),

                QuantizedConv1x1BNRelu(16, kernel_quantization_bits=6),
                QuantizedConvSpatialBNRelu(kernel_quantization_bits=6),  # -> 2,2
                QuantizedConv1x1BNRelu(16, kernel_quantization_bits=6),

                MaxPool2D(),                                             # ->  1,1
            ])
        elif self == self.__class__.V3:
            layers.extend([
                QuantizedConv1x1BNRelu(8, kernel_quantization_bits=6),
                AvgPool2D(),                                             # -> 16,16

                QuantizedConv1x1BNRelu(16, kernel_quantization_bits=6),
                QuantizedConvSpatialBNRelu(kernel_quantization_bits=6),  # -> 14,14
                QuantizedConv1x1BNRelu(16, kernel_quantization_bits=6),

                MaxPool2D(),                                             # -> 7,7

                QuantizedConv1x1BNRelu(16, kernel_quantization_bits=6),
                QuantizedConvSpatialBNRelu(kernel_quantization_bits=6),  # -> 5,5
                QuantizedConv1x1BNRelu(16, kernel_quantization_bits=6),
                QuantizedConvSpatialBNRelu(kernel_quantization_bits=6),  # -> 3,3
                QuantizedConv1x1BNRelu(16, kernel_quantization_bits=6),
                QuantizedConvSpatialBNRelu(kernel_quantization_bits=6),  # -> 1,1
                QuantizedConv1x1BNRelu(16, kernel_quantization_bits=6),
            ])
        elif self == self.__class__.V4:
            layers.extend([
                QuantizedConvStridedBNRelu(8, kernel_size=(4, 4), kernel_quantization_bits=6),  # -> 8,8

                QuantizedConv1x1BNRelu(16, kernel_quantization_bits=6),
                QuantizedConvSpatialBNRelu(kernel_quantization_bits=6),  # -> 6,6
                QuantizedConv1x1BNRelu(16, kernel_quantization_bits=6),

                MaxPool2D(),                                             # -> 3,3

                QuantizedConv1x1BNRelu(16, kernel_quantization_bits=6),
                QuantizedConvSpatialBNRelu(kernel_quantization_bits=6),  # -> 1,1
                QuantizedConv1x1BNRelu(16, kernel_quantization_bits=6),
            ])
        elif self == self.__class__.V4_FLOAT:
            layers.extend([
                tf.keras.layers.Conv2D(8, kernel_size=(4, 4), strides=(4, 4), use_bias=False),  # -> 8,8
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),

                tf.keras.layers.Conv2D(16, kernel_size=(1, 1), use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), use_bias=False),            # -> 6,6
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(16, kernel_size=(1, 1), use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),

                tf.keras.layers.MaxPool2D(),                                                    # -> 3,3

                tf.keras.layers.Conv2D(16, kernel_size=(1, 1), use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), use_bias=False),            # -> 1,1
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(16, kernel_size=(1, 1), use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
            ])
        elif self == self.__class__.GERRIT_ORIGINAL:
            layers.extend([
                tf.keras.layers.Conv2D(8, kernel_size=(3, 3), padding='same', use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(),                                                    # -> 16,16
                tf.keras.layers.Conv2D(16, kernel_size=(3, 3), padding='same', use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(),                                                    # -> 8,8
                tf.keras.layers.Conv2D(16, kernel_size=(3, 3), padding='same', use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(),                                                    # -> 4,4
                tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(),                                                    # -> 2,2
            ])
        elif self == self.__class__.GERRIT_ORIGINAL_QUANTIZED:
            layers.extend([
                QuantizedConvBNRelu(8, kernel_size=(3, 3), padding='same', kernel_quantization_bits = 6),
                MaxPool2D(),
                QuantizedConvBNRelu(16, kernel_size=(3, 3), padding='same', kernel_quantization_bits = 6),
                MaxPool2D(),
                QuantizedConvBNRelu(16, kernel_size=(3, 3), padding='same', kernel_quantization_bits = 6),
                MaxPool2D(),
                QuantizedConvBNRelu(32, kernel_size=(3, 3), padding='same', kernel_quantization_bits = 6),
                MaxPool2D(),
            ])
        elif self == self.__class__.GERRIT_VALIDPAD:
            layers.extend([
                tf.keras.layers.Conv2D(8, kernel_size=(3, 3), use_bias=False),                  # -> 30,30
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(),                                                    # -> 15,15
                tf.keras.layers.Conv2D(16, kernel_size=(2, 2), use_bias=False),                 # -> 14,14
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(),                                                    # -> 7,7
                tf.keras.layers.Conv2D(16, kernel_size=(2, 2), use_bias=False),                 # -> 6,6
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(),                                                    # -> 3,3
                tf.keras.layers.Conv2D(32, kernel_size=(3, 3), use_bias=False),                 # -> 1,1
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
            ])
        elif self == self.__class__.GERRIT_VALIDPAD_QUANTIZED:
            layers.extend([
                QuantizedConvBNRelu(8, kernel_size=(3, 3), kernel_quantization_bits = 6),       # -> 30,30
                MaxPool2D(),                                                                    # -> 15,15
                QuantizedConvBNRelu(16, kernel_size=(2, 2), kernel_quantization_bits = 6),      # -> 14,14
                MaxPool2D(),                                                                    # -> 7,7
                QuantizedConvBNRelu(16, kernel_size=(2, 2), kernel_quantization_bits = 6),      # -> 6,6
                MaxPool2D(),                                                                    # -> 3,3
                QuantizedConvBNRelu(32, kernel_size=(3, 3), kernel_quantization_bits = 6),      # -> 1,1
            ])
        elif self == self.__class__.GERRIT_VALIDPAD_SPLITCONVS:
            layers.extend([
                tf.keras.layers.Conv2D(8, kernel_size=(3, 3), use_bias=False),                  # -> 30,30
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(),                                                    # -> 15,15
                tf.keras.layers.DepthwiseConv2D(kernel_size=(2, 2), use_bias=False),            # -> 14,14
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(16, kernel_size=(1, 1), use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(),                                                    # -> 7,7
                tf.keras.layers.DepthwiseConv2D(kernel_size=(2, 2), use_bias=False),            # -> 6,6
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(16, kernel_size=(1, 1), use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(),                                                    # -> 3,3
                tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), use_bias=False),            # -> 1,1
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(32, kernel_size=(1, 1), use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
            ])
        elif self == self.__class__.GERRIT_VALIDPAD_SPLITCONVS_QUANTIZED:
            layers.extend([
                QuantizedConvBNRelu(8, kernel_size=(3, 3), kernel_quantization_bits=6),       # -> 30,30
                MaxPool2D(),                                                                  # -> 15,15
                QuantizedConvSpatialBNRelu(kernel_size=(2, 2), kernel_quantization_bits=6),   # -> 14,14
                QuantizedConv1x1BNRelu(16, kernel_quantization_bits=6),
                MaxPool2D(),                                                                  # -> 7,7
                QuantizedConvSpatialBNRelu(kernel_size=(2, 2), kernel_quantization_bits=6),   # -> 6,6
                QuantizedConv1x1BNRelu(16, kernel_quantization_bits=6),
                MaxPool2D(),                                                                  # -> 3,3
                QuantizedConvSpatialBNRelu(kernel_size=(3, 3), kernel_quantization_bits=6),   # -> 1,1
                QuantizedConv1x1BNRelu(32, kernel_quantization_bits=6),
            ])
        else:
            raise NotImplementedError()

        return Sequential(layers, name='encoder')


class BallModelClassifier(Enum):
    V1 = 1
    V1_FLOAT = "1_float"
    GERRIT_ORIGINAL = 'gerrit_original'
    GERRIT_QUANTIZED = 'gerrit_quantized'
    GERRIT_ORIGINAL_QUANTIZED = 'gerrit_original_quantized'

    def create(self, input_shape) -> tf.keras.Model:
        layers = [tf.keras.Input(input_shape)]

        if self == self.__class__.V1:
            layers.extend([
                tf.keras.layers.SpatialDropout2D(0.2),
                QuantizedConv1x1Softmax(2, kernel_quantization_bits=6),
                Flatten(),
            ])
        elif self == self.__class__.V1_FLOAT:
            layers.extend([
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(2, use_bias=True),
                tf.keras.layers.Softmax(),
            ])
        elif self == self.__class__.GERRIT_ORIGINAL:
            layers.extend([
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(32, use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(64, use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(16, use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(2, use_bias=True),
                tf.keras.layers.Softmax(),
            ])
        elif self == self.__class__.GERRIT_QUANTIZED:
            layers.extend([
                QuantizedConv1x1BNRelu(32, kernel_quantization_bits=6),
                QuantizedConv1x1BNRelu(64, kernel_quantization_bits=6),
                QuantizedConv1x1BNRelu(16, kernel_quantization_bits=6),
                QuantizedConv1x1Softmax(2, kernel_quantization_bits=6),
                Flatten(),
            ])
        elif self == self.__class__.GERRIT_ORIGINAL_QUANTIZED:
            layers.extend([
                QuantizedConvBNRelu(32, kernel_size=(2, 2), kernel_quantization_bits=6),
                QuantizedConv1x1BNRelu(64, kernel_quantization_bits=6),
                QuantizedConv1x1BNRelu(16, kernel_quantization_bits=6),
                QuantizedConv1x1Softmax(2, kernel_quantization_bits=6),
                Flatten(),
            ])
        else:
            raise NotImplementedError()

        return Sequential(layers, name='classifier')


class BallModelDetector(Enum):
    V1 = 1
    V1_FLOAT = "1_float"
    GERRIT_ORIGINAL = 'gerrit_original'
    GERRIT_QUANTIZED = 'gerrit_quantized'
    GERRIT_ORIGINAL_QUANTIZED = 'gerrit_original_quantized'

    def create(self, input_shape, range_center, range_radius) -> tf.keras.Model:
        layers = [tf.keras.Input(input_shape)]

        if self == self.__class__.V1:
            layers.extend([
                QuantizedConv1x1BNRelu(16, kernel_quantization_bits=6),
                QuantizedConv1x1Scale(3, (range_center, range_center, range_radius), kernel_quantization_bits=6),
            ])
        elif self == self.__class__.V1_FLOAT:
            layers.extend([
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(16, use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(3, use_bias=False),
                tf.keras.layers.BatchNormalization(),
            ])
        elif self == self.__class__.GERRIT_ORIGINAL:
            layers.extend([
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(32, use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(64, use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(3, use_bias=False),
                tf.keras.layers.BatchNormalization(),
            ])
        elif self == self.__class__.GERRIT_QUANTIZED:
            layers.extend([
                QuantizedConv1x1BNRelu(32, kernel_quantization_bits=6),
                QuantizedConv1x1BNRelu(64, kernel_quantization_bits=6),
                QuantizedConv1x1Scale(3, (range_center, range_center, range_radius), kernel_quantization_bits=6),
                Flatten(),
            ])
        elif self == self.__class__.GERRIT_ORIGINAL_QUANTIZED:
            layers.extend([
                QuantizedConvBNRelu(32, kernel_size=(2, 2), kernel_quantization_bits=6),
                QuantizedConv1x1BNRelu(64, kernel_quantization_bits=6),
                QuantizedConv1x1Scale(3, (range_center, range_center, range_radius), kernel_quantization_bits=6),
                Flatten(),
            ])
        else:
            raise NotImplementedError()

        layers.append(Flatten())
        return Sequential(layers, name='detector')

class BallModel(tf.keras.Model):
    def __init__(self,
                 range_center,
                 range_radius,
                 encoder: BallModelEncoder = BallModelEncoder.V1,
                 classifier: BallModelClassifier = BallModelClassifier.V1,
                 detector: BallModelDetector = BallModelDetector.V1,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.encoder = encoder.create()
        latent_shape = self.encoder.compute_output_shape(self.input_shape)[1:]
        self.classification_model = classifier.create(latent_shape)
        self.detection_model = detector.create(latent_shape, range_center, range_radius)

        self.build(self.input_shape)

    @property
    def input_shape(self):
        return (None, 32, 32, 1)
    
    def call(self, inputs, training=None, mask=None):
        latent = self.encoder(inputs, training=training, mask=mask)
        scores = self.classification_model(latent, training=training, mask=mask)
        circles = self.detection_model(latent, training=training, mask=mask)

        return {'scores': scores, 'circles': circles}

    def num_layers(self) -> int:
        return sum(
            1
            for submodel in (self.encoder, self.classification_model, self.detection_model)
            for layer in submodel.layers
            if not isinstance(layer, tf.keras.layers.SpatialDropout2D) and not isinstance(layer, tf.keras.layers.Flatten)
        )

    def get_up_to_nth_layer(self, n: int):
        num_encoder_layers = sum(1 for layer in self.encoder.layers if not isinstance(layer, tf.keras.layers.SpatialDropout2D) and not isinstance(layer, tf.keras.layers.Flatten))
        if n < num_encoder_layers:
            layers = [tf.keras.Input(self.input_shape[1:])]
            i = -1
            for layer in self.encoder.layers:
                layers.append(layer)
                if not isinstance(layer, tf.keras.layers.SpatialDropout2D) and not isinstance(layer, tf.keras.layers.Flatten):
                    i += 1
                if i == n:
                    break
            return tf.keras.Sequential(layers)

        n -= num_encoder_layers

        layers = [tf.keras.Input(self.input_shape[1:]), *self.encoder.layers]

        num_classifier_layers = sum(1 for layer in self.classification_model.layers if not isinstance(layer, tf.keras.layers.SpatialDropout2D) and not isinstance(layer, tf.keras.layers.Flatten))
        if n < num_classifier_layers:
            i = -1
            for layer in self.classification_model.layers:
                layers.append(layer)
                if not isinstance(layer, tf.keras.layers.SpatialDropout2D) and not isinstance(layer, tf.keras.layers.Flatten):
                    i += 1
                if i == n:
                    break
            return tf.keras.Sequential(layers)

        n -= num_classifier_layers

        i = -1
        for layer in self.detection_model.layers:
            layers.append(layer)
            if not isinstance(layer, tf.keras.layers.SpatialDropout2D) and not isinstance(layer, tf.keras.layers.Flatten):
                i += 1
            if i == n:
                break
        return tf.keras.Sequential(layers)

    def to_float_model(self) -> tf.keras.Model:
        inputs = tf.keras.Input(self.input_shape[1:], name='inputs')
        x = inputs
        for layer in self.encoder.layers:
            keras_layers = layer.to_keras_layer() if hasattr(layer, 'to_keras_layer') else layer
            if not hasattr(keras_layers, '__len__'):
                keras_layers = (keras_layers,)
            for keras_layer in keras_layers:
                x = keras_layer(x)
        encoded = x
    
        for layer in self.classification_model.layers[:-1]:
            if isinstance(layer, tf.keras.layers.Dropout) or isinstance(layer, tf.keras.layers.SpatialDropout2D):
                continue
            keras_layers = layer.to_keras_layer() if hasattr(layer, 'to_keras_layer') else layer
            if not hasattr(keras_layers, '__len__'):
                keras_layers = (keras_layers,)
            for keras_layer in keras_layers:
                x = keras_layer(x)
        layer = self.classification_model.layers[-1]
        keras_layers = layer.to_keras_layer() if hasattr(layer, 'to_keras_layer') else layer
        if not hasattr(keras_layers, '__len__'):
            keras_layers = [keras_layers]
        keras_layers[-1] = keras_layers[-1].__class__.from_config({**keras_layers[-1].get_config(), 'name': 'scores'})
        for keras_layer in keras_layers:
            x = keras_layer(x)
        scores = x

        x = encoded
        for layer in self.detection_model.layers[:-1]:
            keras_layers = layer.to_keras_layer() if hasattr(layer, 'to_keras_layer') else layer
            if not hasattr(keras_layers, '__len__'):
                keras_layers = (keras_layers,)
            for keras_layer in keras_layers:
                x = keras_layer(x)
        layer = self.detection_model.layers[-1]
        keras_layers = layer.to_keras_layer() if hasattr(layer, 'to_keras_layer') else layer
        if not hasattr(keras_layers, '__len__'):
            keras_layers = [keras_layers]
        keras_layers[-1] = keras_layers[-1].__class__.from_config({**keras_layers[-1].get_config(), 'name': 'circles'})
        for keras_layer in keras_layers:
            x = keras_layer(x)
        circles = x

        return tf.keras.Model(inputs, [scores, circles])
