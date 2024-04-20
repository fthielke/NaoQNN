# SPDX-License-Identifier: MIT
from typing import Sequence, Tuple, Union, Optional
import tensorflow as tf

try:
    import onnx
    from onnx import numpy_helper
except:
    pass
import numpy as np


@tf.custom_gradient
def round_with_gradients(x):
    def grad(dy):
        return dy
    return tf.round(x), grad

@tf.custom_gradient
def clip_i16_with_gradients(x):
    def grad(dy):
        return dy
    return tf.clip_by_value(x, -32768., 32767.), grad


class QuantizedConv1x1BNRelu(tf.keras.layers.Layer):
    def __init__(self,
                 filters: int,
                 kernel_quantization_bits: int = 8,
                 momentum: float = 0.99,
                 epsilon: float = 1e-3,
                 kernel_initializer="he_uniform",
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 beta_initializer="zeros",
                 moving_mean_initializer="zeros",
                 moving_variance_initializer="ones",
                 **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_quantization_bits = kernel_quantization_bits
        self.momentum = momentum
        self.epsilon = epsilon
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.moving_mean_initializer = tf.keras.initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = tf.keras.initializers.get(moving_variance_initializer)
        self.no_rounding = False

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.filters),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype
        )
        self.moving_mean = self.add_weight(
            name="moving_mean",
            shape=(self.filters,),
            dtype=self.dtype,
            initializer=self.moving_mean_initializer,
            trainable=False
        )
        self.moving_variance = self.add_weight(
            name="moving_variance",
            shape=(self.filters,),
            dtype=self.dtype,
            initializer=self.moving_variance_initializer,
            trainable=False
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(self.filters,),
            dtype=self.dtype,
            initializer=self.beta_initializer,
            trainable=True
        )

        self.built = True

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([*input_shape[:-1], self.filters])

    def get_quantized_weights(self):
        batch_norm_factors = tf.math.rsqrt(self.moving_variance + self.epsilon)
        kernel = self.kernel * batch_norm_factors[None, :]

        quantized_scale = 2. ** (self.kernel_quantization_bits - 1)
        quantized_kernel_max = quantized_scale - 1.
        quantized_kernel_min = -quantized_kernel_max

        kernel_norm = tf.reduce_mean(tf.abs(tf.reduce_sum(tf.stop_gradient(kernel), axis=1)))

        quantized_kernel = tf.clip_by_value(round_with_gradients(quantized_kernel_max * tf.math.divide_no_nan(kernel, kernel_norm)), quantized_kernel_min, quantized_kernel_max)
        quantized_offsets = clip_i16_with_gradients(round_with_gradients(256. * self.beta - tf.math.divide_no_nan(self.moving_mean * batch_norm_factors * (quantized_scale / quantized_kernel_max), kernel_norm)))

        return quantized_kernel, quantized_scale, quantized_offsets

    def call(self, inputs, training=None, *args, **kwargs):
        if not self.trainable:
            training = False
        elif training is None:
            training = tf.keras.backend.learning_phase()

        if training:
            # Calculate the convolution once with floats to get the batch norm statistics
            conv_result = tf.tensordot(inputs, tf.stop_gradient(self.kernel), [[inputs.shape.rank - 1], [0]])
            mean = tf.reduce_mean(conv_result, axis=(0,1,2))
            variance = tf.math.reduce_variance(conv_result, axis=(0,1,2))

            # Update the BN statistics
            self.moving_mean.assign_sub((self.moving_mean - mean) * (1. - self.momentum))
            self.moving_variance.assign_sub((self.moving_variance - variance) * (1. - self.momentum))

        # Build the quantized weights
        quantized_kernel, quantized_scale, quantized_offsets = self.get_quantized_weights()

        # Perform calculation
        conv_result = tf.tensordot(inputs, quantized_kernel, [[inputs.shape.rank - 1], [0]])
        conv_result = clip_i16_with_gradients(conv_result)
        conv_result = conv_result / quantized_scale
        if not self.no_rounding:
            conv_result = round_with_gradients(conv_result)

        return tf.clip_by_value(conv_result + quantized_offsets, 0., 255.)
    
    def to_keras_layer(self) -> tf.keras.layers.Conv2D:
        assert self.built

        quantized_kernel, quantized_scale, quantized_offsets = self.get_quantized_weights()

        layer = tf.keras.layers.Conv2D(filters=self.kernel.shape[1], kernel_size=(1, 1), use_bias=True, activation='relu')
        layer.build(self.input_shape)
        layer.set_weights([
            (quantized_kernel / quantized_scale).numpy().reshape((1, 1, *quantized_kernel.shape)),
            quantized_offsets.numpy()
        ])
        return layer

    def add_to_onnx_model(self, model: 'onnx.ModelProto', input_name: str, output_name: Optional[str] = None) -> Tuple['onnx.ModelProto', str]:
        if output_name is None:
            output_name = self.name + '_output'

        quantized_kernel, quantized_scale, quantized_offsets = self.get_quantized_weights()

        # Conv
        conv_node = model.graph.node.add(name=self.name + '_conv', op_type='ConvInteger')
        conv_node.input.extend([input_name, self.name + '_weights'])
        conv_node.output.append(self.name + '_conv_output')
        model.graph.initializer.append(numpy_helper.from_array(quantized_kernel.numpy().astype(np.int8).transpose()[...,None,None], name=self.name + '_weights'))

        # Clip (to int16)
        clip_i16_node = model.graph.node.add(name=self.name + '_clipi16', op_type='Clip')
        clip_i16_node.input.extend([self.name + '_conv_output', 'const_int16_min', 'const_int16_max'])
        clip_i16_node.output.append(self.name + '_clipi16_output')
        if not any(i.name == 'const_int16_min' for i in model.graph.initializer):
            model.graph.initializer.append(numpy_helper.from_array(np.array(-32768, dtype=np.int32), name='const_int16_min'))
        if not any(i.name == 'const_int16_max' for i in model.graph.initializer):
            model.graph.initializer.append(numpy_helper.from_array(np.array(32767, dtype=np.int32), name='const_int16_max'))

        # Cast (to int16)
        cast_i16_node = model.graph.node.add(name=self.name + '_casti16', op_type='Cast')
        cast_i16_node.input.append(self.name + '_clipi16_output')
        cast_i16_node.output.append(self.name + '_casti16_output')
        cast_i16_node.attribute.add(name='to', type=onnx.AttributeProto.INT, i=onnx.TensorProto.INT16)

        # Div (by scale)
        quantized_scale = int(quantized_scale)
        div_node = model.graph.node.add(name=self.name + '_div', op_type='Div')
        div_node.input.extend([self.name + '_casti16_output', f'const_{quantized_scale}_i16'])
        div_node.output.append(self.name + '_div_output')
        if not any(i.name == f'const_{quantized_scale}_i16' for i in model.graph.initializer):
            model.graph.initializer.append(numpy_helper.from_array(np.array(quantized_scale, dtype=np.int16), name=f'const_{quantized_scale}_i16'))

        # Add (offsets)
        add_node = model.graph.node.add(name=self.name + '_add', op_type='Add')
        add_node.input.extend([self.name + '_div_output', self.name + '_offsets'])
        add_node.output.append(self.name + '_add_output')
        model.graph.initializer.append(numpy_helper.from_array(quantized_offsets.numpy().astype(np.int16)[None, :, None, None], name=self.name + '_offsets'))

        # Clip (ReLU)
        clip_node = model.graph.node.add(name=self.name + '_clip', op_type='Clip')
        clip_node.input.extend([self.name + '_add_output', 'const_0_i16', 'const_255_i16'])
        clip_node.output.append(self.name + '_clip_output')
        if not any(i.name == 'const_0_i16' for i in model.graph.initializer):
            model.graph.initializer.append(numpy_helper.from_array(np.array(0, dtype=np.int16), name='const_0_i16'))
        if not any(i.name == 'const_255_i16' for i in model.graph.initializer):
            model.graph.initializer.append(numpy_helper.from_array(np.array(255, dtype=np.int16), name='const_255_i16'))

        # Cast (to uint8)
        cast_u8_node = model.graph.node.add(name=self.name + '_castu8', op_type='Cast')
        cast_u8_node.input.append(self.name + '_clip_output')
        cast_u8_node.output.append(output_name)
        cast_u8_node.attribute.add(name='to', type=onnx.AttributeProto.INT, i=onnx.TensorProto.UINT8)

        return model, output_name

class QuantizedConv1x1Softmax(tf.keras.layers.Layer):
    def __init__(self,
                 filters: int,
                 kernel_quantization_bits: int = 8,
                 kernel_initializer="he_uniform",
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer="zeros",
                 **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_quantization_bits = kernel_quantization_bits
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)

    def get_config(self):
        return {
            'filters': self.filters,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'bias_initializer': self.bias_initializer,
            **super().get_config()
        }

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.filters),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.filters,),
            dtype=self.dtype,
            initializer=self.bias_initializer,
            trainable=True
        )

        self.built = True

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([*input_shape[:-1], self.filters])

    def get_quantized_weights(self):
        quantized_scale = 2. ** (self.kernel_quantization_bits - 1)
        quantized_kernel_max = quantized_scale - 1.
        quantized_kernel_min = -quantized_kernel_max

        kernel_norm = tf.reduce_mean(tf.abs(tf.reduce_sum(tf.stop_gradient(self.kernel), axis=0)))

        quantized_kernel = tf.clip_by_value(round_with_gradients(quantized_kernel_max * tf.math.divide_no_nan(self.kernel, kernel_norm)), quantized_kernel_min, quantized_kernel_max)
        quantized_offsets = clip_i16_with_gradients(round_with_gradients(256. * self.bias))

        return quantized_kernel, quantized_scale, quantized_offsets

    def call(self, inputs, training=None, *args, **kwargs):
        if not self.trainable:
            training = False
        elif training is None:
            training = tf.keras.backend.learning_phase()

        # Build the quantized weights
        quantized_kernel, quantized_scale, quantized_offsets = self.get_quantized_weights()

        # Perform calculation
        conv_result = tf.tensordot(inputs, quantized_kernel, [[inputs.shape.rank - 1], [0]])
        conv_result = clip_i16_with_gradients(conv_result)
        conv_result = round_with_gradients(conv_result / quantized_scale) + quantized_offsets

        return tf.nn.softmax(conv_result - tf.reduce_max(conv_result, axis=-1, keepdims=True), axis=-1)
    
    def to_keras_layer(self) -> Tuple[tf.keras.layers.Conv2D, tf.keras.layers.Softmax]:
        assert self.built

        quantized_kernel, quantized_scale, quantized_offsets = self.get_quantized_weights()

        layer = tf.keras.layers.Conv2D(filters=self.kernel.shape[1], kernel_size=(1, 1), use_bias=True, activation='linear')
        layer.build(self.input_shape)
        layer.set_weights([
            (quantized_kernel / quantized_scale).numpy().reshape((1, 1, *quantized_kernel.shape)),
            quantized_offsets.numpy()
        ])
        return (layer, tf.keras.layers.Softmax(axis=-1))
    
    def add_to_onnx_model(self, model: 'onnx.ModelProto', input_name: str, output_name: Optional[str] = None) -> Tuple['onnx.ModelProto', str]:
        if output_name is None:
            output_name = self.name + '_output'

        quantized_kernel, quantized_scale, quantized_offsets = self.get_quantized_weights()

        # Conv
        conv_node = model.graph.node.add(name=self.name + '_conv', op_type='ConvInteger')
        conv_node.input.extend([input_name, self.name + '_weights'])
        conv_node.output.append(self.name + '_conv_output')
        model.graph.initializer.append(numpy_helper.from_array(quantized_kernel.numpy().astype(np.int8).transpose()[...,None,None], name=self.name + '_weights'))

        # Clip (to int16)
        clip_i16_node = model.graph.node.add(name=self.name + '_clipi16', op_type='Clip')
        clip_i16_node.input.extend([self.name + '_conv_output', 'const_int16_min', 'const_int16_max'])
        clip_i16_node.output.append(self.name + '_clipi16_output')
        if not any(i.name == 'const_int16_min' for i in model.graph.initializer):
            model.graph.initializer.append(numpy_helper.from_array(np.array(-32768, dtype=np.int32), name='const_int16_min'))
        if not any(i.name == 'const_int16_max' for i in model.graph.initializer):
            model.graph.initializer.append(numpy_helper.from_array(np.array(32767, dtype=np.int32), name='const_int16_max'))

        # Cast (to int16)
        cast_i16_node = model.graph.node.add(name=self.name + '_casti16', op_type='Cast')
        cast_i16_node.input.append(self.name + '_clipi16_output')
        cast_i16_node.output.append(self.name + '_casti16_output')
        cast_i16_node.attribute.add(name='to', type=onnx.AttributeProto.INT, i=onnx.TensorProto.INT16)

        # Div (by scale)
        quantized_scale = int(quantized_scale)
        div_node = model.graph.node.add(name=self.name + '_div', op_type='Div')
        div_node.input.extend([self.name + '_casti16_output', f'const_{quantized_scale}_i16'])
        div_node.output.append(self.name + '_div_output')
        if not any(i.name == f'const_{quantized_scale}_i16' for i in model.graph.initializer):
            model.graph.initializer.append(numpy_helper.from_array(np.array(quantized_scale, dtype=np.int16), name=f'const_{quantized_scale}_i16'))

        # Add (offsets)
        add_node = model.graph.node.add(name=self.name + '_add', op_type='Add')
        add_node.input.extend([self.name + '_div_output', self.name + '_offsets'])
        add_node.output.append(output_name)
        model.graph.initializer.append(numpy_helper.from_array(quantized_offsets.numpy().astype(np.int16)[None, :, None, None], name=self.name + '_offsets'))

        # No softmax here -> just use argmax during inference

        return model, output_name

class QuantizedConv1x1Sigmoid(tf.keras.layers.Layer):
    def __init__(self,
                 filters: int,
                 kernel_quantization_bits: int = 8,
                 kernel_initializer="he_uniform",
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer="zeros",
                 **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_quantization_bits = kernel_quantization_bits
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)

    def get_config(self):
        return {
            'filters': self.filters,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'bias_initializer': self.bias_initializer,
            **super().get_config()
        }

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.filters),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.filters,),
            dtype=self.dtype,
            initializer=self.bias_initializer,
            trainable=True
        )

        self.built = True

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([*input_shape[:-1], self.filters])

    def get_quantized_weights(self):
        quantized_scale = 2. ** (self.kernel_quantization_bits - 1)
        quantized_kernel_max = quantized_scale - 1.
        quantized_kernel_min = -quantized_kernel_max

        kernel_norm = tf.reduce_mean(tf.abs(tf.reduce_sum(tf.stop_gradient(self.kernel), axis=0)))

        quantized_kernel = tf.clip_by_value(round_with_gradients(quantized_kernel_max * tf.math.divide_no_nan(self.kernel, kernel_norm)), quantized_kernel_min, quantized_kernel_max)
        quantized_offsets = clip_i16_with_gradients(round_with_gradients(256. * self.bias))

        return quantized_kernel, quantized_scale, quantized_offsets

    def call(self, inputs, training=None, *args, **kwargs):
        if not self.trainable:
            training = False
        elif training is None:
            training = tf.keras.backend.learning_phase()

        # Build the quantized weights
        quantized_kernel, quantized_scale, quantized_offsets = self.get_quantized_weights()

        # Perform calculation
        conv_result = tf.tensordot(inputs, quantized_kernel, [[inputs.shape.rank - 1], [0]])
        conv_result = clip_i16_with_gradients(conv_result)
        conv_result = round_with_gradients(conv_result / quantized_scale) + quantized_offsets

        return tf.nn.sigmoid((conv_result / 32.) - 4.)
    
    def add_to_onnx_model(self, model: 'onnx.ModelProto', input_name: str, output_name: Optional[str] = None) -> Tuple['onnx.ModelProto', str]:
        if output_name is None:
            output_name = self.name + '_output'

        quantized_kernel, quantized_scale, quantized_offsets = self.get_quantized_weights()

        # Conv
        conv_node = model.graph.node.add(name=self.name + '_conv', op_type='ConvInteger')
        conv_node.input.extend([input_name, self.name + '_weights'])
        conv_node.output.append(self.name + '_conv_output')
        model.graph.initializer.append(numpy_helper.from_array(quantized_kernel.numpy().astype(np.int8).transpose()[...,None,None], name=self.name + '_weights'))

        # Clip (to int16)
        clip_i16_node = model.graph.node.add(name=self.name + '_clipi16', op_type='Clip')
        clip_i16_node.input.extend([self.name + '_conv_output', 'const_int16_min', 'const_int16_max'])
        clip_i16_node.output.append(self.name + '_clipi16_output')
        if not any(i.name == 'const_int16_min' for i in model.graph.initializer):
            model.graph.initializer.append(numpy_helper.from_array(np.array(-32768, dtype=np.int32), name='const_int16_min'))
        if not any(i.name == 'const_int16_max' for i in model.graph.initializer):
            model.graph.initializer.append(numpy_helper.from_array(np.array(32767, dtype=np.int32), name='const_int16_max'))

        # Cast (to int16)
        cast_i16_node = model.graph.node.add(name=self.name + '_casti16', op_type='Cast')
        cast_i16_node.input.append(self.name + '_clipi16_output')
        cast_i16_node.output.append(self.name + '_casti16_output')
        cast_i16_node.attribute.add(name='to', type=onnx.AttributeProto.INT, i=onnx.TensorProto.INT16)

        # Div (by scale)
        quantized_scale = int(quantized_scale)
        div_node = model.graph.node.add(name=self.name + '_div', op_type='Div')
        div_node.input.extend([self.name + '_casti16_output', f'const_{quantized_scale}_i16'])
        div_node.output.append(self.name + '_div_output')
        if not any(i.name == f'const_{quantized_scale}_i16' for i in model.graph.initializer):
            model.graph.initializer.append(numpy_helper.from_array(np.array(quantized_scale, dtype=np.int16), name=f'const_{quantized_scale}_i16'))

        # Add (offsets)
        add_node = model.graph.node.add(name=self.name + '_add', op_type='Add')
        add_node.input.extend([self.name + '_div_output', self.name + '_offsets'])
        add_node.output.append(self.name + '_add_output')
        model.graph.initializer.append(numpy_helper.from_array(quantized_offsets.numpy().astype(np.int16)[None, :, None, None], name=self.name + '_offsets'))

        # Cast (to float)
        cast_float_node = model.graph.node.add(name=self.name + '_castf32', op_type='Cast')
        cast_float_node.input.append(self.name + '_add_output')
        cast_float_node.output.append(self.name + '_castf32_output')
        cast_float_node.attribute.add(name='to', type=onnx.AttributeProto.INT, i=onnx.TensorProto.FLOAT)

        # Div (32)
        div32_node = model.graph.node.add(name=self.name + '_div32', op_type='Div')
        div32_node.input.extend([self.name + '_castf32_output', 'const_32_f32'])
        div32_node.output.append(self.name + '_div32_output')
        if not any(i.name == 'const_32_f32' for i in model.graph.initializer):
            model.graph.initializer.append(numpy_helper.from_array(np.array(32, dtype=np.float32), name='const_32_f32'))

        # Sub (4)
        sub_node = model.graph.node.add(name=self.name + '_sub', op_type='Sub')
        sub_node.input.extend([self.name + '_div32_output', 'const_4_f32'])
        sub_node.output.append(self.name + '_sub_output')
        if not any(i.name == 'const_4_f32' for i in model.graph.initializer):
            model.graph.initializer.append(numpy_helper.from_array(np.array(4, dtype=np.float32), name='const_4_f32'))

        # Sigmoid
        sigmoid_node = model.graph.node.add(name=self.name + '_sigmoid', op_type='Sigmoid')
        sigmoid_node.input.append(self.name + '_sub_output')
        sigmoid_node.output.append(output_name)

        return model, output_name

class QuantizedConv1x1Scale(tf.keras.layers.Layer):
    def __init__(self,
                 filters: int,
                 output_ranges: Sequence[Tuple[int, int]],
                 kernel_quantization_bits: int = 8,
                 kernel_initializer="he_uniform",
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer="zeros",
                 **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.output_ranges = output_ranges
        self.kernel_quantization_bits = kernel_quantization_bits
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)

    def get_config(self):
        return {
            'filters': self.filters,
            'output_ranges': self.output_ranges,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'bias_initializer': self.bias_initializer,
            **super().get_config()
        }

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.filters),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.filters,),
            dtype=self.dtype,
            initializer=self.bias_initializer,
            trainable=True
        )
        self.scale_factor = tf.constant([(out_max - out_min) / 255. for (out_min, out_max) in self.output_ranges], dtype=tf.float32)[None,None,None,:]
        self.scale_offset = tf.constant([out_min for (out_min, out_max) in self.output_ranges], dtype=tf.float32)[None,None,None,:]

        self.built = True

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([*input_shape[:-1], self.filters])

    def get_quantized_weights(self):
        quantized_scale = 2. ** (self.kernel_quantization_bits - 1)
        quantized_kernel_max = quantized_scale - 1.
        quantized_kernel_min = -quantized_kernel_max

        kernel_norm = tf.reduce_mean(tf.abs(tf.reduce_sum(tf.stop_gradient(self.kernel), axis=0)))

        quantized_kernel = tf.clip_by_value(round_with_gradients(quantized_kernel_max * tf.math.divide_no_nan(self.kernel, kernel_norm)), quantized_kernel_min, quantized_kernel_max)
        quantized_offsets = clip_i16_with_gradients(round_with_gradients(256. * self.bias))

        return quantized_kernel, quantized_scale, quantized_offsets

    def call(self, inputs, training=None, *args, **kwargs):
        if not self.trainable:
            training = False
        elif training is None:
            training = tf.keras.backend.learning_phase()

        # Build the quantized weights
        quantized_kernel, quantized_scale, quantized_offsets = self.get_quantized_weights()

        # Perform calculation
        conv_result = tf.tensordot(inputs, quantized_kernel, [[inputs.shape.rank - 1], [0]])
        conv_result = clip_i16_with_gradients(conv_result)
        conv_result = round_with_gradients(conv_result / quantized_scale) + quantized_offsets

        return conv_result * self.scale_factor + self.scale_offset
    
    def to_keras_layer(self) -> tf.keras.layers.Conv2D:
        assert self.built

        quantized_kernel, quantized_scale, quantized_offsets = self.get_quantized_weights()
        scale_factors = np.array([(out_max - out_min) / 255. for (out_min, out_max) in self.output_ranges], dtype=np.float32)
        scaled_offsets = quantized_offsets.numpy() * scale_factors + np.array([out_min for (out_min, out_max) in self.output_ranges], dtype=np.float32)
        scale_factors = scale_factors / quantized_scale

        layer = tf.keras.layers.Conv2D(filters=self.kernel.shape[1], kernel_size=(1, 1), use_bias=True, activation='linear')
        layer.build(self.input_shape)
        layer.set_weights([
            (quantized_kernel.numpy() * scale_factors[None, :]).reshape((1, 1, *quantized_kernel.shape)),
            scaled_offsets
        ])
        return layer
    
    def add_to_onnx_model(self, model: 'onnx.ModelProto', input_name: str, output_name: Optional[str] = None) -> Tuple['onnx.ModelProto', str]:
        if output_name is None:
            output_name = self.name + '_output'

        quantized_kernel, quantized_scale, quantized_offsets = self.get_quantized_weights()
        scale_factors = np.array([(out_max - out_min) / 255. for (out_min, out_max) in self.output_ranges], dtype=np.float32)[None,:,None,None]
        scaled_offsets = quantized_offsets.numpy()[None,:,None,None] * scale_factors + np.array([out_min for (out_min, out_max) in self.output_ranges], dtype=np.float32)[None,:,None,None]

        # Conv
        conv_node = model.graph.node.add(name=self.name + '_conv', op_type='ConvInteger')
        conv_node.input.extend([input_name, self.name + '_weights'])
        conv_node.output.append(self.name + '_conv_output')
        model.graph.initializer.append(numpy_helper.from_array(quantized_kernel.numpy().astype(np.int8).transpose()[...,None,None], name=self.name + '_weights'))

        # Clip (to int16)
        clip_i16_node = model.graph.node.add(name=self.name + '_clipi16', op_type='Clip')
        clip_i16_node.input.extend([self.name + '_conv_output', 'const_int16_min', 'const_int16_max'])
        clip_i16_node.output.append(self.name + '_clipi16_output')
        if not any(i.name == 'const_int16_min' for i in model.graph.initializer):
            model.graph.initializer.append(numpy_helper.from_array(np.array(-32768, dtype=np.int32), name='const_int16_min'))
        if not any(i.name == 'const_int16_max' for i in model.graph.initializer):
            model.graph.initializer.append(numpy_helper.from_array(np.array(32767, dtype=np.int32), name='const_int16_max'))

        # Cast (to float)
        cast_float_node = model.graph.node.add(name=self.name + '_castf32', op_type='Cast')
        cast_float_node.input.append(self.name + '_clipi16_output')
        cast_float_node.output.append(self.name + '_castf32_output')
        cast_float_node.attribute.add(name='to', type=onnx.AttributeProto.INT, i=onnx.TensorProto.FLOAT)

        # Mul (scale_factors / quantized_scale)
        mul_node = model.graph.node.add(name=self.name + '_mul', op_type='Mul')
        mul_node.input.extend([self.name + '_castf32_output', self.name + '_scale_factors'])
        mul_node.output.append(self.name + '_mul_output')
        model.graph.initializer.append(numpy_helper.from_array(scale_factors / quantized_scale, name=self.name + '_scale_factors'))

        # Add (offsets)
        add_node = model.graph.node.add(name=self.name + '_add', op_type='Add')
        add_node.input.extend([self.name + '_mul_output', self.name + '_offsets'])
        add_node.output.append(output_name)
        model.graph.initializer.append(numpy_helper.from_array(scaled_offsets, name=self.name + '_offsets'))

        return model, output_name


class QuantizedConvSpatialBNRelu(tf.keras.layers.Layer):
    def __init__(self,
                 kernel_size: Tuple[int, int] = (3, 3),
                 padding: str = 'valid',
                 kernel_quantization_bits: int = 8,
                 momentum: float = 0.99,
                 epsilon: float = 1e-3,
                 kernel_initializer="he_uniform",
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 beta_initializer="zeros",
                 moving_mean_initializer="zeros",
                 moving_variance_initializer="ones",
                 **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.padding = padding
        self.kernel_quantization_bits = kernel_quantization_bits
        self.momentum = momentum
        self.epsilon = epsilon
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.moving_mean_initializer = tf.keras.initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = tf.keras.initializers.get(moving_variance_initializer)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(*self.kernel_size, input_shape[-1]),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype
        )
        self.moving_mean = self.add_weight(
            name="moving_mean",
            shape=(input_shape[-1],),
            dtype=self.dtype,
            initializer=self.moving_mean_initializer,
            trainable=False
        )
        self.moving_variance = self.add_weight(
            name="moving_variance",
            shape=(input_shape[-1],),
            dtype=self.dtype,
            initializer=self.moving_variance_initializer,
            trainable=False
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(input_shape[-1],),
            dtype=self.dtype,
            initializer=self.beta_initializer,
            trainable=True
        )

        self.built = True

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(list(input_shape) if self.padding == 'same' else [input_shape[0]] + [s - (k + 1) // 2 for s, k in zip(input_shape[1:], self.kernel_size)] + [input_shape[-1]])

    def get_quantized_weights(self):
        batch_norm_factors = tf.math.rsqrt(self.moving_variance + self.epsilon)
        kernel = self.kernel[..., None] * batch_norm_factors[None, None, :, None]

        quantized_scale = 2. ** (self.kernel_quantization_bits - 1)
        quantized_kernel_max = quantized_scale - 1.
        quantized_kernel_min = -quantized_kernel_max

        kernel_norm = tf.abs(tf.reduce_sum(tf.stop_gradient(kernel), axis=(0,1,3)))

        quantized_kernel = tf.clip_by_value(round_with_gradients(quantized_kernel_max * tf.math.divide_no_nan(kernel, kernel_norm[None, None, :, None])), quantized_kernel_min, quantized_kernel_max)
        quantized_offsets = clip_i16_with_gradients(round_with_gradients(256. * self.beta - tf.math.divide_no_nan(self.moving_mean * batch_norm_factors * (quantized_scale / quantized_kernel_max), kernel_norm)))

        return quantized_kernel, quantized_scale, quantized_offsets

    def call(self, inputs, training=None, *args, **kwargs):
        if not self.trainable:
            training = False
        elif training is None:
            training = tf.keras.backend.learning_phase()

        if training:
            # Calculate the convolution once with floats to get the batch norm statistics
            conv_result = tf.nn.depthwise_conv2d(inputs, filter=tf.stop_gradient(self.kernel[..., None]), strides=[1, 1, 1, 1], padding=self.padding.upper())
            mean = tf.reduce_mean(conv_result, axis=(0,1,2))
            variance = tf.math.reduce_variance(conv_result, axis=(0,1,2))

            # Update the BN statistics
            self.moving_mean.assign_sub((self.moving_mean - mean) * (1. - self.momentum))
            self.moving_variance.assign_sub((self.moving_variance - variance) * (1. - self.momentum))

        # Build the quantized weights
        quantized_kernel, quantized_scale, quantized_offsets = self.get_quantized_weights()

        # Perform calculation
        conv_result = tf.nn.depthwise_conv2d(inputs, filter=quantized_kernel, strides=[1, 1, 1, 1], padding=self.padding.upper())
        conv_result = clip_i16_with_gradients(conv_result)
        conv_result = round_with_gradients(conv_result / quantized_scale)
            
        return tf.clip_by_value(conv_result + quantized_offsets, 0., 255.)
    
    def to_keras_layer(self) -> tf.keras.layers.DepthwiseConv2D:
        assert self.built

        quantized_kernel, quantized_scale, quantized_offsets = self.get_quantized_weights()

        layer = tf.keras.layers.DepthwiseConv2D(kernel_size=self.kernel_size, use_bias=True, activation='relu')
        layer.build(self.input_shape)
        layer.set_weights([
            (quantized_kernel / quantized_scale).numpy(),
            quantized_offsets.numpy()
        ])
        return layer
    
    def add_to_onnx_model(self, model: 'onnx.ModelProto', input_name: str, output_name: Optional[str] = None) -> Tuple['onnx.ModelProto', str]:
        if output_name is None:
            output_name = self.name + '_output'

        quantized_kernel, quantized_scale, quantized_offsets = self.get_quantized_weights()
        
        # Conv
        conv_node = model.graph.node.add(name=self.name + '_conv', op_type='ConvInteger')
        conv_node.input.extend([input_name, self.name + '_weights'])
        conv_node.output.append(self.name + '_conv_output')
        model.graph.initializer.append(numpy_helper.from_array(np.transpose(quantized_kernel.numpy().astype(np.int8), axes=(2, 3, 0, 1)), name=self.name + '_weights'))
        conv_node.attribute.add(name='group', type=onnx.AttributeProto.INT, i=quantized_kernel.numpy().shape[2])

        # Clip (to int16)
        clip_i16_node = model.graph.node.add(name=self.name + '_clipi16', op_type='Clip')
        clip_i16_node.input.extend([self.name + '_conv_output', 'const_int16_min', 'const_int16_max'])
        clip_i16_node.output.append(self.name + '_clipi16_output')
        if not any(i.name == 'const_int16_min' for i in model.graph.initializer):
            model.graph.initializer.append(numpy_helper.from_array(np.array(-32768, dtype=np.int32), name='const_int16_min'))
        if not any(i.name == 'const_int16_max' for i in model.graph.initializer):
            model.graph.initializer.append(numpy_helper.from_array(np.array(32767, dtype=np.int32), name='const_int16_max'))

        # Cast (to int16)
        cast_i16_node = model.graph.node.add(name=self.name + '_casti16', op_type='Cast')
        cast_i16_node.input.append(self.name + '_clipi16_output')
        cast_i16_node.output.append(self.name + '_casti16_output')
        cast_i16_node.attribute.add(name='to', type=onnx.AttributeProto.INT, i=onnx.TensorProto.INT16)

        # Div (by scale)
        quantized_scale = int(quantized_scale)
        div_node = model.graph.node.add(name=self.name + '_div', op_type='Div')
        div_node.input.extend([self.name + '_casti16_output', f'const_{quantized_scale}_i16'])
        div_node.output.append(self.name + '_div_output')
        if not any(i.name == f'const_{quantized_scale}_i16' for i in model.graph.initializer):
            model.graph.initializer.append(numpy_helper.from_array(np.array(quantized_scale, dtype=np.int16), name=f'const_{quantized_scale}_i16'))

        # Add (offsets)
        add_node = model.graph.node.add(name=self.name + '_add', op_type='Add')
        add_node.input.extend([self.name + '_div_output', self.name + '_offsets'])
        add_node.output.append(self.name + '_add_output')
        model.graph.initializer.append(numpy_helper.from_array(quantized_offsets.numpy().astype(np.int16)[None, :, None, None], name=self.name + '_offsets'))

        # Clip (ReLU)
        clip_node = model.graph.node.add(name=self.name + '_clip', op_type='Clip')
        clip_node.input.extend([self.name + '_add_output', 'const_0_i16', 'const_255_i16'])
        clip_node.output.append(self.name + '_clip_output')
        if not any(i.name == 'const_0_i16' for i in model.graph.initializer):
            model.graph.initializer.append(numpy_helper.from_array(np.array(0, dtype=np.int16), name='const_0_i16'))
        if not any(i.name == 'const_255_i16' for i in model.graph.initializer):
            model.graph.initializer.append(numpy_helper.from_array(np.array(255, dtype=np.int16), name='const_255_i16'))

        # Cast (to uint8)
        cast_u8_node = model.graph.node.add(name=self.name + '_castu8', op_type='Cast')
        cast_u8_node.input.append(self.name + '_clip_output')
        cast_u8_node.output.append(output_name)
        cast_u8_node.attribute.add(name='to', type=onnx.AttributeProto.INT, i=onnx.TensorProto.UINT8)

        return model, output_name


class QuantizedConvStridedBNRelu(tf.keras.layers.Layer):
    def __init__(self,
                 filters: int,
                 kernel_size: Tuple[int, int] = (2, 2),
                 padding: str = 'valid',
                 kernel_quantization_bits: int = 8,
                 momentum: float = 0.99,
                 epsilon: float = 1e-3,
                 kernel_initializer="he_uniform",
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 beta_initializer="zeros",
                 moving_mean_initializer="zeros",
                 moving_variance_initializer="ones",
                 **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.kernel_quantization_bits = kernel_quantization_bits
        self.momentum = momentum
        self.epsilon = epsilon
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.moving_mean_initializer = tf.keras.initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = tf.keras.initializers.get(moving_variance_initializer)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(*self.kernel_size, input_shape[-1], self.filters),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype
        )
        self.moving_mean = self.add_weight(
            name="moving_mean",
            shape=(self.filters,),
            dtype=self.dtype,
            initializer=self.moving_mean_initializer,
            trainable=False
        )
        self.moving_variance = self.add_weight(
            name="moving_variance",
            shape=(self.filters,),
            dtype=self.dtype,
            initializer=self.moving_variance_initializer,
            trainable=False
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(self.filters,),
            dtype=self.dtype,
            initializer=self.beta_initializer,
            trainable=True
        )

        self.built = True

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0]] + [s // k for s, k in zip(input_shape[1:], self.kernel_size)] + [self.filters])

    def get_quantized_weights(self):
        batch_norm_factors = tf.math.rsqrt(self.moving_variance + self.epsilon)
        kernel = self.kernel * batch_norm_factors[None, None, None, :]

        quantized_scale = 2. ** (self.kernel_quantization_bits - 1)
        quantized_kernel_max = quantized_scale - 1.
        quantized_kernel_min = -quantized_kernel_max

        kernel_norm = tf.abs(tf.reduce_sum(tf.stop_gradient(kernel), axis=(0,1,3)))

        quantized_kernel = tf.clip_by_value(round_with_gradients(quantized_kernel_max * tf.math.divide_no_nan(kernel, kernel_norm[None, None, :, None])), quantized_kernel_min, quantized_kernel_max)
        quantized_offsets = clip_i16_with_gradients(round_with_gradients(256. * self.beta - tf.math.divide_no_nan(self.moving_mean * batch_norm_factors * (quantized_scale / quantized_kernel_max), kernel_norm)))

        return quantized_kernel, quantized_scale, quantized_offsets

    def call(self, inputs, training=None, *args, **kwargs):
        if not self.trainable:
            training = False
        elif training is None:
            training = tf.keras.backend.learning_phase()

        if training:
            # Calculate the convolution once with floats to get the batch norm statistics
            conv_result = tf.nn.conv2d(inputs, filters=tf.stop_gradient(self.kernel), strides=[1, *self.kernel_size, 1], padding=self.padding.upper())
            mean = tf.reduce_mean(conv_result, axis=(0,1,2))
            variance = tf.math.reduce_variance(conv_result, axis=(0,1,2))

            # Update the BN statistics
            self.moving_mean.assign_sub((self.moving_mean - mean) * (1. - self.momentum))
            self.moving_variance.assign_sub((self.moving_variance - variance) * (1. - self.momentum))

        # Build the quantized weights
        quantized_kernel, quantized_scale, quantized_offsets = self.get_quantized_weights()

        # Perform calculation
        conv_result = tf.nn.conv2d(inputs, filters=quantized_kernel, strides=[1, *self.kernel_size, 1], padding=self.padding.upper())
        conv_result = clip_i16_with_gradients(conv_result)
        conv_result = round_with_gradients(conv_result / quantized_scale)
            
        return tf.clip_by_value(conv_result + quantized_offsets, 0., 255.)
    
    def to_keras_layer(self) -> tf.keras.layers.Conv2D:
        assert self.built

        quantized_kernel, quantized_scale, quantized_offsets = self.get_quantized_weights()

        layer = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=self.kernel_size, use_bias=True, activation='relu')
        layer.build(self.input_shape)
        layer.set_weights([
            (quantized_kernel / quantized_scale).numpy(),
            quantized_offsets.numpy()
        ])
        return layer
    
    def add_to_onnx_model(self, model: 'onnx.ModelProto', input_name: str, output_name: Optional[str] = None) -> Tuple['onnx.ModelProto', str]:
        if output_name is None:
            output_name = self.name + '_output'

        quantized_kernel, quantized_scale, quantized_offsets = self.get_quantized_weights()
        
        # Conv
        conv_node = model.graph.node.add(name=self.name + '_conv', op_type='ConvInteger')
        conv_node.input.extend([input_name, self.name + '_weights'])
        conv_node.output.append(self.name + '_conv_output')
        model.graph.initializer.append(numpy_helper.from_array(np.transpose(quantized_kernel.numpy().astype(np.int8), axes=(2, 3, 0, 1)), name=self.name + '_weights'))
        conv_node.attribute.add(name='strides', type=onnx.AttributeProto.INTS).ints.extend(self.kernel_size)

        # Clip (to int16)
        clip_i16_node = model.graph.node.add(name=self.name + '_clipi16', op_type='Clip')
        clip_i16_node.input.extend([self.name + '_conv_output', 'const_int16_min', 'const_int16_max'])
        clip_i16_node.output.append(self.name + '_clipi16_output')
        if not any(i.name == 'const_int16_min' for i in model.graph.initializer):
            model.graph.initializer.append(numpy_helper.from_array(np.array(-32768, dtype=np.int32), name='const_int16_min'))
        if not any(i.name == 'const_int16_max' for i in model.graph.initializer):
            model.graph.initializer.append(numpy_helper.from_array(np.array(32767, dtype=np.int32), name='const_int16_max'))

        # Cast (to int16)
        cast_i16_node = model.graph.node.add(name=self.name + '_casti16', op_type='Cast')
        cast_i16_node.input.append(self.name + '_clipi16_output')
        cast_i16_node.output.append(self.name + '_casti16_output')
        cast_i16_node.attribute.add(name='to', type=onnx.AttributeProto.INT, i=onnx.TensorProto.INT16)

        # Div (by scale)
        quantized_scale = int(quantized_scale)
        div_node = model.graph.node.add(name=self.name + '_div', op_type='Div')
        div_node.input.extend([self.name + '_casti16_output', f'const_{quantized_scale}_i16'])
        div_node.output.append(self.name + '_div_output')
        if not any(i.name == f'const_{quantized_scale}_i16' for i in model.graph.initializer):
            model.graph.initializer.append(numpy_helper.from_array(np.array(quantized_scale, dtype=np.int16), name=f'const_{quantized_scale}_i16'))

        # Add (offsets)
        add_node = model.graph.node.add(name=self.name + '_add', op_type='Add')
        add_node.input.extend([self.name + '_div_output', self.name + '_offsets'])
        add_node.output.append(self.name + '_add_output')
        model.graph.initializer.append(numpy_helper.from_array(quantized_offsets.numpy().astype(np.int16)[None, :, None, None], name=self.name + '_offsets'))

        # Clip (ReLU)
        clip_node = model.graph.node.add(name=self.name + '_clip', op_type='Clip')
        clip_node.input.extend([self.name + '_add_output', 'const_0_i16', 'const_255_i16'])
        clip_node.output.append(self.name + '_clip_output')
        if not any(i.name == 'const_0_i16' for i in model.graph.initializer):
            model.graph.initializer.append(numpy_helper.from_array(np.array(0, dtype=np.int16), name='const_0_i16'))
        if not any(i.name == 'const_255_i16' for i in model.graph.initializer):
            model.graph.initializer.append(numpy_helper.from_array(np.array(255, dtype=np.int16), name='const_255_i16'))

        # Cast (to uint8)
        cast_u8_node = model.graph.node.add(name=self.name + '_castu8', op_type='Cast')
        cast_u8_node.input.append(self.name + '_clip_output')
        cast_u8_node.output.append(output_name)
        cast_u8_node.attribute.add(name='to', type=onnx.AttributeProto.INT, i=onnx.TensorProto.UINT8)

        return model, output_name


class QuantizedConvBNRelu(tf.keras.layers.Layer):
    def __init__(self,
                 filters: int,
                 kernel_size: Tuple[int, int] = (3, 3),
                 padding: str = 'valid',
                 kernel_quantization_bits: int = 8,
                 momentum: float = 0.99,
                 epsilon: float = 1e-3,
                 kernel_initializer="he_uniform",
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 beta_initializer="zeros",
                 moving_mean_initializer="zeros",
                 moving_variance_initializer="ones",
                 **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.kernel_quantization_bits = kernel_quantization_bits
        self.momentum = momentum
        self.epsilon = epsilon
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.moving_mean_initializer = tf.keras.initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = tf.keras.initializers.get(moving_variance_initializer)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(*self.kernel_size, input_shape[-1], self.filters),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype
        )
        self.moving_mean = self.add_weight(
            name="moving_mean",
            shape=(self.filters,),
            dtype=self.dtype,
            initializer=self.moving_mean_initializer,
            trainable=False
        )
        self.moving_variance = self.add_weight(
            name="moving_variance",
            shape=(self.filters,),
            dtype=self.dtype,
            initializer=self.moving_variance_initializer,
            trainable=False
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(self.filters,),
            dtype=self.dtype,
            initializer=self.beta_initializer,
            trainable=True
        )

        self.built = True

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([*input_shape[:-1], self.filters] if self.padding == 'same' else [input_shape[0]] + [s - (k + 1) // 2 for s, k in zip(input_shape[1:], self.kernel_size)] + [self.filters])

    def get_quantized_weights(self):
        batch_norm_factors = tf.math.rsqrt(self.moving_variance + self.epsilon)
        kernel = self.kernel * batch_norm_factors[None, None, None, :]

        quantized_scale = 2. ** (self.kernel_quantization_bits - 1)
        quantized_kernel_max = quantized_scale - 1.
        quantized_kernel_min = -quantized_kernel_max

        kernel_norm = tf.abs(tf.reduce_sum(tf.stop_gradient(kernel), axis=(0,1,2)))

        quantized_kernel = tf.clip_by_value(round_with_gradients(quantized_kernel_max * tf.math.divide_no_nan(kernel, kernel_norm[None, None, None, :])), quantized_kernel_min, quantized_kernel_max)
        quantized_offsets = clip_i16_with_gradients(round_with_gradients(256. * self.beta - tf.math.divide_no_nan(self.moving_mean * batch_norm_factors * (quantized_scale / quantized_kernel_max), kernel_norm)))

        return quantized_kernel, quantized_scale, quantized_offsets

    def call(self, inputs, training=None, *args, **kwargs):
        if not self.trainable:
            training = False
        elif training is None:
            training = tf.keras.backend.learning_phase()

        if training:
            # Calculate the convolution once with floats to get the batch norm statistics
            conv_result = tf.nn.conv2d(inputs, filters=tf.stop_gradient(self.kernel), strides=[1, 1, 1, 1], padding=self.padding.upper())
            mean = tf.reduce_mean(conv_result, axis=(0,1,2))
            variance = tf.math.reduce_variance(conv_result, axis=(0,1,2))

            # Update the BN statistics
            self.moving_mean.assign_sub((self.moving_mean - mean) * (1. - self.momentum))
            self.moving_variance.assign_sub((self.moving_variance - variance) * (1. - self.momentum))

        # Build the quantized weights
        quantized_kernel, quantized_scale, quantized_offsets = self.get_quantized_weights()

        # Perform calculation
        conv_result = tf.nn.conv2d(inputs, filters=quantized_kernel, strides=[1, 1, 1, 1], padding=self.padding.upper())
        conv_result = clip_i16_with_gradients(conv_result)
        conv_result = round_with_gradients(conv_result / quantized_scale)
            
        return tf.clip_by_value(conv_result + quantized_offsets, 0., 255.)
    
    def to_keras_layer(self) -> tf.keras.layers.Conv2D:
        assert self.built

        quantized_kernel, quantized_scale, quantized_offsets = self.get_quantized_weights()

        layer = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, use_bias=True, activation='relu')
        layer.build(self.input_shape)
        layer.set_weights([
            (quantized_kernel / quantized_scale).numpy(),
            quantized_offsets.numpy()
        ])
        return layer

    def add_to_onnx_model(self, model: 'onnx.ModelProto', input_name: str, output_name: Optional[str] = None) -> Tuple['onnx.ModelProto', str]:
        if output_name is None:
            output_name = self.name + '_output'

        quantized_kernel, quantized_scale, quantized_offsets = self.get_quantized_weights()
        
        # Conv
        conv_node = model.graph.node.add(name=self.name + '_conv', op_type='ConvInteger')
        conv_node.input.extend([input_name, self.name + '_weights'])
        conv_node.output.append(self.name + '_conv_output')
        model.graph.initializer.append(numpy_helper.from_array(np.transpose(quantized_kernel.numpy().astype(np.int8), axes=(2, 3, 0, 1)), name=self.name + '_weights'))

        # Clip (to int16)
        clip_i16_node = model.graph.node.add(name=self.name + '_clipi16', op_type='Clip')
        clip_i16_node.input.extend([self.name + '_conv_output', 'const_int16_min', 'const_int16_max'])
        clip_i16_node.output.append(self.name + '_clipi16_output')
        if not any(i.name == 'const_int16_min' for i in model.graph.initializer):
            model.graph.initializer.append(numpy_helper.from_array(np.array(-32768, dtype=np.int32), name='const_int16_min'))
        if not any(i.name == 'const_int16_max' for i in model.graph.initializer):
            model.graph.initializer.append(numpy_helper.from_array(np.array(32767, dtype=np.int32), name='const_int16_max'))

        # Cast (to int16)
        cast_i16_node = model.graph.node.add(name=self.name + '_casti16', op_type='Cast')
        cast_i16_node.input.append(self.name + '_clipi16_output')
        cast_i16_node.output.append(self.name + '_casti16_output')
        cast_i16_node.attribute.add(name='to', type=onnx.AttributeProto.INT, i=onnx.TensorProto.INT16)

        # Div (by scale)
        quantized_scale = int(quantized_scale)
        div_node = model.graph.node.add(name=self.name + '_div', op_type='Div')
        div_node.input.extend([self.name + '_casti16_output', f'const_{quantized_scale}_i16'])
        div_node.output.append(self.name + '_div_output')
        if not any(i.name == f'const_{quantized_scale}_i16' for i in model.graph.initializer):
            model.graph.initializer.append(numpy_helper.from_array(np.array(quantized_scale, dtype=np.int16), name=f'const_{quantized_scale}_i16'))

        # Add (offsets)
        add_node = model.graph.node.add(name=self.name + '_add', op_type='Add')
        add_node.input.extend([self.name + '_div_output', self.name + '_offsets'])
        add_node.output.append(self.name + '_add_output')
        model.graph.initializer.append(numpy_helper.from_array(quantized_offsets.numpy().astype(np.int16)[None, :, None, None], name=self.name + '_offsets'))

        # Clip (ReLU)
        clip_node = model.graph.node.add(name=self.name + '_clip', op_type='Clip')
        clip_node.input.extend([self.name + '_add_output', 'const_0_i16', 'const_255_i16'])
        clip_node.output.append(self.name + '_clip_output')
        if not any(i.name == 'const_0_i16' for i in model.graph.initializer):
            model.graph.initializer.append(numpy_helper.from_array(np.array(0, dtype=np.int16), name='const_0_i16'))
        if not any(i.name == 'const_255_i16' for i in model.graph.initializer):
            model.graph.initializer.append(numpy_helper.from_array(np.array(255, dtype=np.int16), name='const_255_i16'))

        # Cast (to uint8)
        cast_u8_node = model.graph.node.add(name=self.name + '_castu8', op_type='Cast')
        cast_u8_node.input.append(self.name + '_clip_output')
        cast_u8_node.output.append(output_name)
        cast_u8_node.attribute.add(name='to', type=onnx.AttributeProto.INT, i=onnx.TensorProto.UINT8)

        return model, output_name


class MaxPool2D(tf.keras.layers.MaxPool2D):
    def to_keras_layer(self) -> tf.keras.layers.MaxPool2D:
        return tf.keras.layers.MaxPool2D.from_config(self.get_config())
    
    def add_to_onnx_model(self, model: 'onnx.ModelProto', input_name: str, output_name: Optional[str] = None) -> Tuple['onnx.ModelProto', str]:
        if output_name is None:
            output_name = self.name + '_output'

        node = model.graph.node.add(name=self.name, op_type='MaxPool')
        node.input.append(input_name)
        node.output.append(output_name)
        node.attribute.add(name='kernel_shape', type=onnx.AttributeProto.INTS).ints.extend(self.pool_size)
        node.attribute.add(name='strides', type=onnx.AttributeProto.INTS).ints.extend(self.strides)
        return model, output_name


class AvgPool2D(tf.keras.layers.AvgPool2D):
    def to_keras_layer(self) -> tf.keras.layers.AvgPool2D:
        return tf.keras.layers.AvgPool2D.from_config(self.get_config())
    
    def add_to_onnx_model(self, model: 'onnx.ModelProto', input_name: str, output_name: Optional[str] = None) -> Tuple['onnx.ModelProto', str]:
        if output_name is None:
            output_name = self.name + '_output'

        node = model.graph.node.add(name=self.name, op_type='AveragePool')
        node.input.append(input_name)
        node.output.append(output_name)
        node.attribute.add(name='kernel_shape', type=onnx.AttributeProto.INTS).ints.extend(self.pool_size)
        node.attribute.add(name='strides', type=onnx.AttributeProto.INTS).ints.extend(self.strides)
        return model, output_name

class Flatten(tf.keras.layers.Flatten):
    def to_keras_layer(self) -> tf.keras.layers.Flatten:
        return tf.keras.layers.Flatten.from_config(self.get_config())

    def add_to_onnx_model(self, model: 'onnx.ModelProto', input_name: str, output_name: Optional[str] = None) -> Tuple['onnx.ModelProto', str]:
        if output_name is None:
            output_name = self.name + '_output'

        node = model.graph.node.add(name=self.name, op_type='Flatten')
        node.input.append(input_name)
        node.output.append(output_name)
        return model, output_name

class Sequential(tf.keras.Sequential):
    def add_to_onnx_model(self, model: 'onnx.ModelProto', input_name: str, output_name: Optional[str] = None) -> Tuple['onnx.ModelProto', str]:
        tensor_name = input_name

        layers_to_export = [layer for layer in self.layers if hasattr(layer, 'add_to_onnx_model')]
        if len(layers_to_export) == 0:
            return model, tensor_name

        for layer in layers_to_export[:-1]:
            model, tensor_name = layer.add_to_onnx_model(model, tensor_name)
        return layers_to_export[-1].add_to_onnx_model(model, tensor_name, output_name)

    def to_quantized_onnx_model(self, output_shape: Sequence[Union[int, str]] = ["b", "?", "?", "?"], output_type: 'onnx.TensorProto.DataType' = onnx.TensorProto.UINT8 if 'onnx' in globals() else 0, input_name: str = 'input', output_name: str = 'output') -> Tuple['onnx.ModelProto', str]:
        onnx_model = onnx.ModelProto()

        # Fill in meta info
        onnx_model.ir_version = onnx.IR_VERSION
        onnx_model.producer_name = "train.ipynb"
        onnx_model.producer_version = "1"
        onnx_model.opset_import.add(domain="", version=14)
        onnx_model.graph.name = self.name

        # Add input
        input_spec = onnx_model.graph.input.add(name=input_name)
        input_spec.type.tensor_type.elem_type = onnx.TensorProto.UINT8
        input_spec.type.tensor_type.shape.dim.extend(tensor_shape_to_onnx_dimensions([int(dim_shape) if dim_shape is not None else dim for dim, dim_shape in zip("bcyx", [self.input_shape[0], self.input_shape[-1], *self.input_shape[1:-1]])]))

        # Add layers
        onnx_model, output_name = self.add_to_onnx_model(onnx_model, input_name, output_name)

        # Add output
        output_spec = onnx_model.graph.output.add(name=output_name)
        output_spec.type.tensor_type.elem_type = output_type
        output_spec.type.tensor_type.shape.dim.extend(tensor_shape_to_onnx_dimensions(output_shape))

        return onnx_model

def tensor_shape_to_onnx_dimensions(shape: Sequence[Union[int, str]]):
    return [onnx.TensorShapeProto.Dimension(dim_value=dim_shape) if isinstance(dim_shape, int) else onnx.TensorShapeProto.Dimension(dim_param=dim_shape) for dim_shape in shape]
