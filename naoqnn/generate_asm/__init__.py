# SPDX-License-Identifier: MIT
import numpy as np

from ..layers import QuantizedConv1x1BNRelu, QuantizedConv1x1Scale, QuantizedConv1x1Softmax, QuantizedConvSpatialBNRelu, QuantizedConvStridedBNRelu, QuantizedConvBNRelu, AvgPool2D, MaxPool2D

from .state import LabelManager, State
from .input_dense import asm_input_layer_8, asm_input_layer_16
from .pool import PoolingMode, asm_pool2x2
from .conv_spatial import asm_conv_spatial2x2_8, asm_conv_spatial2x2_16, asm_conv_spatial3x3_16
from .dense import asm_dense_16, asm_dense_16_32, asm_dense_32_16, asm_dense_32_32, asm_dense_32_64, asm_dense_64_16, asm_dense_8_16, asm_dense_x16_x16
from .input_stridedconv import asm_input_strided4x4_8
from .output_argmax import asm_dense_16_argmax, asm_dense_32_argmax
from .output_scale import asm_dense_16_scale_output, asm_dense_64_scale_output
from .conv import asm_conv_x_16, asm_conv_fullimage, asm_conv_padded_3x3_16_32_4x4
from .input_conv import asm_input_conx3x3_8, asm_input_conx3x3_8_padded


def dispatch_layer(layer, label_manager: LabelManager, state: State, first_layer: bool = False):
    if isinstance(layer, QuantizedConv1x1BNRelu):
        if first_layer:
            if layer.output_shape[-1] == 16:
                asm_input_layer_16(layer, label_manager, state)
            elif layer.output_shape[-1] == 8:
                asm_input_layer_8(layer, label_manager, state)
            else:
                raise NotImplementedError(f"Input dense layer not implemented for {layer.output_shape[-1]} channels")
            first_layer = False
        else:
            if layer.input_shape[-1] == 16 and layer.output_shape[-1] == 16:
                asm_dense_16(layer, label_manager, state)
            elif layer.input_shape[-1] == 8 and layer.output_shape[-1] == 16:
                asm_dense_8_16(layer, label_manager, state)
            elif layer.input_shape[-1] == 16 and layer.output_shape[-1] == 32:
                asm_dense_16_32(layer, label_manager, state)
            elif layer.input_shape[-1] == 32 and layer.output_shape[-1] == 32:
                asm_dense_32_32(layer, label_manager, state)
            elif layer.input_shape[-1] == 32 and layer.output_shape[-1] == 16:
                asm_dense_32_16(layer, label_manager, state)
            elif layer.input_shape[-1] == 32 and layer.output_shape[-1] == 64:
                asm_dense_32_64(layer, label_manager, state)
            elif layer.input_shape[-1] == 64 and layer.output_shape[-1] == 16:
                asm_dense_64_16(layer, label_manager, state)
            else:
                raise NotImplementedError(f"Dense layer {layer.input_shape[-1]}x{layer.output_shape[-1]} not implemented")
    elif isinstance(layer, QuantizedConvSpatialBNRelu):
        if layer.input_shape[-1] == 16 and layer.kernel_size == (3, 3):
            asm_conv_spatial3x3_16(layer, label_manager, state)
        elif layer.input_shape[-1] == 8 and layer.kernel_size == (2, 2):
            asm_conv_spatial2x2_8(layer, label_manager, state)
        elif layer.input_shape[-1] == 16 and layer.kernel_size == (2, 2):
            asm_conv_spatial2x2_16(layer, label_manager, state)
        else:
            raise NotImplementedError(f"Spatial conv layer not implemented for kernel {layer.kernel_size} with {layer.input_shape[-1]} channels")
    elif isinstance(layer, QuantizedConvStridedBNRelu) and first_layer and tuple(layer.kernel_size) == (4, 4) and layer.output_shape[-1] == 8:
        asm_input_strided4x4_8(layer, label_manager, state)
        first_layer = False
    elif isinstance(layer, QuantizedConvBNRelu):
        if first_layer:
            if layer.output_shape[-1] == 8 and layer.kernel_size == (3, 3):
                if layer.padding == 'valid':
                    asm_input_conx3x3_8(layer, label_manager, state)
                else:
                    asm_input_conx3x3_8_padded(layer, label_manager, state)
            else:
                raise NotImplementedError(f"Input conv layer not implemented for kernel {layer.kernel_size} with {layer.output_shape[-1]} channels")
            first_layer = False
        elif state.height == layer.kernel_size[0] and state.width == layer.kernel_size[1] and (layer.output_shape[-1] % 16) == 0 and (layer.input_shape[-1] % 16) == 0 and layer.padding == 'valid':
            asm_conv_fullimage(layer, label_manager, state)
        elif layer.output_shape[-1] == 16:
            asm_conv_x_16(layer, label_manager, state)
        elif layer.padding == 'same' and layer.kernel_size == (3, 3) and layer.input_shape[-1] == 16 and layer.output_shape[-1] == 32 and state.width == 4 and state.height == 4:
            asm_conv_padded_3x3_16_32_4x4(layer, label_manager, state)
        else:
            raise NotImplementedError(f"Conv layer not implemented for {layer.padding} padding, kernel {layer.kernel_size} with {layer.output_shape[-1]} channels")
    elif isinstance(layer, MaxPool2D):
        asm_pool2x2(label_manager, state, PoolingMode.MAX, layer.input_shape[-1])
    elif isinstance(layer, AvgPool2D):
        asm_pool2x2(label_manager, state, PoolingMode.AVG, layer.input_shape[-1])
    elif isinstance(layer, QuantizedConv1x1Scale):
        assert layer.output_shape[-1] == 3
        if layer.input_shape[-1] == 16:
            asm_dense_16_scale_output(layer, label_manager, state)
        elif layer.input_shape[-1] == 64:
            asm_dense_64_scale_output(layer, label_manager, state)
        else:
            raise NotImplementedError(f"Output dense + scale layer not implemented for {layer.input_shape[-1]} input channels")
    else:
        raise NotImplementedError(f"Layer type {layer.__class__.__name__} not implemented")
