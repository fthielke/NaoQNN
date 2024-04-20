# SPDX-License-Identifier: MIT
import numpy as np
from ..layers import QuantizedConv1x1Scale
from .state import LabelManager, State


def asm_dense_16_scale_output(layer: QuantizedConv1x1Scale, label_manager: LabelManager, state: State):
    kernel, scale, offsets = layer.get_quantized_weights()
    kernel = kernel.numpy().astype(np.int8)
    scale_factors = np.array([(out_max - out_min) / 255. for (out_min, out_max) in layer.output_ranges] + [0], dtype=np.float32)
    scaled_offsets = np.concatenate([offsets.numpy(), [0.]], axis=0) * scale_factors + np.array([out_min for (out_min, out_max) in layer.output_ranges] + [0], dtype=np.float32)
    scale_factors = scale_factors / scale

    # Store weights
    state.data += f"""
    .byte {', '.join(str(x) for x in np.ravel(kernel, order='F'))}"""

    # Store scale parameters
    state.data += f"""
    .single {', '.join(f'{factor:0.40f}' for factor in np.ravel(scale_factors))}"""
    state.data += f"""
    .single {', '.join(f'{offset:0.40f}' for offset in np.ravel(scaled_offsets))}"""

    # Load weights and scale parameters into registers
    state.text += f"""
    movdqa {state.data_offset}(%rax), %xmm4
    movdqa {state.data_offset + 0x10}(%rax), %xmm5
    movdqa {state.data_offset + 0x20}(%rax), %xmm6
    movdqa {state.data_offset + 0x30}(%rax), %xmm8
    movdqa {state.data_offset + 0x40}(%rax), %xmm9"""
    state.data_offset += 0x50

    # Loop over pointwise samples (ECX)
    loop = label_manager.get_next_label()
    if (state.height * state.width) > 1:
        state.text += f"""
    imull ${state.height * state.width}, %r8d, %ecx"""
    else:
        state.text += """
    movl %r8d, %ecx"""
    state.text += f"""
{loop}:"""

    # Calculate output channels
    state.text += """
    movdqa (%rdx), %xmm0
    movdqa %xmm0, %xmm1
    movdqa %xmm0, %xmm2
    pmaddubsw %xmm4, %xmm0
    pmaddubsw %xmm5, %xmm1
    pmaddubsw %xmm6, %xmm2
    movdqa %xmm0, %xmm3
    punpcklwd %xmm1, %xmm0
    punpckhwd %xmm1, %xmm3
    paddsw %xmm3, %xmm0
    movdqa %xmm2, %xmm3
    punpcklwd %xmm8, %xmm2
    punpckhwd %xmm8, %xmm3
    paddsw %xmm3, %xmm2"""

    state.text += """
    movdqa %xmm0, %xmm1
    punpckldq %xmm2, %xmm0
    punpckhdq %xmm2, %xmm1
    paddsw %xmm1, %xmm0"""

    state.text += """
    movdqa %xmm0, %xmm1
    punpcklqdq %xmm8, %xmm0
    punpckhqdq %xmm8, %xmm1
    paddsw %xmm1, %xmm0"""

    # Convert to float
    state.text += """
    punpcklwd %xmm0, %xmm0
    psrad $16, %xmm0
    cvtdq2ps %xmm0, %xmm0"""

    # Multiply by scales
    state.text += """
    mulps %xmm8, %xmm0"""

    # Add offsets
    state.text += """
    addps %xmm9, %xmm0"""

    # Store results
    state.text += """
    movups %xmm0, (%rdi)"""

    # Next sample
    state.text += f"""
    add $0x10, %rdx
    add $12, %rdi
    dec %ecx
    jnz {loop}"""


def asm_dense_64_scale_output(layer: QuantizedConv1x1Scale, label_manager: LabelManager, state: State):
    kernel, scale, offsets = layer.get_quantized_weights()
    kernel = kernel.numpy().astype(np.int8)
    scale_factors = np.array([(out_max - out_min) / 255. for (out_min, out_max) in layer.output_ranges] + [0], dtype=np.float32)
    scaled_offsets = np.concatenate([offsets.numpy(), [0.]], axis=0) * scale_factors + np.array([out_min for (out_min, out_max) in layer.output_ranges] + [0], dtype=np.float32)
    scale_factors = scale_factors / scale

    # Store scale parameters
    state.data += f"""
    .single {', '.join(f'{factor:0.40f}' for factor in np.ravel(scale_factors))}"""
    state.data += f"""
    .single {', '.join(f'{offset:0.40f}' for offset in np.ravel(scaled_offsets))}"""

    # Store weights
    state.data += f"""
    .byte {', '.join(str(x) for x in np.ravel(kernel, order='F'))}"""

    # Load scale parameters into registers
    state.text += f"""
    addq ${state.data_offset + 0x20}, %rax
    movdqa -0x20(%rax), %xmm9
    movdqa -0x10(%rax), %xmm10"""
    state.data_offset = 0xC0

    # Loop over pointwise samples (ECX)
    loop = label_manager.get_next_label()
    if (state.height * state.width) > 1:
        state.text += f"""
    imull ${state.height * state.width}, %r8d, %ecx"""
    else:
        state.text += """
    movl %r8d, %ecx"""
    state.text += f"""
{loop}:"""

    # Calculate output channels
    state.text += """
    movdqa (%rdx), %xmm0
    movdqa 0x10(%rdx), %xmm1
    movdqa 0x20(%rdx), %xmm2
    movdqa 0x30(%rdx), %xmm3
    movdqa %xmm0, %xmm4
    movdqa %xmm1, %xmm5
    movdqa %xmm2, %xmm6
    movdqa %xmm3, %xmm7
    pmaddubsw (%rax), %xmm4
    pmaddubsw 0x10(%rax), %xmm5
    pmaddubsw 0x20(%rax), %xmm6
    pmaddubsw 0x30(%rax), %xmm7
    paddsw %xmm4, %xmm5
    paddsw %xmm6, %xmm7
    paddsw %xmm5, %xmm7
    movdqa %xmm7, %xmm8
    movdqa %xmm0, %xmm4
    movdqa %xmm1, %xmm5
    movdqa %xmm2, %xmm6
    movdqa %xmm3, %xmm7
    pmaddubsw 0x40(%rax), %xmm0
    pmaddubsw 0x50(%rax), %xmm1
    pmaddubsw 0x60(%rax), %xmm2
    pmaddubsw 0x70(%rax), %xmm3
    pmaddubsw 0x80(%rax), %xmm4
    pmaddubsw 0x90(%rax), %xmm5
    pmaddubsw 0xA0(%rax), %xmm6
    pmaddubsw 0xB0(%rax), %xmm7
    paddsw %xmm1, %xmm0
    paddsw %xmm3, %xmm2
    paddsw %xmm5, %xmm4
    paddsw %xmm7, %xmm6
    paddsw %xmm2, %xmm0
    paddsw %xmm6, %xmm4"""

    # Partially accumulated word results per output channel are now in XMM8, XMM0 and XMM4

    state.text += """
    movdqa %xmm8, %xmm1
    punpcklwd %xmm0, %xmm1
    punpckhwd %xmm0, %xmm8
    paddsw %xmm8, %xmm1""" # XMM1: 01010101

    state.text += """
    movdqa %xmm4, %xmm2
    movdqa %xmm1, %xmm0
    punpcklwd %xmm2, %xmm2
    punpckhwd %xmm4, %xmm4
    paddsw %xmm4, %xmm2""" # XMM2: 2-2-2-2-

    state.text += """
    punpckldq %xmm2, %xmm0
    punpckhdq %xmm2, %xmm1
    paddsw %xmm1, %xmm0""" # 012-012-

    state.text += """
    movdqa %xmm0, %xmm1
    punpcklwd %xmm0, %xmm0
    punpckhwd %xmm1, %xmm1
    paddsw %xmm1, %xmm0
    psrad $16, %xmm0"""

    # Convert to float
    state.text += """
    cvtdq2ps %xmm0, %xmm0"""

    # Multiply by scales
    state.text += """
    mulps %xmm9, %xmm0"""

    # Add offsets
    state.text += """
    addps %xmm10, %xmm0"""

    # Store results
    state.text += """
    movups %xmm0, (%rdi)"""

    # Next sample
    state.text += f"""
    add $0x40, %rdx
    add $12, %rdi
    dec %ecx
    jnz {loop}"""
