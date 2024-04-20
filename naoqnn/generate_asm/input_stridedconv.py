# SPDX-License-Identifier: MIT
import numpy as np
from ..layers import QuantizedConvStridedBNRelu
from .state import LabelManager, State


def asm_input_strided4x4_8(layer: QuantizedConvStridedBNRelu, label_manager: LabelManager, state: State):
    kernel, scale, offsets = layer.get_quantized_weights()
    kernel, scale, offsets = kernel.numpy().astype(np.int8), np.log2(scale).round().astype(np.uint8), offsets.numpy().astype(np.int16)

    assert (state.height % 4) == 0 and (state.width % 16) == 0

    # Store biases
    state.data += f"""
    .short {', '.join(str(x) for x in np.ravel(offsets))}"""

    # Store weights
    state.data += f"""
    .byte {', '.join(str(x) for x in np.ravel(np.transpose(kernel.squeeze(2), axes=(0,2,1))))}"""

    # Make r11 the output_ptr
    state.text += """
    movq %rdx, %r11"""

    # Load biases into XMM12
    state.text += f"""
    movdqa {state.data_offset}(%rax), %xmm12
    add ${state.data_offset + 0x10}, %rax"""

    # Loop over all rows of all samples (R9D)
    if (2 ** np.log2(state.height // 4).round().astype(int)) == (state.height // 4):
        state.text += """
    movl %r8d, %r9d"""
        if (state.height // 4) > 1:
            state.text += f"""
    sall ${int(np.log2(state.height // 4).round().astype(int))}, %r9d"""
    else:
        state.text += f"""
    imull ${state.height // 4}, %r8d, %r9d"""
    row_loop = label_manager.get_next_label()
    state.text += f"""
{row_loop}:"""

    # Loop over cols (ECX)
    if state.width > 16:
        col_loop = label_manager.get_next_label()
        state.text += f"""
    movl ${state.width // 16}, %ecx
{col_loop}:"""

    # Load 16 pixels (4 output pixels) from 4 consecutive rows
    state.text += f"""
    movdqa (%rdi), %xmm8
    movdqa {state.width}(%rdi), %xmm9
    movdqa {2 * state.width}(%rdi), %xmm10
    movdqa {3 * state.width}(%rdi), %xmm11"""

    def calc_convolution_for_pixel(pixel_id: int):
        return f"""
    movdqa %xmm8, %xmm0
    movdqa %xmm9, %xmm2
    movdqa %xmm10, %xmm4
    movdqa %xmm11, %xmm6
    shufps ${pixel_id | (pixel_id << 2) | (pixel_id << 4) | (pixel_id << 6)}, %xmm0, %xmm0
    shufps ${pixel_id | (pixel_id << 2) | (pixel_id << 4) | (pixel_id << 6)}, %xmm2, %xmm2
    shufps ${pixel_id | (pixel_id << 2) | (pixel_id << 4) | (pixel_id << 6)}, %xmm4, %xmm4
    shufps ${pixel_id | (pixel_id << 2) | (pixel_id << 4) | (pixel_id << 6)}, %xmm6, %xmm6
    movdqa %xmm0, %xmm1
    movdqa %xmm2, %xmm3
    movdqa %xmm4, %xmm5
    movdqa %xmm6, %xmm7
    pmaddubsw (%rax), %xmm0
    pmaddubsw 0x10(%rax), %xmm1
    pmaddubsw 0x20(%rax), %xmm2
    pmaddubsw 0x30(%rax), %xmm3
    pmaddubsw 0x40(%rax), %xmm4
    pmaddubsw 0x50(%rax), %xmm5
    pmaddubsw 0x60(%rax), %xmm6
    pmaddubsw 0x70(%rax), %xmm7
    paddsw %xmm2, %xmm0
    paddsw %xmm3, %xmm1
    paddsw %xmm6, %xmm4
    paddsw %xmm7, %xmm5
    paddsw %xmm4, %xmm0
    paddsw %xmm5, %xmm1
    phaddsw %xmm1, %xmm0
    psraw ${scale}, %xmm0
    paddsw %xmm12, %xmm0"""

    # Calculate pixels 0 and 1
    state.text += calc_convolution_for_pixel(0)
    state.text += """
    movdqa %xmm0, %xmm13"""
    state.text += calc_convolution_for_pixel(1)
    state.text += """
    packuswb %xmm0, %xmm13
    movdqa %xmm13, (%r11)"""

    # Calculate pixels 2 and 3
    state.text += calc_convolution_for_pixel(2)
    state.text += """
    movdqa %xmm0, %xmm13"""
    state.text += calc_convolution_for_pixel(3)
    state.text += """
    packuswb %xmm0, %xmm13
    movdqa %xmm13, 0x10(%r11)
    add $0x20, %r11"""

    # Next col
    if state.width > 16:
        state.text += f"""
    add $0x10, %rdi
    dec %ecx
    jnz {col_loop}"""

    # Move to next row (cursor is currently at first pixel of first skipped row)
    state.text += f"""
    add ${state.width * 3}, %rdi"""

    # Next row
    state.text += f"""
    dec %r9d
    jnz {row_loop}"""

    # Reset pointers
    state.text += """
    movq %rdx, %rdi
    movq %rdx, %r11"""

    state.width //= 4
    state.height //= 4
    state.data_offset = 0x80
