# SPDX-License-Identifier: MIT
import numpy as np
from ..layers import QuantizedConv1x1BNRelu
from .state import LabelManager, State
from .copy import asm_copy_memory


def asm_dense_8_16(layer: QuantizedConv1x1BNRelu, label_manager: LabelManager, state: State):
    kernel, scale, offsets = layer.get_quantized_weights()
    kernel, scale, offsets = kernel.numpy().astype(np.int8), np.log2(scale).round().astype(np.uint8), offsets.numpy().astype(np.int16)
    
    assert ((state.width * state.height) % 2) == 0

    # Store biases
    state.data += f"""
    .short {', '.join(str(x) for x in np.ravel(offsets))}"""

    # Store weights
    state.data += f"""
    .byte {', '.join(f'{kernel[input_channel, output_channel]}, {kernel[input_channel+1, output_channel]}, {kernel[input_channel, output_channel+1]}, {kernel[input_channel+1, output_channel+1]}' for output_channel in range(0, kernel.shape[1], 2) for input_channel in range(0, kernel.shape[0], 2))}"""

    # Load biases (8, 9) into registers
    # xmm8 to 15 are not used for weights since the Silvermont CPU suffers a 4 cycle performance penalty when using them in SSSE3 or later instructions
    state.text += f"""
    movdqa {state.data_offset}(%rax), %xmm8
    movdqa {state.data_offset + 0x10}(%rax), %xmm9
    add ${state.data_offset + 0x20}, %rax"""

    # Loop over 2 pointwise samples at a time (ECX)
    if (2 ** np.log2(((state.height * state.width) // 2)).round().astype(int)) == ((state.height * state.width) // 2):
        state.text += """
    movl %r8d, %ecx"""
        if ((state.height * state.width) // 2) > 1:
            state.text += f"""
    sall ${int(np.log2((state.height * state.width) // 2).round().astype(int))}, %ecx"""
    else:
        state.text += f"""
    imull ${(state.height * state.width) // 2}, %r8d, %ecx"""

    # Use rdx as input, r10 as output pointer
    # Since output size > input size, we iterate backwards to avoid overwriting the inputs with our outputs
    state.text += f"""
    movl %ecx, %r9d
    sal $5, %r9
    mov %rdx, %r10
    add %r9, %r10
    sar $1, %r9
    add %r9, %rdx"""

    loop = label_manager.get_next_label()
    state.text += f"""
{loop}:"""

    # Load two pixels into XMM10 and XMM11
    state.text += """
    movdqa -0x10(%rdx), %xmm10
    movdqa %xmm10, %xmm11
    punpcklwd %xmm10, %xmm10
    punpckhwd %xmm11, %xmm11"""
    
    # Define calculation of 8 output channels for both pixels
    def calculate_8_outputs_both_pixels(data_offset, bias_reg):
        return f"""
    movdqa %xmm10, %xmm0
    movdqa %xmm10, %xmm1
    movdqa %xmm11, %xmm2
    movdqa %xmm11, %xmm3
    movdqa {data_offset}(%rax), %xmm4
    movdqa {data_offset + 0x10}(%rax), %xmm5
    pmaddubsw %xmm4, %xmm0
    pmaddubsw %xmm5, %xmm1
    pmaddubsw %xmm4, %xmm2
    pmaddubsw %xmm5, %xmm3
    movdqa %xmm0, %xmm4
    movdqa %xmm2, %xmm5
    punpckldq %xmm1, %xmm0
    punpckldq %xmm3, %xmm2
    punpckhdq %xmm1, %xmm4
    punpckhdq %xmm3, %xmm5
    paddsw %xmm4, %xmm0
    paddsw %xmm5, %xmm2
    movdqa %xmm10, %xmm1
    movdqa %xmm10, %xmm3
    movdqa %xmm11, %xmm4
    movdqa %xmm11, %xmm5
    movdqa {data_offset + 0x20}(%rax), %xmm6
    movdqa {data_offset + 0x30}(%rax), %xmm7
    pmaddubsw %xmm6, %xmm1
    pmaddubsw %xmm7, %xmm3
    pmaddubsw %xmm6, %xmm4
    pmaddubsw %xmm7, %xmm5
    movdqa %xmm1, %xmm6
    movdqa %xmm4, %xmm7
    punpckldq %xmm3, %xmm1
    punpckldq %xmm5, %xmm4
    punpckhdq %xmm3, %xmm6
    punpckhdq %xmm5, %xmm7
    paddsw %xmm6, %xmm1
    paddsw %xmm7, %xmm4
    movdqa %xmm0, %xmm3
    movdqa %xmm2, %xmm5
    punpcklqdq %xmm1, %xmm0
    punpckhqdq %xmm1, %xmm3
    punpcklqdq %xmm4, %xmm2
    punpckhqdq %xmm4, %xmm5
    paddsw %xmm3, %xmm0
    paddsw %xmm5, %xmm2
    psraw ${scale}, %xmm0
    psraw ${scale}, %xmm2
    paddsw %{bias_reg}, %xmm0
    paddsw %{bias_reg}, %xmm2"""

    # Calculate output channels 0 to 7 for both pixels
    state.text += calculate_8_outputs_both_pixels(0, 'xmm8')

    # Copy results to XMM12 and XMM13
    state.text += """
    movdqa %xmm0, %xmm12
    movdqa %xmm2, %xmm13"""

    # Calculate output channels 8 to F for both pixels
    state.text += calculate_8_outputs_both_pixels(0x40, 'xmm9')
    state.data_offset = 0x80

    # Clip and pack results
    state.text += """
    packuswb %xmm0, %xmm12
    packuswb %xmm2, %xmm13"""

    # Store results
    state.text += """
    add $-0x20, %r10
    movdqa %xmm12, (%r10)
    movdqa %xmm13, 0x10(%r10)
    add $-0x10, %rdx"""

    # Next iteration
    state.text += f"""
    dec %ecx
    jnz {loop}
    movq %rdi, %rdx"""

def asm_dense_16(layer: QuantizedConv1x1BNRelu, label_manager: LabelManager, state: State):
    kernel, scale, offsets = layer.get_quantized_weights()
    kernel, scale, offsets = kernel.numpy().astype(np.int8), np.log2(scale).round().astype(np.uint8), offsets.numpy().astype(np.int16)
    
    # Store biases
    state.data += f"""
    .short {', '.join(str(x) for x in np.ravel(offsets))}"""

    # Store weights
    state.data += f"""
    .byte {', '.join(str(x) for x in np.ravel(kernel, order='F'))}"""

    # Load biases and first two weights into registers
    # xmm8 to 15 are not used for weights since the Silvermont CPU suffers a 4 cycle performance penalty when using them in SSSE3 or later instructions
    state.text += f"""
    movdqa {state.data_offset}(%rax), %xmm8
    movdqa {state.data_offset + 0x10}(%rax), %xmm9
    movdqa {state.data_offset + 0x20}(%rax), %xmm6
    movdqa {state.data_offset + 0x30}(%rax), %xmm7
    add ${state.data_offset + 0x40}, %rax"""

    # Loop over pointwise samples (ECX)
    if (2 ** np.log2(state.height * state.width).round().astype(int)) == (state.height * state.width):
        state.text += """
    movl %r8d, %ecx"""
        if state.height * state.width > 1:
            state.text += f"""
    sall ${int(np.log2(state.height * state.width).round().astype(int))}, %ecx"""
    else:
        state.text += f"""
    imull ${state.height * state.width}, %r8d, %ecx"""
    loop = label_manager.get_next_label()
    state.text += f"""
{loop}:"""

    # Load all 16 channels of one pixel into XMM0
    state.text += """
    movdqa (%rdx), %xmm0"""

    # Calculate output channels 0 to 3
    state.text += """
    movdqa %xmm0, %xmm1
    movdqa %xmm0, %xmm3
    pmaddubsw %xmm6, %xmm1
    pmaddubsw %xmm7, %xmm3
    movdqa %xmm1, %xmm2
    punpcklwd %xmm3, %xmm1
    punpckhwd %xmm3, %xmm2
    paddsw %xmm2, %xmm1"""

    state.text += """
    movdqa %xmm0, %xmm2
    movdqa %xmm0, %xmm4
    pmaddubsw (%rax), %xmm2
    pmaddubsw 0x10(%rax), %xmm4
    movdqa %xmm2, %xmm3
    punpcklwd %xmm4, %xmm2
    punpckhwd %xmm4, %xmm3
    paddsw %xmm3, %xmm2"""

    state.text += """
    movdqa %xmm1, %xmm3
    punpckldq %xmm2, %xmm1
    punpckhdq %xmm2, %xmm3
    paddsw %xmm3, %xmm1"""

    # Calculate output channels 4 to 7
    state.text += """
    movdqa %xmm0, %xmm2
    movdqa %xmm0, %xmm4
    pmaddubsw 0x20(%rax), %xmm2
    pmaddubsw 0x30(%rax), %xmm4
    movdqa %xmm2, %xmm3
    punpcklwd %xmm4, %xmm2
    punpckhwd %xmm4, %xmm3
    paddsw %xmm3, %xmm2"""
    
    state.text += """
    movdqa %xmm0, %xmm3
    movdqa %xmm0, %xmm5
    pmaddubsw 0x40(%rax), %xmm3
    pmaddubsw 0x50(%rax), %xmm5
    movdqa %xmm3, %xmm4
    punpcklwd %xmm5, %xmm3
    punpckhwd %xmm5, %xmm4
    paddsw %xmm4, %xmm3"""

    state.text += """
    movdqa %xmm2, %xmm4
    punpckldq %xmm3, %xmm2
    punpckhdq %xmm3, %xmm4
    paddsw %xmm4, %xmm2"""

    # Accumulate output channels 0 to 7
    state.text += """
    movdqa %xmm1, %xmm3
    punpcklqdq %xmm2, %xmm1
    punpckhqdq %xmm2, %xmm3
    paddsw %xmm3, %xmm1"""

    # Divide by 2**scale
    state.text += f"""
    psraw ${scale}, %xmm1"""

    # Add bias
    state.text += """
    paddsw %xmm8, %xmm1"""

    # Calculate output channels 8 to B
    state.text += """
    movdqa %xmm0, %xmm2
    movdqa %xmm0, %xmm4
    pmaddubsw 0x60(%rax), %xmm2
    pmaddubsw 0x70(%rax), %xmm4
    movdqa %xmm2, %xmm3
    punpcklwd %xmm4, %xmm2
    punpckhwd %xmm4, %xmm3
    paddsw %xmm3, %xmm2"""
    
    state.text += """
    movdqa %xmm0, %xmm3
    movdqa %xmm0, %xmm5
    pmaddubsw 0x80(%rax), %xmm3
    pmaddubsw 0x90(%rax), %xmm5
    movdqa %xmm3, %xmm4
    punpcklwd %xmm5, %xmm3
    punpckhwd %xmm5, %xmm4
    paddsw %xmm4, %xmm3"""

    state.text += """
    movdqa %xmm2, %xmm4
    punpckldq %xmm3, %xmm2
    punpckhdq %xmm3, %xmm4
    paddsw %xmm4, %xmm2"""

    # Calculate output channels C to F
    state.text += """
    movdqa %xmm0, %xmm3
    movdqa %xmm0, %xmm5
    pmaddubsw 0xA0(%rax), %xmm3
    pmaddubsw 0xB0(%rax), %xmm5
    movdqa %xmm3, %xmm4
    punpcklwd %xmm5, %xmm3
    punpckhwd %xmm5, %xmm4
    paddsw %xmm4, %xmm3"""

    state.text += """
    movdqa %xmm0, %xmm4
    pmaddubsw 0xC0(%rax), %xmm4
    pmaddubsw 0xD0(%rax), %xmm0
    movdqa %xmm4, %xmm5
    punpcklwd %xmm0, %xmm4
    punpckhwd %xmm0, %xmm5
    paddsw %xmm5, %xmm4"""

    state.text += """
    movdqa %xmm3, %xmm5
    punpckldq %xmm4, %xmm3
    punpckhdq %xmm4, %xmm5
    paddsw %xmm5, %xmm3"""

    # Accumulate output channels 8 to F
    state.text += """
    movdqa %xmm2, %xmm4
    punpcklqdq %xmm3, %xmm2
    punpckhqdq %xmm3, %xmm4
    paddsw %xmm4, %xmm2"""

    # Divide by 2**scale
    state.text += f"""
    psraw ${scale}, %xmm2"""

    # Add bias
    state.text += """
    paddsw %xmm9, %xmm2"""

    # Clip and pack results
    state.text += """
    packuswb %xmm2, %xmm1"""

    # Store results in-place
    state.text += """
    movdqa %xmm1, (%rdx)
    add $0x10, %rdx"""

    # Next sample
    state.text += f"""
    dec %ecx
    jnz {loop}
    movq %rdi, %rdx"""

    state.data_offset = 0xE0


def asm_dense_16_32(layer: QuantizedConv1x1BNRelu, label_manager: LabelManager, state: State):
    kernel, scale, offsets = layer.get_quantized_weights()
    kernel, scale, offsets = kernel.numpy().astype(np.int8), np.log2(scale).round().astype(np.uint8), offsets.numpy().astype(np.int16)
    asm_dense_x16_x16(state.height * state.width, kernel, scale, offsets, label_manager, state)


def asm_dense_32_16(layer: QuantizedConv1x1BNRelu, label_manager: LabelManager, state: State):
    kernel, scale, offsets = layer.get_quantized_weights()
    kernel, scale, offsets = kernel.numpy().astype(np.int8), np.log2(scale).round().astype(np.uint8), offsets.numpy().astype(np.int16)
    asm_dense_x16_x16(state.height * state.width, kernel, scale, offsets, label_manager, state)


def asm_dense_32_32(layer: QuantizedConv1x1BNRelu, label_manager: LabelManager, state: State):
    kernel, scale, offsets = layer.get_quantized_weights()
    kernel, scale, offsets = kernel.numpy().astype(np.int8), np.log2(scale).round().astype(np.uint8), offsets.numpy().astype(np.int16)
    asm_dense_x16_x16(state.height * state.width, kernel, scale, offsets, label_manager, state)


def asm_dense_32_64(layer: QuantizedConv1x1BNRelu, label_manager: LabelManager, state: State):
    kernel, scale, offsets = layer.get_quantized_weights()
    kernel, scale, offsets = kernel.numpy().astype(np.int8), np.log2(scale).round().astype(np.uint8), offsets.numpy().astype(np.int16)
    asm_dense_x16_x16(state.height * state.width, kernel, scale, offsets, label_manager, state)


def asm_dense_64_16(layer: QuantizedConv1x1BNRelu, label_manager: LabelManager, state: State):
    kernel, scale, offsets = layer.get_quantized_weights()
    kernel, scale, offsets = kernel.numpy().astype(np.int8), np.log2(scale).round().astype(np.uint8), offsets.numpy().astype(np.int16)
    asm_dense_x16_x16(state.height * state.width, kernel, scale, offsets, label_manager, state)


def asm_dense_x16_x16(num_samples: int, kernel: np.ndarray, scale: int, offsets: np.ndarray, label_manager: LabelManager, state: State):
    input_channels = kernel.shape[0]
    output_channels = kernel.shape[1]

    assert output_channels in (16, 32, 64)

    # Store biases
    state.data += f"""
    .short {', '.join(str(x) for x in np.ravel(offsets))}"""
    state.text += f"""
    add ${state.data_offset + 2 * np.size(offsets)}, %rax"""

    # Store weights
    input_channel_step = 8 if output_channels == 16 else 4
    slices = (
        (slice(input_channel_start, input_channel_end), output_channel)
        for input_channel_offset in range(0, input_channels, input_channel_step)
        for output_channel_offset in range(0, output_channels, 8)
        for input_channel_start, input_channel_end in zip(range(input_channel_offset, input_channel_offset + input_channel_step, 2), range(input_channel_offset + 2, input_channel_offset + input_channel_step + 2, 2))
        for output_channel in range(output_channel_offset, output_channel_offset + 8)
    )
    state.data += f"""
    .byte {', '.join(str(x) for s in slices for x in np.ravel(kernel[s]))}"""
    state.data_offset = np.size(kernel)

    # Optionally load biases into registers
    if output_channels == 16:
        state.text += f"""
    movdqa -0x20(%rax), %xmm14
    movdqa -0x10(%rax), %xmm15"""
        bias_locations = ('%xmm14', '%xmm15')
    elif output_channels == 32:
        state.text += f"""
    movdqa -0x40(%rax), %xmm10
    movdqa -0x30(%rax), %xmm11"""
        bias_locations = ('%xmm10', '%xmm11', *(f'{i - (2 * np.size(offsets))}(%rax)' for i in range(0x20, 2 * np.size(offsets), 0x10)))
    else:
        bias_locations = tuple(f'{i - (2 * np.size(offsets))}(%rax)' for i in range(0, 2 * np.size(offsets), 0x10))

    # Use R11 as output pointer; set up pointer if operation cannot be inplace
    out_ptr = '%r11'
    if input_channels < output_channels:
        state.text += f"""
    imull ${input_channels}, %r8d, %r11d
    addq %rdx, %r11
    movq %r11, %r12"""
    elif input_channels == output_channels:
        out_ptr = '%rdx'

    # Loop over pointwise samples (ECX)
    if (2 ** np.log2(num_samples).round().astype(int)) == (num_samples):
        state.text += """
    movl %r8d, %ecx"""
        if num_samples > 1:
            state.text += f"""
    sall ${int(np.log2(num_samples).round().astype(int))}, %ecx"""
    else:
        state.text += f"""
    imull ${num_samples}, %r8d, %ecx"""
    loop = label_manager.get_next_label()
    state.text += f"""
{loop}:"""

    load_offsets = range(0, input_channels, 0x10)
    weights_offsets = range(0, input_channels * output_channels, 0x10 * output_channels)
    for load_offset, weights_offset in zip(load_offsets, weights_offsets):
        if output_channels == 16:
            for i, prolog in enumerate((
                f"""
    movdqa {load_offset}(%rdx), %xmm4
    movdqa %xmm4, %xmm8
    punpcklwd %xmm4, %xmm4
    punpckhwd %xmm8, %xmm8
    movdqa %xmm4, %xmm6
    movdqa %xmm8, %xmm10
    punpckldq %xmm4, %xmm4
    punpckhdq %xmm6, %xmm6
    punpckldq %xmm8, %xmm8
    punpckhdq %xmm10, %xmm10
    movdqa %xmm4, %xmm5
    movdqa %xmm6, %xmm7
    movdqa %xmm8, %xmm9
    movdqa %xmm10, %xmm11
    punpcklqdq %xmm4, %xmm4
    punpckhqdq %xmm5, %xmm5
    punpcklqdq %xmm6, %xmm6
    punpckhqdq %xmm7, %xmm7
    punpcklqdq %xmm8, %xmm8
    punpckhqdq %xmm9, %xmm9
    punpcklqdq %xmm10, %xmm10
    punpckhqdq %xmm11, %xmm11
    movdqa %xmm4, %xmm0
    movdqa %xmm5, %xmm1
    movdqa %xmm6, %xmm2
    movdqa %xmm7, %xmm3""",
                """
    movdqa %xmm8, %xmm0
    movdqa %xmm9, %xmm1
    movdqa %xmm10, %xmm2
    movdqa %xmm11, %xmm3
    movdqa %xmm8, %xmm4
    movdqa %xmm9, %xmm5
    movdqa %xmm10, %xmm6
    movdqa %xmm11, %xmm7"""
            )):
                result_op = 'movdqa' if load_offset == 0 and i == 0 else 'paddsw'
                state.text += f"""{prolog}
    pmaddubsw {weights_offset + i * 0x80}(%rax), %xmm0
    pmaddubsw {weights_offset + i * 0x80 + 0x10}(%rax), %xmm1
    pmaddubsw {weights_offset + i * 0x80 + 0x20}(%rax), %xmm2
    pmaddubsw {weights_offset + i * 0x80 + 0x30}(%rax), %xmm3
    pmaddubsw {weights_offset + i * 0x80 + 0x40}(%rax), %xmm4
    pmaddubsw {weights_offset + i * 0x80 + 0x50}(%rax), %xmm5
    pmaddubsw {weights_offset + i * 0x80 + 0x60}(%rax), %xmm6
    pmaddubsw {weights_offset + i * 0x80 + 0x70}(%rax), %xmm7
    paddsw %xmm1, %xmm0
    paddsw %xmm3, %xmm2
    paddsw %xmm5, %xmm4
    paddsw %xmm7, %xmm6
    paddsw %xmm2, %xmm0
    paddsw %xmm6, %xmm4
    {result_op} %xmm0, %xmm12
    {result_op} %xmm4, %xmm13"""
        elif output_channels == 32:
            for i, prolog in enumerate((
                f"""
    movdqa {load_offset}(%rdx), %xmm0
    movdqa %xmm0, %xmm8
    punpcklwd %xmm0, %xmm0
    punpckhwd %xmm8, %xmm8
    movdqa %xmm0, %xmm9
    punpckldq %xmm0, %xmm0
    punpckhdq %xmm9, %xmm9""",
                """
    movdqa %xmm9, %xmm0""",
                """
    movdqa %xmm8, %xmm0
    movdqa %xmm0, %xmm9
    punpckldq %xmm0, %xmm0
    punpckhdq %xmm9, %xmm9""",
                """
    movdqa %xmm9, %xmm0"""
            )):
                result_op = 'movdqa' if load_offset == 0 and i == 0 else 'paddsw'
                state.text += f"""{prolog}
    movdqa %xmm0, %xmm1
    punpcklqdq %xmm0, %xmm0
    punpckhqdq %xmm1, %xmm1
    movdqa %xmm0, %xmm2
    movdqa %xmm1, %xmm3
    movdqa %xmm0, %xmm4
    movdqa %xmm1, %xmm5
    movdqa %xmm0, %xmm6
    movdqa %xmm1, %xmm7
    pmaddubsw {weights_offset + i * 0x80}(%rax), %xmm0
    pmaddubsw {weights_offset + i * 0x80 + 0x10}(%rax), %xmm1
    pmaddubsw {weights_offset + i * 0x80 + 0x20}(%rax), %xmm2
    pmaddubsw {weights_offset + i * 0x80 + 0x30}(%rax), %xmm3
    pmaddubsw {weights_offset + i * 0x80 + 0x40}(%rax), %xmm4
    pmaddubsw {weights_offset + i * 0x80 + 0x50}(%rax), %xmm5
    pmaddubsw {weights_offset + i * 0x80 + 0x60}(%rax), %xmm6
    pmaddubsw {weights_offset + i * 0x80 + 0x70}(%rax), %xmm7
    paddsw %xmm1, %xmm0
    paddsw %xmm3, %xmm2
    paddsw %xmm5, %xmm4
    paddsw %xmm7, %xmm6
    {result_op} %xmm0, %xmm12
    {result_op} %xmm2, %xmm13
    {result_op} %xmm4, %xmm14
    {result_op} %xmm6, %xmm15"""
        elif output_channels == 64:
            for i, prolog in enumerate((
                f"""
    movdqa {load_offset}(%rdx), %xmm0
    movdqa %xmm0, %xmm1
    punpcklwd %xmm0, %xmm0
    punpckhwd %xmm1, %xmm1
    movdqa %xmm0, %xmm2
    punpckldq %xmm0, %xmm0
    punpckhdq %xmm2, %xmm2""",
                """
    movdqa %xmm2, %xmm0""",
                """
    movdqa %xmm1, %xmm0
    punpckldq %xmm0, %xmm0
    punpckhdq %xmm1, %xmm1""",
                """
    movdqa %xmm1, %xmm0"""
            )):
                state.text += f"""{prolog}
    movdqa %xmm0, %xmm3
    punpcklqdq %xmm0, %xmm0
    punpckhqdq %xmm3, %xmm3"""
                result_op = 'movdqa' if load_offset == 0 and i == 0 else 'paddsw'
                for inner_weights_offset, result_reg1, result_reg2 in zip(range(4), range(8, 0x10, 2), range(9, 0x10, 2)):
                    state.text += f"""
    movdqa %xmm0, %xmm4
    movdqa %xmm3, %xmm5
    movdqa %xmm0, %xmm6
    movdqa %xmm3, %xmm7
    pmaddubsw {weights_offset + i * 4 * 0x40 + inner_weights_offset * 0x40}(%rax), %xmm4
    pmaddubsw {weights_offset + i * 4 * 0x40 + inner_weights_offset * 0x40 + 0x10}(%rax), %xmm5
    pmaddubsw {weights_offset + i * 4 * 0x40 + inner_weights_offset * 0x40 + 0x20}(%rax), %xmm6
    pmaddubsw {weights_offset + i * 4 * 0x40 + inner_weights_offset * 0x40 + 0x30}(%rax), %xmm7
    paddsw %xmm5, %xmm4
    paddsw %xmm7, %xmm6
    {result_op} %xmm4, %xmm{result_reg1}
    {result_op} %xmm6, %xmm{result_reg2}"""
        else:
            raise NotImplementedError()

    output_registers = range(12 if output_channels in (16, 32) else 8, 12 + (output_channels // 8) if output_channels in (16, 32) else 16)

    # Divide by 2**scale
    for r in output_registers:
        state.text += f"""
    psraw ${scale}, %xmm{r}"""

    # Add bias
    for r, bias in zip(output_registers, bias_locations):
        state.text += f"""
    paddsw {bias}, %xmm{r}"""

    # Clip and pack results
    for r1, r2 in zip(output_registers[::2], output_registers[1::2]):
        state.text += f"""
    packuswb %xmm{r2}, %xmm{r1}"""

    # Store results
    for offset, r in enumerate(output_registers[::2]):
        state.text += f"""
    movdqa %xmm{r}, {offset * 0x10}({out_ptr})"""

    if out_ptr != '%rdx':
        state.text += f"""
    add ${output_channels}, {out_ptr}"""

    # Next sample
    state.text += f"""
    add ${input_channels}, %rdx
    dec %ecx
    jnz {loop}"""

    # Copy results to start of memory region if operation couldn't be inplace
    if input_channels < output_channels:
        # Reset pointers
        state.text += """
    movq %rdi, %rdx
    movq %r12, %r11"""

        # Copy results to start of memory region
        asm_copy_memory(
            in_ptr='%r11',
            out_ptr='%rdx',
            num_bytes_per_sample=output_channels,
            label_manager=label_manager,
            state=state
        )

    # Reset pointers
    state.text += """
    movq %rdi, %rdx"""
    if out_ptr == '%r11':
        state.text += """
    movq %rdx, %r11"""
