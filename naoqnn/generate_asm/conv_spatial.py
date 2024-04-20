# SPDX-License-Identifier: MIT
import numpy as np
from ..layers import QuantizedConvSpatialBNRelu
from .state import LabelManager, State


def asm_conv_spatial3x3_16(layer: QuantizedConvSpatialBNRelu, label_manager: LabelManager, state: State):
    kernel, scale, offsets = layer.get_quantized_weights()
    kernel, scale, offsets = np.squeeze(kernel.numpy().astype(np.int8)), np.log2(scale).round().astype(np.uint8), offsets.numpy().astype(np.int16)

    # Store biases
    state.data += f"""
    .short {', '.join(str(x) for x in np.ravel(offsets))}"""

    # Store weights
    state.data += f"""
    .byte {', '.join(
        f"{kernel[first][channel]}, {0 if second is None else kernel[second][channel]}"
        for first, second in (((0, 0), (0, 1)), ((0, 2), (1, 0)), ((1, 1), (1, 2)), ((2, 0), (2, 1)), ((2, 2), None))
        for channel in range(16)
    )}"""

    # Load some weights and biases into registers, set XMM15 to zero
    state.text += f"""
    add ${state.data_offset + 0x60}, %rax
    movdqa -0x60(%rax), %xmm9
    movdqa -0x50(%rax), %xmm10
    movdqa -0x40(%rax), %xmm4
    movdqa -0x30(%rax), %xmm5
    movdqa -0x20(%rax), %xmm6
    movdqa -0x10(%rax), %xmm7
    pxor %xmm15, %xmm15"""

    # Loop over samples (R10D)
    batch_loop = label_manager.get_next_label()
    state.text += f"""
    movl %r8d, %r10d
{batch_loop}:"""

    # Loop over rows (R9D)
    if state.height > 3:
        row_loop = label_manager.get_next_label()
        state.text += f"""
    movl ${state.height - 2}, %r9d
{row_loop}:"""

    # Loop over cols (ECX)
    if state.width > 3:
        col_loop = label_manager.get_next_label()
        state.text += f"""
    movl ${state.width - 2}, %ecx
{col_loop}:"""

    # Apply filter (top-left and top-middle)
    state.text += """
    movdqa (%rdx), %xmm0
    movdqa 0x10(%rdx), %xmm8
    movdqa %xmm0, %xmm1
    punpcklbw %xmm8, %xmm0
    punpckhbw %xmm8, %xmm1
    pmaddubsw %xmm4, %xmm0
    pmaddubsw %xmm5, %xmm1""" # xmm0 holds partial int16 results 01234567, xmm1 holds partial int16 results 89ABCDEF

    # Apply filter (top-right and middle-left)
    state.text += f"""
    movdqa 0x20(%rdx), %xmm2
    movdqa {state.width * 0x10}(%rdx), %xmm8
    movdqa %xmm2, %xmm3
    punpcklbw %xmm8, %xmm2
    punpckhbw %xmm8, %xmm3
    pmaddubsw %xmm6, %xmm2
    pmaddubsw %xmm7, %xmm3
    paddsw %xmm2, %xmm0
    paddsw %xmm3, %xmm1"""

    # Apply filter (middle-middle and middle-right)
    state.text += f"""
    movdqa {state.width * 0x10 + 0x10}(%rdx), %xmm2
    movdqa {state.width * 0x10 + 0x20}(%rdx), %xmm8
    movdqa %xmm2, %xmm3
    punpcklbw %xmm8, %xmm2
    punpckhbw %xmm8, %xmm3
    pmaddubsw (%rax), %xmm2
    pmaddubsw 0x10(%rax), %xmm3
    paddsw %xmm2, %xmm0
    paddsw %xmm3, %xmm1"""

    # Apply filter (bottom-left and bottom-middle)
    state.text += f"""
    movdqa {2 * state.width * 0x10}(%rdx), %xmm2
    movdqa {2 * state.width * 0x10 + 0x10}(%rdx), %xmm8
    movdqa %xmm2, %xmm3
    punpcklbw %xmm8, %xmm2
    punpckhbw %xmm8, %xmm3
    pmaddubsw 0x20(%rax), %xmm2
    pmaddubsw 0x30(%rax), %xmm3
    paddsw %xmm2, %xmm0
    paddsw %xmm3, %xmm1"""

    # Apply filter (bottom-right)
    state.text += f"""
    movdqa {2 * state.width * 0x10 + 0x20}(%rdx), %xmm2
    movdqa %xmm2, %xmm3
    punpcklbw %xmm15, %xmm2
    punpckhbw %xmm15, %xmm3
    pmaddubsw 0x40(%rax), %xmm2
    pmaddubsw 0x50(%rax), %xmm3
    paddsw %xmm2, %xmm0
    paddsw %xmm3, %xmm1"""

    state.data_offset = 0x60

    # Divide by 2**scale
    state.text += f"""
    psraw ${scale}, %xmm0
    psraw ${scale}, %xmm1"""

    # Add bias
    state.text += """
    paddsw %xmm9, %xmm0
    paddsw %xmm10, %xmm1"""

    # Clip and pack results
    state.text += """
    packuswb %xmm1, %xmm0"""

    # Store results
    state.text += """
    movdqa %xmm0, (%r11)
    add $0x10, %r11"""

    # Next col
    if state.width > 3:
        state.text += f"""
    add $0x10, %rdx
    dec %ecx
    jnz {col_loop}"""

    # Move to next row (cursor is currently at second-to-last pixel in previous row)
    if state.height > 3:
        state.text += """
    add $0x20, %rdx"""

    # Next row
    if state.height > 3:
        state.text += f"""
    dec %r9d
    jnz {row_loop}"""

    # Move to next image
    if state.height > 3:
        # cursor is currently at first pixel of second-to-last row of the previous image
        state.text += f"""
    add ${2 * state.width * 0x10}, %rdx"""
    else:
        # cursor is currently at first/second pixel of the previous image
        state.text += f"""
    add ${state.height * state.width * 0x10 - (0x10 if state.width > 3 else 0)}, %rdx"""

    # Next sample
    state.text += f"""
    dec %r10d
    jnz {batch_loop}"""

    state.width -= 2
    state.height -= 2

    # Reset pointers
    state.text += """
    movq %rdi, %rdx
    movq %rdx, %r11"""


def asm_conv_spatial2x2_8_odd(layer: QuantizedConvSpatialBNRelu, label_manager: LabelManager, state: State):
    kernel, scale, offsets = layer.get_quantized_weights()
    kernel, scale, offsets = np.squeeze(kernel.numpy().astype(np.int8)), np.log2(scale).round().astype(np.uint8), offsets.numpy().astype(np.int16)

    assert (state.width % 2) == 1 and (state.height % 2) == 1

    # Store biases
    state.data += f"""
    .short {', '.join(str(x) for x in np.ravel(offsets))}"""

    # Store weights
    state.data += f"""
    .byte {', '.join(
        f"{kernel[pair[0][0], pair[0][1], channel]}, {kernel[pair[1][0], pair[1][1], channel]}"
        for pair in (((0, 0), (1, 1)), ((0, 1), (1, 0)))
        for channel in range(8)
    )}"""

    # Load biases into registers
    state.text += f"""
    add ${state.data_offset + 0x30}, %rax
    movdqa -0x30(%rax), %xmm5"""
    state.data_offset = 0

    # 0123456789ABCDE
    # FGHIJKLMNOPQRST

    # Loop over samples (R9D)
    batch_loop = label_manager.get_next_label()
    state.text += f"""
    movl %r8d, %r9d
{batch_loop}:"""

    # Loop over two rows at a time (ECX)
    if state.height > 3:
        row_loop = label_manager.get_next_label()
        state.text += f"""
    movl ${(state.height - 1) // 2}, %ecx
{row_loop}:"""

    # First row
    startReg = 0
    writeOffset = 0
    state.text += f"""
    movdqu (%rdx), %xmm0
    movdqu {(state.width - 1) * 8}(%rdx), %xmm8
    movdqa -0x20(%rax), %xmm6
    movdqa -0x10(%rax), %xmm7"""

    for i, readOffset in zip(range((state.width - 1) // 2), range(0x10, state.width * 8, 0x10)):
        bottomLower = 8 if (i % 2) == 0 else 9
        bottomUpper = 9 if (i % 2) == 0 else 8

        state.text += f"""
    movdqu {readOffset}(%rdx), %xmm{(startReg + 3) % 5}
    movdqa %xmm{startReg}, %xmm{(startReg + 1) % 5}
    movdqu {(state.width - 1) * 8 + readOffset}(%rdx), %xmm{bottomUpper}
    movdqa %xmm{startReg}, %xmm{(startReg + 2) % 5}
    movdqa %xmm{(startReg + 3) % 5}, %xmm{(startReg + 4) % 5}
    punpcklbw %xmm{bottomUpper}, %xmm{startReg}
    punpckhbw %xmm{bottomLower}, %xmm{(startReg + 1) % 5}
    punpckhbw %xmm{bottomUpper}, %xmm{(startReg + 2) % 5}
    punpcklbw %xmm{bottomUpper}, %xmm{(startReg + 3) % 5}
    pmaddubsw %xmm6, %xmm{startReg}
    pmaddubsw %xmm7, %xmm{(startReg + 1) % 5}
    pmaddubsw %xmm6, %xmm{(startReg + 2) % 5}
    pmaddubsw %xmm7, %xmm{(startReg + 3) % 5}
    paddsw %xmm{(startReg + 1) % 5}, %xmm{startReg}
    paddsw %xmm{(startReg + 3) % 5}, %xmm{(startReg + 2) % 5}
    psraw ${scale}, %xmm{startReg}
    psraw ${scale}, %xmm{(startReg + 2) % 5}
    paddsw %xmm5, %xmm{startReg}
    paddsw %xmm5, %xmm{(startReg + 2) % 5}
    packuswb %xmm{(startReg + 2) % 5}, %xmm{startReg}
    movdqa %xmm{startReg}, {writeOffset}(%r11)"""

        writeOffset += 0x10
        startReg = (startReg + 4) % 5

    # Second row; startReg still holds EF
    state.text += f"""
    movdqa %xmm6, %xmm10
    movdqa %xmm7, %xmm11
    psllw $8, %xmm6
    psrlw $8, %xmm10
    psllw $8, %xmm7
    psrlw $8, %xmm11
    por %xmm10, %xmm6
    por %xmm11, %xmm7
    movdqa %xmm{startReg}, %xmm8
    movdqu {2 * state.width * 8}(%rdx), %xmm0"""
    startReg = 0

    for i, readOffset in zip(range((state.width - 1) // 2), range(0x10, state.width * 8, 0x10)):
        topLower = 8 if (i % 2) == 0 else 9
        topUpper = 9 if (i % 2) == 0 else 8

        state.text += f"""
    movdqu {2 * state.width * 8 + readOffset}(%rdx), %xmm{(startReg + 3) % 5}
    movdqa %xmm{startReg}, %xmm{(startReg + 1) % 5}
    movdqu {(state.width - 1) * 8 + readOffset}(%rdx), %xmm{topUpper}
    movdqa %xmm{startReg}, %xmm{(startReg + 2) % 5}
    movdqa %xmm{(startReg + 3) % 5}, %xmm{(startReg + 4) % 5}
    punpcklbw %xmm{topUpper}, %xmm{startReg}
    punpckhbw %xmm{topLower}, %xmm{(startReg + 1) % 5}
    punpckhbw %xmm{topUpper}, %xmm{(startReg + 2) % 5}
    punpcklbw %xmm{topUpper}, %xmm{(startReg + 3) % 5}
    pmaddubsw %xmm7, %xmm{startReg}
    pmaddubsw %xmm6, %xmm{(startReg + 1) % 5}
    pmaddubsw %xmm7, %xmm{(startReg + 2) % 5}
    pmaddubsw %xmm6, %xmm{(startReg + 3) % 5}
    paddsw %xmm{(startReg + 1) % 5}, %xmm{startReg}
    paddsw %xmm{(startReg + 3) % 5}, %xmm{(startReg + 2) % 5}
    psraw ${scale}, %xmm{startReg}
    psraw ${scale}, %xmm{(startReg + 2) % 5}
    paddsw %xmm5, %xmm{startReg}
    paddsw %xmm5, %xmm{(startReg + 2) % 5}
    packuswb %xmm{(startReg + 2) % 5}, %xmm{startReg}
    movdqa %xmm{startReg}, {writeOffset}(%r11)"""

        writeOffset += 0x10
        startReg = (startReg + 4) % 5

    state.text += f"""
    add ${writeOffset}, %r11"""

    # Next pair of rows
    if state.height > 3:
        state.text += f"""
    add ${2 * state.width * 8}, %rdx
    dec %ecx
    jnz {row_loop}"""

    # Next sample
    offset = (state.width if state.height > 3 else 3 * state.width) * 8
    state.text += f"""
    add ${offset}, %rdx
    dec %r9d
    jnz {batch_loop}"""

    state.width -= 1
    state.height -= 1

    # Reset pointers
    state.text += """
    movq %rdi, %rdx
    movq %rdx, %r11"""


def asm_conv_spatial2x2_8(layer: QuantizedConvSpatialBNRelu, label_manager: LabelManager, state: State):
    assert state.width > 2
    assert (state.width % 2) == 1
    if (state.height % 2) == 1:
        asm_conv_spatial2x2_8_odd(layer, label_manager, state)
        return

    kernel, scale, offsets = layer.get_quantized_weights()
    kernel, scale, offsets = kernel.numpy().astype(np.int8), np.log2(scale).round().astype(np.uint8), offsets.numpy().astype(np.int16)

    # Store biases
    state.data += f"""
    .short {', '.join(str(x) for x in np.ravel(offsets))}"""

    # Store weights
    state.data += f"""
    .byte {', '.join(str(x) for x in np.ravel(np.transpose(kernel, axes=(1,2,0,3))))}"""

    # Load weights and biases into registers
    state.text += f"""
    add ${state.data_offset + 0x30}, %rax
    movdqa -0x30(%rax), %xmm8
    movdqa -0x20(%rax), %xmm6
    movdqa -0x10(%rax), %xmm7"""
    state.data_offset = 0

    # Loop over samples (R10D)
    batch_loop = label_manager.get_next_label()
    state.text += f"""
    movl %r8d, %r10d
{batch_loop}:"""

    # Loop over rows (R9D)
    if state.height > 2:
        row_loop = label_manager.get_next_label()
        state.text += f"""
    movl ${state.height - 1}, %r9d
{row_loop}:"""

    # Loop over cols (ECX)
    if state.width > 3:
        col_loop = label_manager.get_next_label()
        state.text += f"""
    movl ${(state.width - 1) // 2}, %ecx
{col_loop}:"""

    # Calculate kernel
    state.text += f"""
    movdqu (%rdx), %xmm0
    movdqu 8(%rdx), %xmm1
    movdqu {state.width * 8}(%rdx), %xmm2
    movdqu {state.width * 8 + 8}(%rdx), %xmm3
    movdqa %xmm0, %xmm4
    movdqa %xmm1, %xmm5
    punpcklbw %xmm2, %xmm0
    punpcklbw %xmm3, %xmm1
    punpckhbw %xmm2, %xmm4
    punpckhbw %xmm3, %xmm5
    pmaddubsw %xmm6, %xmm0
    pmaddubsw %xmm7, %xmm4
    pmaddubsw %xmm6, %xmm1
    pmaddubsw %xmm7, %xmm5
    paddsw %xmm4, %xmm0
    paddsw %xmm5, %xmm1"""

    # Divide by 2**scale
    state.text += f"""
    psraw ${scale}, %xmm0
    psraw ${scale}, %xmm1"""

    # Add bias
    state.text += """
    paddsw %xmm8, %xmm0
    paddsw %xmm8, %xmm1"""

    # Clip and pack results
    state.text += """
    packuswb %xmm1, %xmm0"""

    # Store results
    state.text += """
    movdqa %xmm0, (%r11)
    add $0x10, %r11"""

    # Next column
    if state.width > 3:
        state.text += f"""
    add $0x10, %rdx
    dec %ecx
    jnz {col_loop}"""
    
    # Next row
    if state.height > 2:
        offset = 8
        if state.width == 3:
            offset = state.width * 8
        state.text += f"""
    add ${offset}, %rdx
    dec %r9d
    jnz {row_loop}"""
    
    # Next sample
    if state.height == 2 and state.width == 3:
        offset = 2 * state.width * 8
    elif state.height == 2:
        offset = 2 * state.width * 8 + 8
    else:
        offset = 8 + state.width * 8
    state.text += f"""
    add ${offset}, %rdx
    dec %r10d
    jnz {batch_loop}"""

    state.width -= 1
    state.height -= 1

    # Reset pointers
    state.text += """
    movq %rdi, %rdx
    movq %rdx, %r11"""


def asm_conv_spatial2x2_16(layer: QuantizedConvSpatialBNRelu, label_manager: LabelManager, state: State):
    kernel, scale, offsets = layer.get_quantized_weights()
    kernel, scale, offsets = np.squeeze(kernel.numpy().astype(np.int8)), np.log2(scale).round().astype(np.uint8), offsets.numpy().astype(np.int16)

    # Store biases
    state.data += f"""
    .short {', '.join(str(x) for x in np.ravel(offsets))}"""

    # Store weights
    for first, second in (((0, 0), (0, 1)), ((1, 0), (1, 1))):
        for channel in range(16):
            state.data += f"""
    .byte {kernel[first][channel]}, {kernel[second][channel]}"""

    # Load weights and biases into registers
    state.text += f"""
    add ${state.data_offset + 0x60}, %rax
    movdqa -0x60(%rax), %xmm8
    movdqa -0x50(%rax), %xmm9
    movdqa -0x40(%rax), %xmm4
    movdqa -0x30(%rax), %xmm5
    movdqa -0x20(%rax), %xmm6
    movdqa -0x10(%rax), %xmm7"""
    state.data_offset = 0

    # Loop over samples (R10D)
    batch_loop = label_manager.get_next_label()
    state.text += f"""
    movl %r8d, %r10d
{batch_loop}:"""

    # Loop over rows (R9D)
    if state.height > 2:
        row_loop = label_manager.get_next_label()
        state.text += f"""
    movl ${state.height - 1}, %r9d
{row_loop}:"""

    state.text += f"""
    movdqa (%rdx), %xmm10
    movdqa {state.width * 16}(%rdx), %xmm11"""

    # Loop over cols (ECX)
    if state.width > 2:
        col_loop = label_manager.get_next_label()
        state.text += f"""
    movl ${state.width - 1}, %ecx
{col_loop}:"""

    state.text += f"""
    movdqa %xmm10, %xmm0
    movdqa %xmm11, %xmm2
    movdqa 0x10(%rdx), %xmm10
    movdqa %xmm0, %xmm1
    movdqa {state.width * 16 + 16}(%rdx), %xmm11
    movdqa %xmm2, %xmm3
    punpcklbw %xmm10, %xmm0
    punpcklbw %xmm11, %xmm2
    punpckhbw %xmm10, %xmm1
    punpckhbw %xmm11, %xmm3
    pmaddubsw %xmm4, %xmm0
    pmaddubsw %xmm5, %xmm1
    pmaddubsw %xmm6, %xmm2
    pmaddubsw %xmm7, %xmm3
    paddsw %xmm2, %xmm0
    paddsw %xmm3, %xmm1
    psraw ${scale}, %xmm0
    psraw ${scale}, %xmm1
    paddsw %xmm8, %xmm0
    paddsw %xmm9, %xmm1
    packuswb %xmm1, %xmm0
    movdqa %xmm0, (%r11)
    add $0x10, %r11"""

    # Next col
    if state.width > 2:
        state.text += f"""
    add $0x10, %rdx
    dec %ecx
    jnz {col_loop}"""

    # Next row
    if state.height > 2:
        offset = 0x10 if state.width > 2 else 0x20
        state.text += f"""
    add ${offset}, %rdx
    dec %r9d
    jnz {row_loop}"""
    
    # Next sample
    offset = (state.width * 0x10) if state.height > 2 else ((state.width - 1) * 0x10 if state.width > 2 else 2 * state.width * 0x10)
    state.text += f"""
    add ${offset}, %rdx
    dec %r10d
    jnz {batch_loop}"""

    state.width -= 1
    state.height -= 1

    # Reset pointers
    state.text += """
    movq %rdi, %rdx
    movq %rdx, %r11"""
