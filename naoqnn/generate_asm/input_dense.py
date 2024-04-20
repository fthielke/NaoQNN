# SPDX-License-Identifier: MIT
import numpy as np
from ..layers import QuantizedConv1x1BNRelu
from .state import LabelManager, State


def asm_input_layer_8(layer: QuantizedConv1x1BNRelu, label_manager: LabelManager, state: State):
    kernel, scale, offsets = layer.get_quantized_weights()
    kernel, scale, offsets = kernel.numpy().astype(np.int16), np.log2(scale).round().astype(np.uint8), offsets.numpy().astype(np.int16)
    state.data += f"""
    .byte 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0
    .byte 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255
    .short {', '.join(str(x) for x in np.ravel(kernel))}
    .short {', '.join(str(x) for x in np.ravel(offsets))}"""

    # Load constants:
    # Counting offset (8), shuffle mask (9), weights (10), biases (11)
    state.text += f"""
    movdqa {state.data_offset}(%rax), %xmm8
    movdqa {state.data_offset + 0x10}(%rax), %xmm9
    movdqa {state.data_offset + 0x20}(%rax), %xmm10
    movdqa {state.data_offset + 0x30}(%rax), %xmm11"""
    state.data_offset += 0x40

    # Make r11 the output_ptr
    state.text += """
    movq %rdx, %r11"""

    # Loop over batchsize * height (32) * width (32) // 16
    # (batchsize is still stored in ecx)
    assert state.width == 32 and state.height == 32
    loop_label = label_manager.get_next_label()
    state.text += f"""
    shll $6, %ecx
{loop_label}:"""

    # Reset shuffle mask in XMM7
    state.text += """
    movdqa %xmm9, %xmm7"""
    
    # Load 16 pixels
    state.text += """
    movdqa (%rdi), %xmm0"""

    # Calculate pixels 0 to 11
    for offset in (0, 0x30):
        # Collect pixels in XMM1 - XMM6
        if offset == 0:
            state.text += """
    movdqa %xmm0, %xmm1
    pshufb %xmm7, %xmm1"""
        for i in range(2 if offset == 0 else 1, 7):
            state.text += f"""
    movdqa %xmm0, %xmm{i}
    paddb %xmm8, %xmm7
    pshufb %xmm7, %xmm{i}"""

        # Multiply with weights, divide by 2**scale, add bias
        for i in range(1, 7):
            state.text += f"""
    pmullw %xmm10, %xmm{i}
    psraw ${scale}, %xmm{i}
    paddsw %xmm11, %xmm{i}"""

        # Clip and pack results
        state.text += """
    packuswb %xmm2, %xmm1
    packuswb %xmm4, %xmm3
    packuswb %xmm6, %xmm5"""

        # Store results
        state.text += f"""
    movdqa %xmm1, {offset}(%r11)
    movdqa %xmm3, {offset + 0x10}(%r11)
    movdqa %xmm5, {offset + 0x20}(%r11)"""

    # Calculate pixels 12 to 15
    for i in range(1, 4):
        state.text += f"""
    movdqa %xmm0, %xmm{i}
    paddb %xmm8, %xmm7
    pshufb %xmm7, %xmm{i}"""
    # The last pixel can directly use XMM0
    state.text += """
    paddb %xmm8, %xmm7
    pshufb %xmm7, %xmm0"""

    # Multiply with weights, divide by 2**scale, add bias
    for i in range(4):
        state.text += f"""
    pmullw %xmm10, %xmm{i}
    psraw ${scale}, %xmm{i}
    paddsw %xmm11, %xmm{i}"""

    # Clip and pack results
    state.text += """
    packuswb %xmm2, %xmm1
    packuswb %xmm0, %xmm3"""

    # Store results
    state.text += f"""
    movdqa %xmm1, 0x60(%r11)
    movdqa %xmm3, 0x70(%r11)"""

    # Loop
    state.text += f"""
    add $0x80, %r11
    add $0x10, %rdi
    dec %ecx
    jnz {loop_label}
    movq %rdx, %rdi
    movq %rdx, %r11"""


def asm_input_layer_16(layer: QuantizedConv1x1BNRelu, label_manager: LabelManager, state: State):
    kernel, scale, offsets = layer.get_quantized_weights()
    kernel, scale, offsets = kernel.numpy().astype(np.int16), np.log2(scale).round().astype(np.uint8), offsets.numpy().astype(np.int16)
    state.data += f"""
    .byte 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0
    .byte 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255
    .short {', '.join(str(x) for x in np.ravel(kernel))}
    .short {', '.join(str(x) for x in np.ravel(offsets))}"""

    # Load constants:
    # Counting offset (7), shuffle mask (10), weights (4/5), biases (8/9)
    state.text += f"""
    movdqa {state.data_offset}(%rax), %xmm7
    movdqa {state.data_offset + 0x10}(%rax), %xmm10
    movdqa {state.data_offset + 0x20}(%rax), %xmm4
    movdqa {state.data_offset + 0x30}(%rax), %xmm5
    movdqa {state.data_offset + 0x40}(%rax), %xmm8
    movdqa {state.data_offset + 0x50}(%rax), %xmm9"""
    state.data_offset += 0x60

    # Make r11 the output_ptr
    state.text += """
    movq %rdx, %r11"""

    # Loop over batchsize * height (32) * width (32) // 16
    # (batchsize is still stored in ecx)
    assert state.width == 32 and state.height == 32
    loop_label = label_manager.get_next_label()
    state.text += f"""
    shll $6, %ecx
{loop_label}:"""

    # Reset shuffle mask
    state.text += """
    movdqa %xmm10, %xmm6"""
    
    # Load 16 pixels
    state.text += """
    movdqa (%rdi), %xmm0"""

    for pixel in range(15):
        # Fill xmm1 and xmm2 with the current pixel value and increase the shuffle mask entries
        state.text += """
    movdqa %xmm0, %xmm1
    pshufb %xmm6, %xmm1
    movdqa %xmm1, %xmm2
    paddb %xmm7, %xmm6"""
        
        # Multiply with weights
        state.text += """
    pmullw %xmm4, %xmm1
    pmullw %xmm5, %xmm2"""
        
        # Divide by 2**scale
        state.text += f"""
    psraw ${scale}, %xmm1
    psraw ${scale}, %xmm2"""
        
        # Add bias
        state.text += """
    paddsw %xmm8, %xmm1
    paddsw %xmm9, %xmm2"""

        # Clip and pack results
        state.text += """
    packuswb %xmm2, %xmm1"""
        
        # Store results
        state.text += f"""
    movdqa %xmm1, {pixel * 0x10}(%r11)"""

    # The last pixel can directly use XMM0
    state.text += """
    pshufb %xmm6, %xmm0
    movdqa %xmm0, %xmm2"""
        
    # Multiply with weights
    state.text += """
    pmullw %xmm4, %xmm0
    pmullw %xmm5, %xmm2"""
        
    # Divide by 2**scale
    state.text += f"""
    psraw ${scale}, %xmm0
    psraw ${scale}, %xmm2"""
        
    # Add bias
    state.text += """
    paddsw %xmm8, %xmm0
    paddsw %xmm9, %xmm2"""
        
    # Clip and pack results
    state.text += """
    packuswb %xmm2, %xmm0"""
        
    # Store results
    state.text += """
    movdqa %xmm0, 0xF0(%r11)
    add $0x100, %r11"""

    # Loop
    state.text += f"""
    add $0x10, %rdi
    dec %ecx
    jnz {loop_label}
    movq %rdx, %rdi
    movq %rdx, %r11"""
