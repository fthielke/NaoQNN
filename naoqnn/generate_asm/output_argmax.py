# SPDX-License-Identifier: MIT
from ..layers import QuantizedConv1x1Softmax
from .state import LabelManager, State


def asm_dense_16_argmax(layer: QuantizedConv1x1Softmax, label_manager: LabelManager, state: State, additional_offset: int = 0, output_classification_logits: bool = False):
    kernel, scale, offsets = layer.get_quantized_weights()
    kernel, scale, offsets = kernel.numpy().astype(int), int(scale), offsets.numpy().astype(int)
    assert kernel.shape[1] == 2
    offset = (offsets[0] - offsets[1]) * scale + additional_offset

    # Store weights
    state.data += f"""
    .byte {', '.join(str(channel[1] - channel[0]) for channel in kernel)}
    .short {', '.join(str(offset) for _ in range(8))}"""

    # Load weights into XMM7, offsets into XMM8
    state.text += f"""
    movdqa {state.data_offset}(%rax), %xmm7
    movdqa {state.data_offset + 0x10}(%rax), %xmm8"""
    state.data_offset += 0x20

    # Calculate number of iterations per loop
    after_loop_8samples = label_manager.get_next_label()
    loop_1sample = label_manager.get_next_label()
    if (state.height * state.width) > 1:
        state.text += f"""
    imull ${state.height * state.width}, %r8d, %r9d"""
    else:
        state.text += """
    movl %r8d, %r9d"""

    state.text += f"""
    movl %r9d, %ecx
    andl $7, %ecx
    shrl $3, %r9d
    jz {after_loop_8samples}"""

    # Loop over 8 pointwise samples (R9D)
    loop_8samples = label_manager.get_next_label()
    state.text += f"""
{loop_8samples}:"""

    # Calculate output channels for 8 samples at once
    state.text += """
    movdqa (%rdx), %xmm0
    movdqa 0x10(%rdx), %xmm1
    movdqa 0x20(%rdx), %xmm2
    movdqa 0x30(%rdx), %xmm3
    pmaddubsw %xmm7, %xmm0
    pmaddubsw %xmm7, %xmm1
    pmaddubsw %xmm7, %xmm2
    pmaddubsw %xmm7, %xmm3
    movdqa %xmm0, %xmm4
    movdqa %xmm2, %xmm5
    punpcklwd %xmm1, %xmm0
    punpcklwd %xmm3, %xmm2
    punpckhwd %xmm1, %xmm4
    punpckhwd %xmm3, %xmm5
    paddsw %xmm4, %xmm0
    paddsw %xmm5, %xmm2
    movdqa %xmm0, %xmm1
    punpckldq %xmm2, %xmm0
    punpckhdq %xmm2, %xmm1
    paddsw %xmm1, %xmm0""" # XMM0 contains interleaved word results for samples 0 to 3

    state.text += """
    movdqa 0x40(%rdx), %xmm1
    movdqa 0x50(%rdx), %xmm2
    movdqa 0x60(%rdx), %xmm3
    movdqa 0x70(%rdx), %xmm4
    pmaddubsw %xmm7, %xmm1
    pmaddubsw %xmm7, %xmm2
    pmaddubsw %xmm7, %xmm3
    pmaddubsw %xmm7, %xmm4
    movdqa %xmm1, %xmm5
    movdqa %xmm3, %xmm6
    punpcklwd %xmm2, %xmm1
    punpcklwd %xmm4, %xmm3
    punpckhwd %xmm2, %xmm5
    punpckhwd %xmm4, %xmm6
    paddsw %xmm5, %xmm1
    paddsw %xmm6, %xmm3
    movdqa %xmm1, %xmm2
    punpckldq %xmm3, %xmm1
    punpckhdq %xmm3, %xmm2
    paddsw %xmm2, %xmm1""" # XMM1 contains interleaved word results for samples 4 to 7

    state.text += """
    movdqa %xmm0, %xmm2
    punpcklqdq %xmm1, %xmm0
    punpckhqdq %xmm1, %xmm2
    paddsw %xmm2, %xmm0""" # XMM0 contains word results for samples 0 to 7

    if output_classification_logits:
        state.text += f"""
    paddsw %xmm8, %xmm0
    movdqa %xmm0, (%rsi)"""
    else:
        state.text += f"""
    pcmpgtw %xmm8, %xmm0
    psrlw $15, %xmm0
    packuswb %xmm0, %xmm0
    movdqu %xmm0, (%rsi)"""

    # Next iteration
    state.text += f"""
    add $0x80, %rdx
    add ${16 if output_classification_logits else 8}, %rsi
    dec %r9d
    jnz {loop_8samples}"""

    # Check if another loop is necessary
    after_loop_1sample = label_manager.get_next_label()
    state.text += f"""
{after_loop_8samples}:
    cmpl $0, %ecx
    je {after_loop_1sample}"""

    # Loop over pointwise samples (ECX)
    loop_1sample = label_manager.get_next_label()
    state.text += f"""
{loop_1sample}:"""

    # Calculate output channels
    state.text += f"""
    movdqa (%rdx), %xmm0
    pmaddubsw %xmm7, %xmm0
    movdqa %xmm0, %xmm1
    psrldq $8, %xmm0
    paddsw %xmm1, %xmm0
    movdqa %xmm0, %xmm1
    psrldq $4, %xmm0
    paddsw %xmm1, %xmm0
    movdqa %xmm0, %xmm1
    psrldq $2, %xmm0
    paddsw %xmm1, %xmm0
    pextrw $0, %xmm0, %r9d"""

    if output_classification_logits:
        state.text += f"""
    addw ${offset}, %r9w
    movw %r9w, (%rsi)"""
    else:
        state.text += f"""
    cmpw ${offset}, %r9w
    setg (%rsi)"""

    # Next sample
    state.text += """
    add $0x10, %rdx"""
    if output_classification_logits:
        state.text += """
    add $2, %rsi"""
    else:
        state.text += """
    inc %rsi"""
    state.text += f"""
    dec %ecx
    jnz {loop_1sample}
{after_loop_1sample}:"""
    
    # Cleanup
    state.text += """
    movq %rdi, %rdx"""


def asm_dense_32_argmax(layer: QuantizedConv1x1Softmax, label_manager: LabelManager, state: State, additional_offset: int = 0, output_classification_logits: bool = False):
    kernel, scale, offsets = layer.get_quantized_weights()
    kernel, scale, offsets = kernel.numpy().astype(int), int(scale), offsets.numpy().astype(int)
    assert kernel.shape[1] == 2
    offset = (offsets[0] - offsets[1]) * scale + additional_offset

    # Store weights
    state.data += f"""
    .byte {', '.join(str(channel[1] - channel[0]) for channel in kernel)}
    .short {', '.join(str(offset) for _ in range(8))}"""

    # Load weights into XMM6/7, offsets into XMM15
    state.text += f"""
    movdqa {state.data_offset}(%rax), %xmm6
    movdqa {state.data_offset + 0x10}(%rax), %xmm7
    movdqa {state.data_offset + 0x20}(%rax), %xmm15"""
    state.data_offset += 0x30
    
    # Loop over pointwise samples (ECX)
    loop_1sample = label_manager.get_next_label()
    if (state.height * state.width) > 1:
        state.text += f"""
    imull ${state.height * state.width}, %r8d, %ecx"""
    else:
        state.text += """
    movl %r8d, %ecx"""
    state.text += f"""
{loop_1sample}:"""

    # Calculate output channels
    state.text += f"""
    movdqa (%rdx), %xmm0
    movdqa 0x10(%rdx), %xmm1
    pmaddubsw %xmm6, %xmm0
    pmaddubsw %xmm7, %xmm1
    paddsw %xmm1, %xmm0
    movdqa %xmm0, %xmm1
    psrldq $8, %xmm0
    paddsw %xmm1, %xmm0
    movdqa %xmm0, %xmm1
    psrldq $4, %xmm0
    paddsw %xmm1, %xmm0
    movdqa %xmm0, %xmm1
    psrldq $2, %xmm0
    paddsw %xmm1, %xmm0
    pextrw $0, %xmm0, %r9d"""

    if output_classification_logits:
        state.text += f"""
    addw ${offset}, %r9w
    movw %r9w, (%rsi)"""
    else:
        state.text += f"""
    cmpw ${offset}, %r9w
    setg (%rsi)"""

    # Next sample
    state.text += """
    add $0x20, %rdx"""
    if output_classification_logits:
        state.text += """
    add $2, %rsi"""
    else:
        state.text += """
    inc %rsi"""
    state.text += f"""
    dec %ecx
    jnz {loop_1sample}"""

    # Cleanup
    state.text += """
    movq %rdi, %rdx"""
