# SPDX-License-Identifier: MIT
import numpy as np
from ..layers import QuantizedConvBNRelu
from .state import LabelManager, State
from .copy import asm_copy_memory, asm_pad_image
from .dense import asm_dense_x16_x16


def asm_conv_x_16(layer: QuantizedConvBNRelu, label_manager: LabelManager, state: State):
    kernel, scale, offsets = layer.get_quantized_weights()
    kernel, scale, offsets = kernel.numpy().astype(np.int8), np.log2(scale).round().astype(np.uint8), offsets.numpy().astype(np.int16)

    num_input_channels = kernel.shape[2]
    num_output_channels = kernel.shape[3]

    if ((kernel.shape[1] * num_input_channels) % 16) == num_input_channels:
        new_kernel = np.zeros((kernel.shape[0] + 1, kernel.shape[1] + 1, *kernel.shape[2:]), dtype=np.int8)
        new_kernel[:kernel.shape[0], :kernel.shape[1]] = kernel
        kernel = new_kernel

    assert ((kernel.shape[1] * num_input_channels) % 16) == 0
    assert num_output_channels == 16

    # Use R11 as output pointer
    state.text += f"""
    imull ${state.height * state.width * num_input_channels}, %r8d, %r11d
    addq %rdx, %r11"""

    if ((state.height * state.width * num_input_channels) % 16) != 0:
        state.text += """
    addq $15, %r11
    andq $-16, %r11"""
    
    state.text += """
    movq %r11, %r12"""

    if layer.padding == 'same':
        state.text += """
    movq %rdx, %r10"""
        asm_pad_image(
            in_ptr='%rdx',
            out_ptr='%r12',
            num_channels=num_input_channels,
            pads=((kernel.shape[0] - 1) // 2, kernel.shape[1] // 2, kernel.shape[0] // 2, (kernel.shape[1] - 1) // 2),
            label_manager=label_manager,
            state=state
        )
        state.text += """
    movq %r11, %rdx
    movq %r10, %r11"""

    # Store biases
    state.data += f"""
    .short {', '.join(str(x) for x in np.ravel(offsets))}"""
    state.text += f"""
    add ${state.data_offset + 2 * np.size(offsets)}, %rax"""

    # Store weights
    slices = (
        (row, pixel, slice(input_channel_start, input_channel_end), output_channel)
        for row in range(kernel.shape[0])
        for pixel in range(kernel.shape[1])
        for input_channel_offset in range(0, num_input_channels, 8)
        for output_channel_offset in range(0, num_output_channels, 8)
        for input_channel_start, input_channel_end in zip(range(input_channel_offset, input_channel_offset + 8, 2), range(input_channel_offset + 2, input_channel_offset + 10, 2))
        for output_channel in range(output_channel_offset, output_channel_offset + 8)
    )
    state.data += f"""
    .byte {', '.join(str(x) for s in slices for x in np.ravel(kernel[s]))}"""

    # Load first two bias vectors into XMM14 and XMM15
    state.text += f"""
    movdqa {- (2 * np.size(offsets))}(%rax), %xmm14
    movdqa {0x10 - (2 * np.size(offsets))}(%rax), %xmm15"""
    state.data_offset = np.size(kernel)

    # Loop over samples (R10D)
    batch_loop = label_manager.get_next_label()
    state.text += f"""
    movl %r8d, %r10d
{batch_loop}:"""

    # Loop over rows (R9D)
    if state.height > kernel.shape[0]:
        row_loop = label_manager.get_next_label()
        state.text += f"""
    movl ${state.height - ((kernel.shape[0] + 1) // 2)}, %r9d
{row_loop}:"""

    # Loop over cols (ECX)
    if state.width > kernel.shape[1]:
        col_loop = label_manager.get_next_label()
        state.text += f"""
    movl ${state.width - ((kernel.shape[1] + 1) // 2)}, %ecx
{col_loop}:"""

    load_op = 'movdqa' if (num_input_channels % 16) == 0 else 'movdqu'

    load_offsets = (
        row * state.width * num_input_channels + offset_in_row
        for row in range(kernel.shape[0])
        for offset_in_row in range(0, kernel.shape[1] * num_input_channels, 0x10)
    )
    weight_offsets = range(0, np.size(kernel), 0x100)
    for load_offset, weight_offset in zip(load_offsets, weight_offsets):
        result_op = 'movdqa' if load_offset == 0 else 'paddsw'

        state.text += f"""
    {load_op} {load_offset}(%rdx), %xmm4
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
    punpckhqdq %xmm11, %xmm11"""

        state.text += f"""
    movdqa %xmm4, %xmm0
    movdqa %xmm5, %xmm1
    movdqa %xmm6, %xmm2
    movdqa %xmm7, %xmm3
    pmaddubsw {weight_offset}(%rax), %xmm0
    pmaddubsw {weight_offset + 0x10}(%rax), %xmm1
    pmaddubsw {weight_offset + 0x20}(%rax), %xmm2
    pmaddubsw {weight_offset + 0x30}(%rax), %xmm3
    pmaddubsw {weight_offset + 0x40}(%rax), %xmm4
    pmaddubsw {weight_offset + 0x50}(%rax), %xmm5
    pmaddubsw {weight_offset + 0x60}(%rax), %xmm6
    pmaddubsw {weight_offset + 0x70}(%rax), %xmm7
    paddsw %xmm1, %xmm0
    paddsw %xmm3, %xmm2
    paddsw %xmm5, %xmm4
    paddsw %xmm7, %xmm6
    paddsw %xmm2, %xmm0
    paddsw %xmm6, %xmm4
    {result_op} %xmm0, %xmm12
    {result_op} %xmm4, %xmm13"""

        state.text += f"""
    movdqa %xmm8, %xmm0
    movdqa %xmm9, %xmm1
    movdqa %xmm10, %xmm2
    movdqa %xmm11, %xmm3
    movdqa %xmm8, %xmm4
    movdqa %xmm9, %xmm5
    movdqa %xmm10, %xmm6
    movdqa %xmm11, %xmm7
    pmaddubsw {weight_offset + 0x80}(%rax), %xmm0
    pmaddubsw {weight_offset + 0x90}(%rax), %xmm1
    pmaddubsw {weight_offset + 0xA0}(%rax), %xmm2
    pmaddubsw {weight_offset + 0xB0}(%rax), %xmm3
    pmaddubsw {weight_offset + 0xC0}(%rax), %xmm4
    pmaddubsw {weight_offset + 0xD0}(%rax), %xmm5
    pmaddubsw {weight_offset + 0xE0}(%rax), %xmm6
    pmaddubsw {weight_offset + 0xF0}(%rax), %xmm7
    paddsw %xmm1, %xmm0
    paddsw %xmm3, %xmm2
    paddsw %xmm5, %xmm4
    paddsw %xmm7, %xmm6
    paddsw %xmm2, %xmm0
    paddsw %xmm6, %xmm4
    paddsw %xmm0, %xmm12
    paddsw %xmm4, %xmm13"""

    # Divide by 2**scale
    for r in range(12, 12 + (num_output_channels // 8)):
        state.text += f"""
    psraw ${scale}, %xmm{r}"""

    # Add bias
    for r, bias in zip(range(12, 12 + (num_output_channels // 8)), ('%xmm14', '%xmm15', *(f'{i - (2 * np.size(offsets))}(%rax)' for i in range(0x20, 2 * np.size(offsets), 0x10)))):
        state.text += f"""
    paddsw {bias}, %xmm{r}"""

    # Clip and pack results
    for r1, r2 in zip(range(12, 12 + (num_output_channels // 8), 2), range(13, 13 + (num_output_channels // 8), 2)):
        state.text += f"""
    packuswb %xmm{r2}, %xmm{r1}"""

    # Store results
    for offset, r in enumerate(range(12, 12 + (num_output_channels // 8), 2)):
        state.text += f"""
    movdqa %xmm{r}, {offset * 0x10}(%r11)"""

    state.text += f"""
    add ${num_output_channels}, %r11"""

    # Next column
    if state.width > kernel.shape[1]:
        state.text += f"""
    add ${num_input_channels}, %rdx
    dec %ecx
    jnz {col_loop}"""

    # Next row
    if state.height > kernel.shape[0]:
        state.text += f"""
    add ${(kernel.shape[1] - 1) * num_input_channels if state.width > kernel.shape[1] else state.width * num_input_channels}, %rdx
    dec %r9d
    jnz {row_loop}"""

    # Next sample
    if state.height == kernel.shape[0]:
        pixel_offset = state.height * state.width
        if state.width > kernel.shape[1]:
            pixel_offset -= 1
    else:
        pixel_offset = state.width
    state.text += f"""
    add ${pixel_offset * num_input_channels}, %rdx
    dec %r10d
    jnz {batch_loop}"""

    state.height -= ((kernel.shape[0] - 1) // 2 + kernel.shape[0] // 2)
    state.width -= ((kernel.shape[1] - 1) // 2 + kernel.shape[1] // 2)

    if layer.padding == 'valid':
        # Reset pointers
        state.text += """
    movq %rdi, %rdx
    movq %r12, %r11"""

        # Copy results to start of memory region
        asm_copy_memory(
            in_ptr='%r11',
            out_ptr='%rdx',
            num_bytes_per_sample=state.height * state.width * num_output_channels,
            label_manager=label_manager,
            state=state
        )

    # Reset pointers
    state.text += """
    movq %rdi, %rdx
    movq %rdi, %r11"""


def asm_conv_fullimage(layer: QuantizedConvBNRelu, label_manager: LabelManager, state: State):
    assert state.height == layer.kernel_size[0] and state.width == layer.kernel_size[1]
    assert (layer.output_shape[-1] % 16) == 0 and (layer.input_shape[-1] % 16) == 0
    assert layer.padding == 'valid'

    kernel, scale, offsets = layer.get_quantized_weights()
    kernel, scale, offsets = kernel.numpy().astype(np.int8).reshape(-1, kernel.shape[-1]), np.log2(scale).round().astype(np.uint8), offsets.numpy().astype(np.int16)
    asm_dense_x16_x16(1, kernel, scale, offsets, label_manager, state)

    state.height = 1
    state.width = 1

def asm_conv_padded_3x3_16_32_4x4(layer: QuantizedConvBNRelu, label_manager: LabelManager, state: State):
    # Generated using clang; to be optimized in the future

    kernel, scale, offsets = layer.get_quantized_weights()
    kernel, scale, offsets = kernel.numpy().astype(np.int8), np.log2(scale).round().astype(np.uint8), offsets.numpy().astype(np.int16)

    assert layer.padding == 'same'
    assert kernel.shape == (3, 3, 16, 32)
    assert state.width == 4 and state.height == 4

    # Store biases
    state.data += f"""
    .short {', '.join(str(x) for x in np.ravel(offsets))}"""

    # Store weights
    state.data += f"""
    .byte {', '.join(str(x) for x in np.ravel(kernel))}"""

    if state.data_offset > 0:
        state.text += f"""
    addq ${state.data_offset}, %rax"""

    labels = [label_manager.get_next_label(prefix='_gen_label_') for _ in range(18)]

    state.text += f"""
    pushq   %rbp
    pushq   %r15
    pushq   %r14
    pushq   %r13
    pushq   %r8
    pushq   %rdi
    pushq   %rsi
    pushq   %rbx
    pushq   %rax
    movq %rax, %rsi
    leaq    256(%rdi), %rax
    movl   %r8d, %edx
    movq    %rax, -128(%rsp)                # 8-byte Spill
    movq    %rax, %rbx
    movl    %edx, %eax
    leaq    64(%rsi), %r8
    leaq    544(%rsi), %r14
    movq    %rax, -120(%rsp)                # 8-byte Spill
    xorl    %eax, %eax
    jmp     {labels[1]}
{labels[0]}:                               #   in Loop: Header=BB0_2 Depth=1
    movq    -112(%rsp), %rax                # 8-byte Reload
    addq    $1, %rax
    cmpq    -120(%rsp), %rax                # 8-byte Folded Reload
    je      {labels[17]}
{labels[1]}:                                # =>This Loop Header: Depth=1
    movq    %rax, -112(%rsp)                # 8-byte Spill
    leaq    (,%rax,4), %rax
    xorl    %r15d, %r15d
    movq    %rax, -104(%rsp)                # 8-byte Spill
    jmp     {labels[3]}
{labels[2]}:                               #   in Loop: Header=BB0_3 Depth=2
    addq    $1, %r15
    cmpq    $4, %r15
    je      {labels[0]}
{labels[3]}:                                #   Parent Loop BB0_2 Depth=1
    movq    -104(%rsp), %rax                # 8-byte Reload
    xorl    %r13d, %r13d
    leaq    (%r15,%rax), %r12
    jmp     {labels[6]}
{labels[4]}:                               #   in Loop: Header=BB0_4 Depth=3
    movdqa  -96(%rsp), %xmm0
    movdqu  (%rsi), %xmm2
    movdqa  -80(%rsp), %xmm1
    psraw   $5, %xmm0
    psraw   $5, %xmm1
    paddsw  %xmm0, %xmm2
    movdqu  16(%rsi), %xmm0
    paddsw  %xmm1, %xmm0
    packuswb        %xmm0, %xmm2
    movdqu  %xmm2, (%rbx)
    movdqa  -64(%rsp), %xmm0
    movdqa  -48(%rsp), %xmm1
    movdqu  32(%rsi), %xmm2
    movdqu  48(%rsi), %xmm3
    psraw   $5, %xmm0
    psraw   $5, %xmm1
    paddsw  %xmm0, %xmm2
    paddsw  %xmm1, %xmm3
    packuswb        %xmm3, %xmm2
    movdqu  %xmm2, 16(%rbx)
    movq    %rax, %rbx
{labels[5]}:                               #   in Loop: Header=BB0_4 Depth=3
    addq    $1, %r13
    cmpq    $4, %r13
    je      {labels[2]}
{labels[6]}:                                #   Parent Loop BB0_2 Depth=1
    pxor    %xmm0, %xmm0
    movq    $-1, %rbp
    movq    %r14, %rcx
    movdqa  %xmm0, -48(%rsp)
    movdqa  %xmm0, -64(%rsp)
    movdqa  %xmm0, -80(%rsp)
    movdqa  %xmm0, -96(%rsp)
    jmp     {labels[8]}
{labels[7]}:                               #   in Loop: Header=BB0_5 Depth=4
    addq    $1, %rbp
    addq    $1536, %rcx                     # imm = 0x600
    cmpq    $2, %rbp
    je      {labels[12]}
{labels[8]}:                                #   Parent Loop BB0_2 Depth=1
    leal    (%rbp,%r15), %eax
    testl   $-4, %eax
    jne     {labels[7]}
    leaq    (%r12,%rbp), %rax
    movq    $-1, %r10
    movq    %rcx, %r9
    leaq    (%r13,%rax,4), %rax
    jmp     {labels[10]}
{labels[9]}:                               #   in Loop: Header=BB0_7 Depth=5
    addq    $1, %r10
    addq    $512, %r9                       # imm = 0x200
    cmpq    $2, %r10
    je      {labels[7]}
{labels[10]}:                                #   Parent Loop BB0_2 Depth=1
    leal    (%r10,%r13), %edx
    testl   $-4, %edx
    jne     {labels[9]}
    leaq    (%rax,%r10), %rdx
    shlq    $4, %rdx
    movzbl  (%rdi,%rdx), %r11d
    movd    %r11d, %xmm0
    movzbl  1(%rdi,%rdx), %r11d
    pshuflw $0, %xmm0, %xmm0                # xmm0 = xmm0[0,0,0,0,4,5,6,7]
    pshufd  $0, %xmm0, %xmm0                # xmm0 = xmm0[0,0,0,0]
    movdqa  %xmm0, -16(%rsp)                # 16-byte Spill
    movd    %r11d, %xmm0
    movzbl  2(%rdi,%rdx), %r11d
    pshuflw $0, %xmm0, %xmm0                # xmm0 = xmm0[0,0,0,0,4,5,6,7]
    pshufd  $0, %xmm0, %xmm0                # xmm0 = xmm0[0,0,0,0]
    movdqa  %xmm0, -32(%rsp)                # 16-byte Spill
    movd    %r11d, %xmm0
    movzbl  3(%rdi,%rdx), %r11d
    pshuflw $0, %xmm0, %xmm0                # xmm0 = xmm0[0,0,0,0,4,5,6,7]
    pshufd  $0, %xmm0, %xmm3                # xmm3 = xmm0[0,0,0,0]
    movd    %r11d, %xmm0
    movzbl  4(%rdi,%rdx), %r11d
    pshuflw $0, %xmm0, %xmm0                # xmm0 = xmm0[0,0,0,0,4,5,6,7]
    pshufd  $0, %xmm0, %xmm4                # xmm4 = xmm0[0,0,0,0]
    movd    %r11d, %xmm0
    movzbl  5(%rdi,%rdx), %r11d
    pshuflw $0, %xmm0, %xmm0                # xmm0 = xmm0[0,0,0,0,4,5,6,7]
    pshufd  $0, %xmm0, %xmm5                # xmm5 = xmm0[0,0,0,0]
    movd    %r11d, %xmm0
    movzbl  6(%rdi,%rdx), %r11d
    pshuflw $0, %xmm0, %xmm0                # xmm0 = xmm0[0,0,0,0,4,5,6,7]
    pshufd  $0, %xmm0, %xmm6                # xmm6 = xmm0[0,0,0,0]
    movd    %r11d, %xmm0
    movzbl  7(%rdi,%rdx), %r11d
    pshuflw $0, %xmm0, %xmm0                # xmm0 = xmm0[0,0,0,0,4,5,6,7]
    pshufd  $0, %xmm0, %xmm7                # xmm7 = xmm0[0,0,0,0]
    movd    %r11d, %xmm0
    movzbl  8(%rdi,%rdx), %r11d
    pshuflw $0, %xmm0, %xmm0                # xmm0 = xmm0[0,0,0,0,4,5,6,7]
    pshufd  $0, %xmm0, %xmm8                # xmm8 = xmm0[0,0,0,0]
    movd    %r11d, %xmm0
    movzbl  9(%rdi,%rdx), %r11d
    pshuflw $0, %xmm0, %xmm0                # xmm0 = xmm0[0,0,0,0,4,5,6,7]
    pshufd  $0, %xmm0, %xmm9                # xmm9 = xmm0[0,0,0,0]
    movd    %r11d, %xmm0
    movzbl  10(%rdi,%rdx), %r11d
    pshuflw $0, %xmm0, %xmm0                # xmm0 = xmm0[0,0,0,0,4,5,6,7]
    pshufd  $0, %xmm0, %xmm10               # xmm10 = xmm0[0,0,0,0]
    movd    %r11d, %xmm0
    movzbl  11(%rdi,%rdx), %r11d
    pshuflw $0, %xmm0, %xmm0                # xmm0 = xmm0[0,0,0,0,4,5,6,7]
    pshufd  $0, %xmm0, %xmm11               # xmm11 = xmm0[0,0,0,0]
    movd    %r11d, %xmm0
    movzbl  12(%rdi,%rdx), %r11d
    pshuflw $0, %xmm0, %xmm0                # xmm0 = xmm0[0,0,0,0,4,5,6,7]
    pshufd  $0, %xmm0, %xmm12               # xmm12 = xmm0[0,0,0,0]
    movd    %r11d, %xmm0
    movzbl  13(%rdi,%rdx), %r11d
    pshuflw $0, %xmm0, %xmm0                # xmm0 = xmm0[0,0,0,0,4,5,6,7]
    pshufd  $0, %xmm0, %xmm13               # xmm13 = xmm0[0,0,0,0]
    movd    %r11d, %xmm0
    movzbl  14(%rdi,%rdx), %r11d
    movzbl  15(%rdi,%rdx), %edx
    pshuflw $0, %xmm0, %xmm0                # xmm0 = xmm0[0,0,0,0,4,5,6,7]
    pshufd  $0, %xmm0, %xmm14               # xmm14 = xmm0[0,0,0,0]
    movd    %r11d, %xmm0
    pshuflw $0, %xmm0, %xmm0                # xmm0 = xmm0[0,0,0,0,4,5,6,7]
    pshufd  $0, %xmm0, %xmm15               # xmm15 = xmm0[0,0,0,0]
    movd    %edx, %xmm0
    movq    $-32, %rdx
    pshuflw $0, %xmm0, %xmm0                # xmm0 = xmm0[0,0,0,0,4,5,6,7]
    pshufd  $0, %xmm0, %xmm0                # xmm0 = xmm0[0,0,0,0]
{labels[11]}:                                #   Parent Loop BB0_2 Depth=1
    pmovsxbw        -448(%r9,%rdx), %xmm1
    pmovsxbw        -416(%r9,%rdx), %xmm2
    pmullw  -16(%rsp), %xmm1                # 16-byte Folded Reload
    pmullw  -32(%rsp), %xmm2                # 16-byte Folded Reload
    paddsw  -32(%rsp,%rdx,2), %xmm1
    paddsw  %xmm1, %xmm2
    pmovsxbw        -384(%r9,%rdx), %xmm1
    pmullw  %xmm3, %xmm1
    paddsw  %xmm2, %xmm1
    pmovsxbw        -352(%r9,%rdx), %xmm2
    pmullw  %xmm4, %xmm2
    paddsw  %xmm1, %xmm2
    pmovsxbw        -320(%r9,%rdx), %xmm1
    pmullw  %xmm5, %xmm1
    paddsw  %xmm2, %xmm1
    pmovsxbw        -288(%r9,%rdx), %xmm2
    pmullw  %xmm6, %xmm2
    paddsw  %xmm1, %xmm2
    pmovsxbw        -256(%r9,%rdx), %xmm1
    pmullw  %xmm7, %xmm1
    paddsw  %xmm2, %xmm1
    pmovsxbw        -224(%r9,%rdx), %xmm2
    pmullw  %xmm8, %xmm2
    paddsw  %xmm1, %xmm2
    pmovsxbw        -192(%r9,%rdx), %xmm1
    pmullw  %xmm9, %xmm1
    paddsw  %xmm2, %xmm1
    pmovsxbw        -160(%r9,%rdx), %xmm2
    pmullw  %xmm10, %xmm2
    paddsw  %xmm1, %xmm2
    pmovsxbw        -128(%r9,%rdx), %xmm1
    pmullw  %xmm11, %xmm1
    paddsw  %xmm2, %xmm1
    pmovsxbw        -96(%r9,%rdx), %xmm2
    pmullw  %xmm12, %xmm2
    paddsw  %xmm1, %xmm2
    pmovsxbw        -64(%r9,%rdx), %xmm1
    pmullw  %xmm13, %xmm1
    paddsw  %xmm2, %xmm1
    pmovsxbw        -32(%r9,%rdx), %xmm2
    pmullw  %xmm14, %xmm2
    paddsw  %xmm1, %xmm2
    pmovsxbw        (%r9,%rdx), %xmm1
    pmullw  %xmm15, %xmm1
    paddsw  %xmm2, %xmm1
    pmovsxbw        32(%r9,%rdx), %xmm2
    pmullw  %xmm0, %xmm2
    paddsw  %xmm1, %xmm2
    movdqa  %xmm2, -32(%rsp,%rdx,2)
    addq    $8, %rdx
    jne     {labels[11]}
    jmp     {labels[9]}
{labels[12]}:                               #   in Loop: Header=BB0_4 Depth=3
    leaq    -32(%rsp), %rcx
    leaq    32(%rbx), %rax
    cmpq    %rcx, %rbx
    leaq    -96(%rsp), %rcx
    setb    %r9b
    cmpq    %rax, %rcx
    setb    %r10b
    cmpq    %r8, %rbx
    setb    %cl
    cmpq    %rsi, %rax
    seta    %dl
    testb   %r10b, %r9b
    jne     {labels[13]}
    andb    %dl, %cl
    je      {labels[4]}
{labels[13]}:                               #   in Loop: Header=BB0_4 Depth=3
    movq    $-64, %rax
    jmp     {labels[15]}
{labels[14]}:                               #   in Loop: Header=BB0_15 Depth=4
    movb    %cl, (%rbx)
    addq    $1, %rbx
    addq    $2, %rax
    je      {labels[5]}
{labels[15]}:                               #   Parent Loop BB0_2 Depth=1
    movswl  -32(%rsp,%rax), %ecx
    movzwl  64(%rsi,%rax), %edx
    shrl    $5, %ecx
    leal    (%rcx,%rdx), %r9d
    movswl  %r9w, %r9d
    sarl    $15, %r9d
    xorl    $-32768, %r9d                   # imm = 0x8000
    addw    %dx, %cx
    cmovol  %r9d, %ecx
    testw   %cx, %cx
    jg      {labels[16]}
    xorl    %ecx, %ecx
{labels[16]}:                               #   in Loop: Header=BB0_15 Depth=4
    movswl  %cx, %edx
    cmpl    $255, %edx
    jl      {labels[14]}
    movl    $255, %ecx
    jmp     {labels[14]}
{labels[17]}:
    movq    -128(%rsp), %r11                # 8-byte Reload
    popq    %rax
    popq    %rbx
    popq    %rsi
    popq    %rdi
    popq    %r8
    popq    %r13
    popq    %r14
    popq    %r15
    popq    %rbp
    mov %rdi, %rdx"""

    # Copy results to start of memory region
    asm_copy_memory(
        in_ptr='%r11',
        out_ptr='%rdx',
        num_bytes_per_sample=state.height * state.width * kernel.shape[-1],
        label_manager=label_manager,
        state=state
    )

    # Reset pointers
    state.text += """
    movq %rdi, %rdx
    movq %rdi, %r11"""

    state.data_offset = np.size(kernel) + 2 * np.size(offsets)
