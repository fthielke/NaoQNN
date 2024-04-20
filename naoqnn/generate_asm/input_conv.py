# SPDX-License-Identifier: MIT
import numpy as np
from ..layers import QuantizedConvBNRelu
from .state import LabelManager, State


def asm_input_conx3x3_8(layer: QuantizedConvBNRelu, label_manager: LabelManager, state: State):
    kernel, scale, offsets = layer.get_quantized_weights()
    kernel, scale, offsets = kernel.numpy().astype(np.int8), np.log2(scale).round().astype(np.uint8), offsets.numpy().astype(np.int16)

    assert state.width == 32

    # Store biases
    state.data += f"""
    .short {', '.join(str(x) for x in np.ravel(offsets))}"""

    # Store shuffling constants
    state.data += f"""
    .byte {', '.join(f'{i}, {i+1 if i < 15 else 255}' for i in range(16) for _ in range(8))}"""
    num_constants = 16 * 0x10

    # Store weights
    state.data += f"""
    .byte {', '.join(str(x) for x in np.ravel(np.transpose(kernel[:,:2,0], axes=(0,2,1))))}
    .byte {', '.join(str(x) for x in np.ravel(kernel[:2,2], order='F'))}
    .byte {', 0, '.join(str(x) for x in np.ravel(kernel[2,2]))}, 0"""

    # Load biases into XMM15
    state.text += f"""
    movdqa {state.data_offset}(%rax), %xmm15
    add ${state.data_offset + 0x10 + num_constants}, %rax"""
    state.data_offset = 0x50

    # Make r11 the output_ptr
    state.text += """
    movq %rdx, %r11"""

    # Loop over samples (R9D)
    batch_loop = label_manager.get_next_label()
    state.text += f"""
    movl %r8d, %r9d
{batch_loop}:"""

    # Load input pixels (3 rows, 32 pixels each)
    state.text += """
    movdqa (%rdi), %xmm8
    movdqa 0x10(%rdi), %xmm9
    movdqa 0x20(%rdi), %xmm10
    movdqa 0x30(%rdi), %xmm11
    movdqa 0x40(%rdi), %xmm12
    movdqa 0x50(%rdi), %xmm13
    addq $0x60, %rdi"""

    # Loop over rows (ECX)
    row_loop = label_manager.get_next_label()
    state.text += f"""
    movl ${state.height - 2}, %ecx
{row_loop}:"""

    # XMM8:  X: 0123456789ABCDEF
    # XMM9:  X: GHIJKLMNOPQRSTUV
    # XMM10: Y: 0123456789ABCDEF
    # XMM11: Y: GHIJKLMNOPQRSTUV
    # XMM12: Z: 0123456789ABCDEF
    # XMM13: Z: GHIJKLMNOPQRSTUV

    def calc_two_pixels(dest_offset: int, load_x_y_z_xy: str, load_z: str, shuffle_offset_0_x_y_z: int, shuffle_offset_1_x_y_z: int, shuffle_offset_0_xy: int, shuffle_offset_1_xy: int, shuffle_offset_0_z: int, shuffle_offset_1_z: int):
        return f"""
    {load_x_y_z_xy.strip()}
    pshufb {shuffle_offset_0_x_y_z - num_constants}(%rax), %xmm0
    pshufb {shuffle_offset_0_x_y_z - num_constants}(%rax), %xmm1
    pshufb {shuffle_offset_0_x_y_z - num_constants}(%rax), %xmm2
    pshufb {shuffle_offset_1_x_y_z - num_constants}(%rax), %xmm4
    pshufb {shuffle_offset_1_x_y_z - num_constants}(%rax), %xmm5
    pshufb {shuffle_offset_1_x_y_z - num_constants}(%rax), %xmm6
    pshufb {shuffle_offset_0_xy - num_constants}(%rax), %xmm3
    pshufb {shuffle_offset_1_xy - num_constants}(%rax), %xmm7
    pmaddubsw (%rax), %xmm0
    pmaddubsw (%rax), %xmm4
    pmaddubsw 0x10(%rax), %xmm1
    pmaddubsw 0x10(%rax), %xmm5
    pmaddubsw 0x20(%rax), %xmm2
    pmaddubsw 0x20(%rax), %xmm6
    pmaddubsw 0x30(%rax), %xmm3
    pmaddubsw 0x30(%rax), %xmm7
    paddsw %xmm1, %xmm0
    paddsw %xmm3, %xmm2
    paddsw %xmm5, %xmm4
    paddsw %xmm7, %xmm6
    paddsw %xmm2, %xmm0
    paddsw %xmm6, %xmm4
    {load_z.strip()}
    pshufb {shuffle_offset_0_z - num_constants}(%rax), %xmm1
    pshufb {shuffle_offset_1_z - num_constants}(%rax), %xmm2
    pmaddubsw 0x40(%rax), %xmm1
    pmaddubsw 0x40(%rax), %xmm2
    paddsw %xmm1, %xmm0
    paddsw %xmm2, %xmm4
    psraw ${scale}, %xmm0
    psraw ${scale}, %xmm4
    paddsw %xmm15, %xmm0
    paddsw %xmm15, %xmm4
    packuswb %xmm4, %xmm0
    movdqa %xmm0, {dest_offset}(%r11)"""

    state.text += """
    movdqa %xmm8, %xmm14
    punpcklbw %xmm10, %xmm14"""    # XMM14: X0 Y0 X1 Y1 X2 Y2 X3 Y3 X4 Y4 X5 Y5 X6 Y6 X7 Y7

    load_x_y_z_xy = """
    movdqa %xmm8, %xmm0
    movdqa %xmm10, %xmm1
    movdqa %xmm12, %xmm2
    movdqa %xmm14, %xmm3
    movdqa %xmm8, %xmm4
    movdqa %xmm10, %xmm5
    movdqa %xmm12, %xmm6
    movdqa %xmm14, %xmm7"""
    load_z = """
    movdqa %xmm12, %xmm1
    movdqa %xmm12, %xmm2"""

    state.text += calc_two_pixels( # 0 and 1
        dest_offset=0,
        load_x_y_z_xy=load_x_y_z_xy,
        load_z=load_z,
        shuffle_offset_0_x_y_z=0,
        shuffle_offset_1_x_y_z=0x10,
        shuffle_offset_0_xy=0x40,
        shuffle_offset_1_xy=0x60,
        shuffle_offset_0_z=0x20,
        shuffle_offset_1_z=0x30
    )
    state.text += calc_two_pixels( # 2 and 3
        dest_offset=0x10,
        load_x_y_z_xy=load_x_y_z_xy,
        load_z=load_z,
        shuffle_offset_0_x_y_z=0x20,
        shuffle_offset_1_x_y_z=0x30,
        shuffle_offset_0_xy=0x80,
        shuffle_offset_1_xy=0xA0,
        shuffle_offset_0_z=0x40,
        shuffle_offset_1_z=0x50
    )
    state.text += calc_two_pixels( # 4 and 5
        dest_offset=0x20,
        load_x_y_z_xy=load_x_y_z_xy,
        load_z=load_z,
        shuffle_offset_0_x_y_z=0x40,
        shuffle_offset_1_x_y_z=0x50,
        shuffle_offset_0_xy=0xC0,
        shuffle_offset_1_xy=0xE0,
        shuffle_offset_0_z=0x60,
        shuffle_offset_1_z=0x70
    )

    state.text += """
    movdqa %xmm8, %xmm14
    punpckhbw %xmm10, %xmm14"""    # XMM14: X8 Y8 X9 Y9 XA YA XB YB XC YC XD YD XE YE XF YF

    state.text += calc_two_pixels( # 6 and 7
        dest_offset=0x30,
        load_x_y_z_xy=load_x_y_z_xy,
        load_z=load_z,
        shuffle_offset_0_x_y_z=0x60,
        shuffle_offset_1_x_y_z=0x70,
        shuffle_offset_0_xy=0,
        shuffle_offset_1_xy=0x20,
        shuffle_offset_0_z=0x80,
        shuffle_offset_1_z=0x90
    )
    state.text += calc_two_pixels( # 8 and 9
        dest_offset=0x40,
        load_x_y_z_xy=load_x_y_z_xy,
        load_z=load_z,
        shuffle_offset_0_x_y_z=0x80,
        shuffle_offset_1_x_y_z=0x90,
        shuffle_offset_0_xy=0x40,
        shuffle_offset_1_xy=0x60,
        shuffle_offset_0_z=0xA0,
        shuffle_offset_1_z=0xB0
    )
    state.text += calc_two_pixels( # 10 and 11
        dest_offset=0x50,
        load_x_y_z_xy=load_x_y_z_xy,
        load_z=load_z,
        shuffle_offset_0_x_y_z=0xA0,
        shuffle_offset_1_x_y_z=0xB0,
        shuffle_offset_0_xy=0x80,
        shuffle_offset_1_xy=0xA0,
        shuffle_offset_0_z=0xC0,
        shuffle_offset_1_z=0xD0
    )
    state.text += calc_two_pixels( # 12 and 13
        dest_offset=0x60,
        load_x_y_z_xy=load_x_y_z_xy,
        load_z=load_z,
        shuffle_offset_0_x_y_z=0xC0,
        shuffle_offset_1_x_y_z=0xD0,
        shuffle_offset_0_xy=0xC0,
        shuffle_offset_1_xy=0xE0,
        shuffle_offset_0_z=0xE0,
        shuffle_offset_1_z=0xF0
    )

    load_z = """
    movdqa %xmm13, %xmm1
    movdqa %xmm13, %xmm2"""

    # XMM0/4:  X: EFGHIJKLMNOPQRST
    # XMM1/5:  Y: EFGHIJKLMNOPQRST
    # XMM2/6:  Z: EFGHIJKLMNOPQRST
    # XMM3/7:  XE YE XF YF XG YG XH YH XI YI XJ YJ XK YK XL YL
    state.text += calc_two_pixels( # 14 and 15
        dest_offset=0x70,
        load_x_y_z_xy="""
    movdqa %xmm8, %xmm4
    movdqa %xmm9, %xmm0
    movdqa %xmm10, %xmm5
    movdqa %xmm11, %xmm1
    movdqa %xmm12, %xmm6
    movdqa %xmm13, %xmm2
    palignr $14, %xmm4, %xmm0
    palignr $14, %xmm5, %xmm1
    palignr $14, %xmm6, %xmm2
    movdqa %xmm0, %xmm4
    movdqa %xmm1, %xmm5
    movdqa %xmm0, %xmm3
    movdqa %xmm2, %xmm6
    punpcklbw %xmm1, %xmm3
    movdqa %xmm3, %xmm7""",
        load_z=load_z,
        shuffle_offset_0_x_y_z=0,
        shuffle_offset_1_x_y_z=0x10,
        shuffle_offset_0_xy=0x40,
        shuffle_offset_1_xy=0x60,
        shuffle_offset_0_z=0,
        shuffle_offset_1_z=0x10
    )

    state.text += """
    movdqa %xmm9, %xmm14
    punpcklbw %xmm11, %xmm14"""    # XMM14: XG YG XH YH XI YI XJ YJ XK YK XL YL XM YM XN YN
    load_x_y_z_xy = """
    movdqa %xmm9, %xmm0
    movdqa %xmm11, %xmm1
    movdqa %xmm13, %xmm2
    movdqa %xmm14, %xmm3
    movdqa %xmm9, %xmm4
    movdqa %xmm11, %xmm5
    movdqa %xmm13, %xmm6
    movdqa %xmm14, %xmm7"""

    state.text += calc_two_pixels( # 16 and 17
        dest_offset=0x80,
        load_x_y_z_xy=load_x_y_z_xy,
        load_z=load_z,
        shuffle_offset_0_x_y_z=0,
        shuffle_offset_1_x_y_z=0x10,
        shuffle_offset_0_xy=0x40,
        shuffle_offset_1_xy=0x60,
        shuffle_offset_0_z=0x20,
        shuffle_offset_1_z=0x30
    )
    state.text += calc_two_pixels( # 18 and 19
        dest_offset=0x90,
        load_x_y_z_xy=load_x_y_z_xy,
        load_z=load_z,
        shuffle_offset_0_x_y_z=0x20,
        shuffle_offset_1_x_y_z=0x30,
        shuffle_offset_0_xy=0x80,
        shuffle_offset_1_xy=0xA0,
        shuffle_offset_0_z=0x40,
        shuffle_offset_1_z=0x50
    )
    state.text += calc_two_pixels( # 20 and 21
        dest_offset=0xA0,
        load_x_y_z_xy=load_x_y_z_xy,
        load_z=load_z,
        shuffle_offset_0_x_y_z=0x40,
        shuffle_offset_1_x_y_z=0x50,
        shuffle_offset_0_xy=0xC0,
        shuffle_offset_1_xy=0xE0,
        shuffle_offset_0_z=0x60,
        shuffle_offset_1_z=0x70
    )
    
    state.text += """
    movdqa %xmm9, %xmm14
    punpckhbw %xmm11, %xmm14"""    # XMM14: XO YO XP YP XQ YQ XR YR XS YS XT YT XU YU XV YV

    state.text += calc_two_pixels( # 22 and 23
        dest_offset=0xB0,
        load_x_y_z_xy=load_x_y_z_xy,
        load_z=load_z,
        shuffle_offset_0_x_y_z=0x60,
        shuffle_offset_1_x_y_z=0x70,
        shuffle_offset_0_xy=0,
        shuffle_offset_1_xy=0x20,
        shuffle_offset_0_z=0x80,
        shuffle_offset_1_z=0x90
    )
    state.text += calc_two_pixels( # 24 and 25
        dest_offset=0xC0,
        load_x_y_z_xy=load_x_y_z_xy,
        load_z=load_z,
        shuffle_offset_0_x_y_z=0x80,
        shuffle_offset_1_x_y_z=0x90,
        shuffle_offset_0_xy=0x40,
        shuffle_offset_1_xy=0x60,
        shuffle_offset_0_z=0xA0,
        shuffle_offset_1_z=0xB0
    )
    state.text += calc_two_pixels( # 26 and 27
        dest_offset=0xD0,
        load_x_y_z_xy=load_x_y_z_xy,
        load_z=load_z,
        shuffle_offset_0_x_y_z=0xA0,
        shuffle_offset_1_x_y_z=0xB0,
        shuffle_offset_0_xy=0x80,
        shuffle_offset_1_xy=0xA0,
        shuffle_offset_0_z=0xC0,
        shuffle_offset_1_z=0xD0
    )
    state.text += calc_two_pixels( # 28 and 29
        dest_offset=0xE0,
        load_x_y_z_xy=load_x_y_z_xy,
        load_z=load_z,
        shuffle_offset_0_x_y_z=0xC0,
        shuffle_offset_1_x_y_z=0xD0,
        shuffle_offset_0_xy=0xC0,
        shuffle_offset_1_xy=0xE0,
        shuffle_offset_0_z=0xE0,
        shuffle_offset_1_z=0xF0
    )

    # Next row
    after_row_loop = label_manager.get_next_label()
    state.text += f"""
    addq $0xF0, %r11
    dec %ecx
    jz {after_row_loop}
    movdqa %xmm10, %xmm8
    movdqa %xmm11, %xmm9
    movdqa %xmm12, %xmm10
    movdqa %xmm13, %xmm11
    movdqa (%rdi), %xmm12
    movdqa 0x10(%rdi), %xmm13
    addq $0x20, %rdi
    jmp {row_loop}
{after_row_loop}:"""

    # Next sample
    state.text += f"""
    dec %r9d
    jnz {batch_loop}"""

    # Reset pointers
    state.text += """
    movq %rdx, %rdi
    movq %rdx, %r11"""

    state.width -= 2
    state.height -= 2


def asm_input_conx3x3_8_padded(layer: QuantizedConvBNRelu, label_manager: LabelManager, state: State):
    # Generated using clang; to be optimized in the future

    kernel, scale, offsets = layer.get_quantized_weights()
    kernel, scale, offsets = kernel.numpy().astype(np.int8), np.log2(scale).round().astype(np.uint8), offsets.numpy().astype(np.int16)

    assert layer.padding == 'same'
    assert state.width == 32 and state.height == 32

    # Store weights
    state.data += f"""
    .byte {', '.join(str(x) for x in np.ravel(kernel))}
    .byte {', '.join('0' for _ in range(8))}""" # Padding to end on 16-byte address

    # Store biases
    state.data += f"""
    .short {', '.join(str(x) for x in np.ravel(offsets))}"""

    # Input in RDI
    # Output in RDX

    label_1 = label_manager.get_next_label(prefix='_gen_label_')
    label_2 = label_manager.get_next_label(prefix='_gen_label_')
    label_3 = label_manager.get_next_label(prefix='_gen_label_')
    label_4 = label_manager.get_next_label(prefix='_gen_label_')
    label_5 = label_manager.get_next_label(prefix='_gen_label_')
    label_6 = label_manager.get_next_label(prefix='_gen_label_')
    label_7 = label_manager.get_next_label(prefix='_gen_label_')
    label_8 = label_manager.get_next_label(prefix='_gen_label_')
    label_9 = label_manager.get_next_label(prefix='_gen_label_')
    label_10 = label_manager.get_next_label(prefix='_gen_label_')
    label_11 = label_manager.get_next_label(prefix='_gen_label_')
    label_12 = label_manager.get_next_label(prefix='_gen_label_')
    label_13 = label_manager.get_next_label(prefix='_gen_label_')
    label_14 = label_manager.get_next_label(prefix='_gen_label_')
    label_15 = label_manager.get_next_label(prefix='_gen_label_')
    label_16 = label_manager.get_next_label(prefix='_gen_label_')
    label_17 = label_manager.get_next_label(prefix='_gen_label_')
    label_18 = label_manager.get_next_label(prefix='_gen_label_')
    label_19 = label_manager.get_next_label(prefix='_gen_label_')
    label_20 = label_manager.get_next_label(prefix='_gen_label_')
    label_21 = label_manager.get_next_label(prefix='_gen_label_')
    label_22 = label_manager.get_next_label(prefix='_gen_label_')

    state.text += f"""
    pushq   %rbp
    pushq   %rsi
    pushq   %r15
    pushq   %r14
    pushq   %rdx
    pushq   %rbx
    pushq   %rax
    pushq   %r8
    movq %rdx, %rsi
    movl    $-1024, %r8d
    xorl    %r9d, %r9d
    movq    %rcx, -16(%rsp)
    pmovsxbw        {state.data_offset}(%rax), %xmm0
    pmovsxbw        {state.data_offset + 8}(%rax), %xmm1
    pmovsxbw        {state.data_offset + 16}(%rax), %xmm2
    pmovsxbw        {state.data_offset + 24}(%rax), %xmm3
    pmovsxbw        {state.data_offset + 32}(%rax), %xmm4
    pmovsxbw        {state.data_offset + 40}(%rax), %xmm5
    pmovsxbw        {state.data_offset + 48}(%rax), %xmm6
    pmovsxbw        {state.data_offset + 56}(%rax), %xmm7
    pmovsxbw        {state.data_offset + 64}(%rax), %xmm8
    movdqa  {state.data_offset + 80}(%rax), %xmm9
    movl    $4294967295, %eax
    leaq    1(%rdi), %rcx
    movq    %rcx, -24(%rsp)
    xorl    %ecx, %ecx
    jmp     {label_2}
{label_1}:
    movq    -8(%rsp), %rcx                  # 8-byte Reload
    subq    %r13, %rbx
    addq    $1, %r9
    addl    $1024, %r8d                     # imm = 0x400
    movq    %rbx, %rsi
    addl    $1024, %ecx                     # imm = 0x400
    cmpq    -16(%rsp), %r9                  # 8-byte Folded Reload
    je      {label_22}
{label_2}:                                # =>This Loop Header: Depth=1
    movslq  %ecx, %r11
    addq    -24(%rsp), %r11                 # 8-byte Folded Reload
    movl    %r8d, %r14d
    movl    %ecx, %r15d
    xorl    %r12d, %r12d
    addq    $992, %r14                      # imm = 0x3E0
    movq    %r15, -8(%rsp)                  # 8-byte Spill
    jmp     {label_4}
{label_3}:                               #   in Loop: Header=BB0_3 Depth=2
    movq    %rbx, %rsi
    addq    $32, %r15
    addq    $32, %r11
    addq    $32, %r14
    subq    %r13, %rsi
    cmpq    $32, %r12
    je      {label_1}
{label_4}:                                #   Parent Loop BB0_2 Depth=1
    movq    %rsi, %rbx
    movq    %r12, %rsi
    leaq    -1(%r12), %rbp
    addq    $1, %r12
    xorl    %r13d, %r13d
    xorl    %ecx, %ecx
    jmp     {label_6}
{label_5}:                               #   in Loop: Header=BB0_4 Depth=3
    psraw   ${scale}, %xmm10
    addq    $-8, %r13
    paddsw  %xmm9, %xmm10
    packuswb        %xmm10, %xmm10
    movq    %xmm10, (%rbx,%rcx,8)
    addq    $1, %rcx
    cmpq    $32, %rcx
    je      {label_3}
{label_6}:                                #   Parent Loop BB0_2 Depth=1
    pxor    %xmm10, %xmm10
    testl   $-32, %ebp
    jne     {label_8}
    leal    (%r14,%rcx), %edx
    leal    (%rax,%rcx), %r10d
    movslq  %edx, %rdx
    testl   $-32, %r10d
    je      {label_11}
    testl   $-32, %ecx
    je      {label_12}
{label_7}:                                #   in Loop: Header=BB0_4 Depth=3
    leal    1(%rcx), %r10d
    testl   $-32, %r10d
    je      {label_13}
{label_8}:                               #   in Loop: Header=BB0_4 Depth=3
    testl   $-32, %esi
    jne     {label_17}
{label_9}:                               #   in Loop: Header=BB0_4 Depth=3
    leal    (%rcx,%rax), %edx
    testl   $-32, %edx
    je      {label_14}
    testl   $-32, %ecx
    je      {label_15}
{label_10}:                               #   in Loop: Header=BB0_4 Depth=3
    leal    1(%rcx), %edx
    testl   $-32, %edx
    jne     {label_17}
    jmp     {label_16}
{label_11}:                                #   in Loop: Header=BB0_4 Depth=3
    movzbl  -1(%rdi,%rdx), %r10d
    movd    %r10d, %xmm10
    pshuflw $0, %xmm10, %xmm10              # xmm10 = xmm10[0,0,0,0,4,5,6,7]
    pshufd  $0, %xmm10, %xmm10              # xmm10 = xmm10[0,0,0,0]
    pmullw  %xmm0, %xmm10
    testl   $-32, %ecx
    jne     {label_7}
{label_12}:                                #   in Loop: Header=BB0_4 Depth=3
    movzbl  (%rdi,%rdx), %r10d
    movd    %r10d, %xmm11
    pshuflw $0, %xmm11, %xmm11              # xmm11 = xmm11[0,0,0,0,4,5,6,7]
    pshufd  $0, %xmm11, %xmm11              # xmm11 = xmm11[0,0,0,0]
    pmullw  %xmm1, %xmm11
    paddsw  %xmm11, %xmm10
    leal    1(%rcx), %r10d
    testl   $-32, %r10d
    jne     {label_8}
{label_13}:                               #   in Loop: Header=BB0_4 Depth=3
    movzbl  1(%rdi,%rdx), %edx
    movd    %edx, %xmm11
    pshuflw $0, %xmm11, %xmm11              # xmm11 = xmm11[0,0,0,0,4,5,6,7]
    pshufd  $0, %xmm11, %xmm11              # xmm11 = xmm11[0,0,0,0]
    pmullw  %xmm2, %xmm11
    paddsw  %xmm11, %xmm10
    testl   $-32, %esi
    jne     {label_17}
    jmp     {label_9}
{label_14}:                               #   in Loop: Header=BB0_4 Depth=3
    movzbl  -2(%r11,%rcx), %edx
    movd    %edx, %xmm11
    pshuflw $0, %xmm11, %xmm11              # xmm11 = xmm11[0,0,0,0,4,5,6,7]
    pshufd  $0, %xmm11, %xmm11              # xmm11 = xmm11[0,0,0,0]
    pmullw  %xmm3, %xmm11
    paddsw  %xmm11, %xmm10
    testl   $-32, %ecx
    jne     {label_10}
{label_15}:                               #   in Loop: Header=BB0_4 Depth=3
    movzbl  -1(%r11,%rcx), %edx
    movd    %edx, %xmm11
    pshuflw $0, %xmm11, %xmm11              # xmm11 = xmm11[0,0,0,0,4,5,6,7]
    pshufd  $0, %xmm11, %xmm11              # xmm11 = xmm11[0,0,0,0]
    pmullw  %xmm4, %xmm11
    paddsw  %xmm11, %xmm10
    leal    1(%rcx), %edx
    testl   $-32, %edx
    jne     {label_17}
{label_16}:                               #   in Loop: Header=BB0_4 Depth=3
    movzbl  (%r11,%rcx), %edx
    movd    %edx, %xmm11
    pshuflw $0, %xmm11, %xmm11              # xmm11 = xmm11[0,0,0,0,4,5,6,7]
    pshufd  $0, %xmm11, %xmm11              # xmm11 = xmm11[0,0,0,0]
    pmullw  %xmm5, %xmm11
    paddsw  %xmm11, %xmm10
{label_17}:                               #   in Loop: Header=BB0_4 Depth=3
    testl   $-32, %r12d
    jne     {label_5}
    leal    32(%r15,%rcx), %edx
    leal    (%rax,%rcx), %r10d
    movslq  %edx, %rdx
    testl   $-32, %r10d
    je      {label_19}
    testl   $-32, %ecx
    je      {label_20}
{label_18}:                               #   in Loop: Header=BB0_4 Depth=3
    leal    1(%rcx), %r10d
    testl   $-32, %r10d
    jne     {label_5}
    jmp     {label_21}
{label_19}:                               #   in Loop: Header=BB0_4 Depth=3
    movzbl  -1(%rdi,%rdx), %r10d
    movd    %r10d, %xmm11
    pshuflw $0, %xmm11, %xmm11              # xmm11 = xmm11[0,0,0,0,4,5,6,7]
    pshufd  $0, %xmm11, %xmm11              # xmm11 = xmm11[0,0,0,0]
    pmullw  %xmm6, %xmm11
    paddsw  %xmm11, %xmm10
    testl   $-32, %ecx
    jne     {label_18}
{label_20}:                               #   in Loop: Header=BB0_4 Depth=3
    movzbl  (%rdi,%rdx), %r10d
    movd    %r10d, %xmm11
    pshuflw $0, %xmm11, %xmm11              # xmm11 = xmm11[0,0,0,0,4,5,6,7]
    pshufd  $0, %xmm11, %xmm11              # xmm11 = xmm11[0,0,0,0]
    pmullw  %xmm7, %xmm11
    paddsw  %xmm11, %xmm10
    leal    1(%rcx), %r10d
    testl   $-32, %r10d
    jne     {label_5}
{label_21}:                               #   in Loop: Header=BB0_4 Depth=3
    movzbl  1(%rdi,%rdx), %edx
    movd    %edx, %xmm11
    pshuflw $0, %xmm11, %xmm11              # xmm11 = xmm11[0,0,0,0,4,5,6,7]
    pshufd  $0, %xmm11, %xmm11              # xmm11 = xmm11[0,0,0,0]
    pmullw  %xmm8, %xmm11
    paddsw  %xmm11, %xmm10
    jmp     {label_5}
{label_22}:
    popq    %r8
    popq    %rax
    popq    %rbx
    popq    %rdx
    popq    %r14
    popq    %r15
    popq    %rsi
    popq    %rbp
    movq %rdx, %rdi
    movq %rdx, %r11"""
    
    state.data_offset += 96
