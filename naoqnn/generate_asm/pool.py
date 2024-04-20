# SPDX-License-Identifier: MIT
from enum import Enum
import numpy as np
from .state import LabelManager, State


class PoolingMode(Enum):
    MAX = 'max'
    AVG = 'avg'


def asm_pool2x2(label_manager: LabelManager, state: State, mode: PoolingMode, channels: int):
    assert state.width % 2 == 0
    assert state.height % 2 == 0

    # Loop over all rows of all samples (R9D)
    if (2 ** np.log2(state.height // 2).round().astype(int)) == (state.height // 2):
        state.text += """
    movl %r8d, %r9d"""
        if (state.height // 2) > 1:
            state.text += f"""
    sall ${int(np.log2(state.height // 2).round().astype(int))}, %r9d"""
    else:
        state.text += f"""
    imull ${state.height // 2}, %r8d, %r9d"""
    row_loop = label_manager.get_next_label()
    state.text += f"""
{row_loop}:"""

    if channels == 16:
        kernels_per_step = next(i for i in range(8, 0, -1) if (state.width % (i * 2)) == 0)
        num_steps = state.width // (kernels_per_step * 2)
        input_stepsize = kernels_per_step * 32
    elif channels == 8:
        try:
            kernels_per_step = next(i for i in range(10, 0, -2) if (state.width % (i * 2)) == 0)
        except StopIteration:
            kernels_per_step = next(i for i in range(9, 0, -2) if (state.width % (i * 2)) == 0)
        num_steps = state.width // (kernels_per_step * 2)
        input_stepsize = kernels_per_step * 16
    elif channels == 32:
        kernels_per_step = next(i for i in range(4, 0, -1) if (state.width % (i * 2)) == 0)
        num_steps = state.width // (kernels_per_step * 2)
        input_stepsize = kernels_per_step * 64
    else:
        raise NotImplementedError()
    output_stepsize = input_stepsize // 2
    output_registers = ['xmm0']

    # Loop over cols (ECX)
    if num_steps > 1:
        col_loop = label_manager.get_next_label()
        state.text += f"""
    movl ${num_steps}, %ecx
{col_loop}:"""

    # Pooling kernel
    pooling_op = 'pmaxub' if mode == PoolingMode.MAX else 'pavgb'
    if channels == 16:
        def register_pairs():
            return zip(range(0, kernels_per_step * 2, 2), range(1, kernels_per_step * 2 + 1, 2))

        for offset, (r1, r2) in enumerate(register_pairs()):
            state.text += f"""
    movdqa {offset * 0x20}(%rdx), %xmm{r1}
    movdqa {offset * 0x20 + 0x10}(%rdx), %xmm{r2}"""

        for offset, (r1, r2) in enumerate(register_pairs()):
            state.text += f"""
    {pooling_op} {state.width * 0x10 + offset * 0x20}(%rdx), %xmm{r1}
    {pooling_op} {state.width * 0x10 + offset * 0x20 + 0x10}(%rdx), %xmm{r2}"""

        for r1, r2 in register_pairs():
            state.text += f"""
    {pooling_op} %xmm{r2}, %xmm{r1}"""

        output_registers = [f'xmm{r}' for r, _ in register_pairs()]
    elif channels == 32:
        def register_quadruples():
            return zip(*(range(i, kernels_per_step * 4 + i, 4) for i in range(4)))

        for offset, (r1, r2, r3, r4) in enumerate(register_quadruples()):
            state.text += f"""
    movdqa {offset * 0x40}(%rdx), %xmm{r1}
    movdqa {offset * 0x40 + 0x10}(%rdx), %xmm{r2}
    movdqa {offset * 0x40 + 0x20}(%rdx), %xmm{r3}
    movdqa {offset * 0x40 + 0x30}(%rdx), %xmm{r4}"""

        for offset, (r1, r2, r3, r4) in enumerate(register_quadruples()):
            state.text += f"""
    {pooling_op} {state.width * 0x20 + offset * 0x40}(%rdx), %xmm{r1}
    {pooling_op} {state.width * 0x20 + offset * 0x40 + 0x10}(%rdx), %xmm{r2}
    {pooling_op} {state.width * 0x20 + offset * 0x40 + 0x20}(%rdx), %xmm{r3}
    {pooling_op} {state.width * 0x20 + offset * 0x40 + 0x30}(%rdx), %xmm{r4}"""

        for r1, r2, r3, r4 in register_quadruples():
            state.text += f"""
    {pooling_op} %xmm{r3}, %xmm{r1}
    {pooling_op} %xmm{r4}, %xmm{r2}"""

        output_registers = [f'xmm{r}' for r1, r2, _, _ in register_quadruples() for r in (r1, r2)]
    elif channels == 8:
        def register_triplets():
            return zip(range(0, (kernels_per_step // 2) * 3, 3), range(1, (kernels_per_step // 2) * 3 + 1, 3), range(2, (kernels_per_step // 2) * 3 + 2, 3))
        def extra_register_tuples():
            if (kernels_per_step % 2) == 0:
                return ()
            else:
                return (((kernels_per_step // 2) * 3, (kernels_per_step // 2) * 3 + 1),)

        for offset, (r1, r2, _) in enumerate(register_triplets()):
            state.text += f"""
    movdqa {offset * 0x20}(%rdx), %xmm{r1}
    movdqa {offset * 0x20 + 0x10}(%rdx), %xmm{r2}"""

        for offset, (r1, _) in enumerate(extra_register_tuples()):
            state.text += f"""
    movdqa {(kernels_per_step // 2) * 0x20 + offset * 0x10}(%rdx), %xmm{r1}"""

        for offset, (r1, r2, _) in enumerate(register_triplets()):
            state.text += f"""
    {pooling_op} {state.width * 8 + offset * 0x20}(%rdx), %xmm{r1}
    {pooling_op} {state.width * 8 + offset * 0x20 + 0x10}(%rdx), %xmm{r2}"""

        for offset, (r1, _) in enumerate(extra_register_tuples()):
            state.text += f"""
    {pooling_op} {state.width * 8 + (kernels_per_step // 2) * 0x20 + offset * 0x10}(%rdx), %xmm{r1}"""

        for r1, _, r3 in register_triplets():
            state.text += f"""
    movdqa %xmm{r1}, %xmm{r3}"""

        for r1, r2 in extra_register_tuples():
            state.text += f"""
    movdqa %xmm{r1}, %xmm{r2}"""

        for r1, r2, _ in register_triplets():
            state.text += f"""
    punpcklqdq %xmm{r2}, %xmm{r1}"""

        for _, r2 in extra_register_tuples():
            state.text += f"""
    psrldq $8, %xmm{r2}"""

        for _, r2, r3 in register_triplets():
            state.text += f"""
    punpckhqdq %xmm{r2}, %xmm{r3}"""

        for r1, _, r3 in register_triplets():
            state.text += f"""
    {pooling_op} %xmm{r3}, %xmm{r1}"""

        for r1, r2 in extra_register_tuples():
            state.text += f"""
    {pooling_op} %xmm{r2}, %xmm{r1}"""

        output_registers = [f'xmm{r}' for r, _, _ in register_triplets()] + [f'xmm{r}' for r, _ in extra_register_tuples()]
    else:
        raise NotImplementedError(mode)

    store_op = 'movdqa' if (output_stepsize % 0x10) == 0 else 'movdqu'
    for i, reg in enumerate(output_registers):
        state.text += f"""
    {store_op} %{reg}, {i * 0x10}(%r11)"""

    state.text += f"""
    add ${output_stepsize}, %r11"""

    # Next col
    if num_steps > 1:
        state.text += f"""
    add ${input_stepsize}, %rdx
    dec %ecx
    jnz {col_loop}"""

    # Move to next row
    if num_steps > 1:
        # cursor is currently at first pixel of skipped row
        pixel_offset = state.width * channels
    else:
        # cursor is currently at first pixel of previous row
        pixel_offset = state.width * 2 * channels
    state.text += f"""
    add ${pixel_offset}, %rdx"""

    # Next row
    state.text += f"""
    dec %r9d
    jnz {row_loop}"""

    state.width //= 2
    state.height //= 2

    # Reset pointers
    state.text += """
    movq %rdi, %rdx
    movq %rdx, %r11"""
