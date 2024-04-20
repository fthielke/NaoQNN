# SPDX-License-Identifier: MIT
from typing import Tuple
from .state import LabelManager, State

def asm_copy_memory(in_ptr: str, out_ptr: str, num_bytes_per_sample: int, label_manager: LabelManager, state: State):
    assert (num_bytes_per_sample % 16) == 0
    registers_per_loop = next(i for i in range(16, 0, -1) if (num_bytes_per_sample // 16) % i == 0)
    num_steps_per_sample = (num_bytes_per_sample) // (16 * registers_per_loop)

    if num_steps_per_sample > 1:
        state.text += f"""
    imull ${num_steps_per_sample}, %r8d, %ecx"""
    else:
        state.text += """
    movl %r8d, %ecx"""

    copy_loop = label_manager.get_next_label()
    state.text += f"""
{copy_loop}:"""

    for i in range(registers_per_loop):
        state.text += f"""
    movdqa {i * 0x10}({in_ptr}), %xmm{i}"""

    for i in range(registers_per_loop):
        state.text += f"""
    movdqa %xmm{i}, {i * 0x10}({out_ptr})"""

    state.text += f"""
    add ${(i + 1) * 0x10}, {in_ptr}
    add ${(i + 1) * 0x10}, {out_ptr}
    dec %ecx
    jnz {copy_loop}"""


def asm_zero_memory(out_ptr: str, num_bytes_per_sample: int, label_manager: LabelManager, state: State):
    registers_per_loop = next(i for i in range(16, 0, -1) if ((num_bytes_per_sample + 15) // 16) % i == 0)
    num_steps_per_sample = ((num_bytes_per_sample + 15) // 16) // registers_per_loop

    if num_steps_per_sample > 1:
        state.text += f"""
    imull ${num_steps_per_sample}, %r8d, %ecx"""
    else:
        state.text += """
    movl %r8d, %ecx"""

    for i in range(registers_per_loop):
        state.text += f"""
    pxor %xmm{i}, %xmm{i}"""

    zero_loop = label_manager.get_next_label()
    state.text += f"""
{zero_loop}:"""

    for i in range(registers_per_loop):
        state.text += f"""
    movdqa %xmm{i}, {i * 0x10}({out_ptr})"""

    state.text += f"""
    add ${(i + 1) * 0x10}, {out_ptr}
    dec %ecx
    jnz {zero_loop}"""


def asm_pad_image(in_ptr: str, out_ptr: str, num_channels: int, pads: Tuple[int, int, int, int], label_manager: LabelManager, state: State):
    """
    pads: top right bottom left
    """

    state.text += f"""
    movq {out_ptr}, %r9"""
    asm_zero_memory('%r9', num_channels * (state.width + pads[1] + pads[3]) * (state.height + pads[0] + pads[2]), label_manager, state)

    assert ((state.width * num_channels) % 16) == 0
    num_registers = (state.width * num_channels) // 16
    store_op = 'movdq' + ('a' if (num_channels % 16) == 0 else 'u')

    state.text += f"""
    addq ${(pads[0] * (state.width + pads[1] + pads[3]) + pads[3]) * num_channels}, {out_ptr}"""

    # Loop over samples (R9D)
    batch_loop = label_manager.get_next_label()
    state.text += f"""
    movl %r8d, %r9d
{batch_loop}:"""

    # Loop over rows (ECX)
    row_loop = label_manager.get_next_label()
    state.text += f"""
    movl ${state.height}, %ecx
{row_loop}:"""

    for i in range(num_registers):
        state.text += f"""
    movdqa {i * 0x10}({in_ptr}), %xmm{i}"""
    for i in range(num_registers):
        state.text += f"""
    {store_op} %xmm{i}, {i*0x10}({out_ptr})"""

    state.text += f"""
    addq ${state.width * num_channels}, {in_ptr}
    addq ${(state.width + pads[1] + pads[3]) * num_channels}, {out_ptr}"""

    # Next row
    state.text += f"""
    dec %ecx
    jnz {row_loop}"""

    state.text += f"""
    addq ${(state.width + pads[1] + pads[3]) * num_channels}, {out_ptr}"""

    # Next sample
    state.text += f"""
    dec %r9d
    jnz {batch_loop}"""

    state.width += pads[1] + pads[3]
    state.height += pads[0] + pads[2]
