# SPDX-License-Identifier: MIT
from dataclasses import dataclass

class LabelManager:
    def __init__(self) -> None:
        self.cur_index = -1
    
    def get_next_label(self, prefix='_loop_'):
        self.cur_index += 1
        return f'{prefix}{self.cur_index}'

@dataclass
class State:
    data_offset: int
    width: int
    height: int
    text: str
    data: str
