"""
Paged KV cache allocator. Owns the pool of fixed-size pages and each
sequence's block table (logical position -> physical page).
"""

from enum import Enum
from dataclasses import dataclass

class AllocStatus(Enum):
    SUCCESS = 0
    INSUFFICIENT_SPACE = 1
    FAILURE = 2

@dataclass
class AllocResult:
    status : AllocStatus
    block_ids : list[int]

class BlockManager:

    def __init__(self, num_blocks, block_size):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.st = list(range(num_blocks))
        self.block_table: dict[int, list[int]] = {}

    def allocate(self, seq_id: int, num_blocks: int) -> AllocResult:
        # main idea for this function is to take the seq_id and the number of blocks it wants and we firstly 
        # check if we have that number of free blocks available or not and then depending on that we take descisions.

        if self.num_free_blocks() < num_blocks:
            return AllocResult(AllocStatus.INSUFFICIENT_SPACE, [])

        free_blocks_indices = []
        for _ in range(num_blocks):
            free_blocks_indices.append(self.st.pop())

        if seq_id in self.block_table:
                self.block_table[seq_id].extend(free_blocks_indices)
        else: 
            self.block_table[seq_id] = free_blocks_indices

        return AllocResult(AllocStatus.SUCCESS, free_blocks_indices)


    def release_blocks(self, seq_id: int) -> AllocStatus:
        # in this func, we take the seq_id and just clear out that key's (seq_id) values from block_table

        if seq_id not in self.block_table:
            return AllocStatus.FAILURE

        allocated_blocks = self.block_table[seq_id]

        for blocks in allocated_blocks:
            self.st.append(blocks)

        del self.block_table[seq_id]

        return AllocStatus.SUCCESS

    def num_free_blocks(self) -> int:
        return len(self.st)