"""
Continuous batching scheduler. Admits waiting sequences, evicts
finished ones, decides what runs each step.
"""

from core.block_manager import BlockManager
from core.sequence import Sequence
import math

class Scheduler:
    def __init__(self, block_manager :BlockManager, block_size, max_len, skip_threshold=3, lookahead=5):
        self.block_manager = block_manager
        self.block_size = block_size
        self.max_len = max_len
        self.skip_threshold = skip_threshold
        self.lookahead = lookahead

        self.waiting_requests = []
        self.running_requests = []
        self.skip_counts = {}

    def schedule(self):
        self._free_finished()
        self._allocate_decode()
        self._admit_waiting()
        return self._build_output()


    def _free_finished(self):   
        # in this function, we check if there are any running requests that needs their blocks to be freed up -> this has two conditions, either we have <EOS> id or max_length reached. This is decided from sequence class

        running_requests_copy = self.running_requests.copy()
        for requests in running_requests_copy:
            if requests.is_finished:
                self.block_manager.release_blocks(requests.seq_id)
                self.running_requests.remove(requests)


    def _allocate_decode(self):
        # iterate through the current running sequences and find which one needs more blocks to be allocated and allocated it.

        for sequences in self.running_requests:
            block_needed = math.ceil((len(sequences.token_ids) + 1) /self.block_size)
            current_block_count = len(self.block_manager.block_table[sequences.seq_id])

            if block_needed > current_block_count:
                self.block_manager.allocate(sequences.seq_id, block_needed-current_block_count)
            
            
    def _admit_waiting(self):
        pass

    def _build_output(self):
        pass