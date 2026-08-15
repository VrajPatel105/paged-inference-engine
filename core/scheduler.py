"""
Continuous batching scheduler. Admits waiting sequences, evicts
finished ones, decides what runs each step.
"""

from core.block_manager import BlockManager
from core.sequence import Sequence
import math
from dataclasses import dataclass

@dataclass
class SchedulerOutput:
    prefill_seqs: list[Sequence]
    decode_seqs: list[Sequence]


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

    def add_request(self, seq: Sequence):
        self.waiting_requests.append(seq)


    def _free_finished(self):   
        # in this function, we check if there are any running requests that needs their blocks to be freed up -> this has two conditions, either we have <EOS> id or max_length reached. This is decided from sequence class

        running_requests_copy = self.running_requests.copy()
        for sequences in running_requests_copy:
            if sequences.is_finished or len(sequences.token_ids) >= self.max_len:
                self.block_manager.release_blocks(sequences.seq_id)
                self.running_requests.remove(sequences)


    def _allocate_decode(self):
        # iterate through the current running sequences and find which one needs more blocks to be allocated and allocated it.

        for sequences in self.running_requests:
            block_needed = math.ceil((len(sequences.token_ids) + 1) /self.block_size)
            current_block_count = len(self.block_manager.block_table[sequences.seq_id])

            if block_needed > current_block_count:
                self.block_manager.allocate(sequences.seq_id, block_needed-current_block_count)
            

# Normal mode: otherwise, walk the queue checking up to self.lookahead candidates. For each, compute needed blocks, check can_allocate. If it fits: allocate, admit, move it. If not: increment its skip count, move to the next candidate (up to the lookahead cap).

            
    def _admit_waiting(self):
        # here instead of FCFS policy, we are implementing a lookahead window threshold with a per sequence threshold.
        # So, we look at the first n (lookhead) items from waiting list and iterate over to check which one will be able to fit in the free memory currently available
        # based on that, the ones that were not able to fit will have their skip_count counter increase by 1. After a threshold (we are taking threshold of 3 here),
        # if a block has excedeed it's skip_count's count by threshold, then we will stop all the admission of smaller sequences and only focus on the one that has exceeded the threshold
        # until that exceeded sequence is not allocated, we will not move ahead for others. 
        # by implementing the lookahead logic, we prevent the starvation. (which is why this i implemented teh lookhead logic at the first place )

        if not self.waiting_requests:
            return
        
        # First : Blocking Mode
        front_sequence = self.waiting_requests[0]

        if self.skip_counts.get(front_sequence.seq_id, 0) >= self.skip_threshold:
            needed_blocks = math.ceil(len(front_sequence.prompt_token_ids) / self.block_size)
            if self.block_manager.can_allocate(needed_blocks):
                self.block_manager.allocate(front_sequence.seq_id, needed_blocks)
                self.running_requests.append(front_sequence)
                self.waiting_requests.remove(front_sequence)
                self.skip_counts.pop(front_sequence.seq_id, None)
            return
        # 2. Second: else, Normal Mode
        else:
            for sequences in self.waiting_requests[:self.lookahead]:
                needed_blocks = math.ceil(len(sequences.prompt_token_ids) / self.block_size)
                if self.block_manager.can_allocate(needed_blocks):
                    self.block_manager.allocate(sequences.seq_id, needed_blocks)
                    self.running_requests.append(sequences)
                    self.waiting_requests.remove(sequences)
                    self.skip_counts.pop(sequences.seq_id, None)
                else:
                    current_count = self.skip_counts.get(sequences.seq_id, 0)
                    self.skip_counts[sequences.seq_id] = current_count + 1
        

    def _build_output(self):
        # this builds output for the model runner which simply decides if currnt one is a prefill or decode step

        prefill_seq = []
        decode_seq = []

        for sequences in self.running_requests:
            if(len(sequences.token_ids) == len(sequences.prompt_token_ids)):
                prefill_seq.append(sequences)
            else:
                decode_seq.append(sequences)

        return SchedulerOutput(prefill_seqs=prefill_seq, decode_seqs=decode_seq)