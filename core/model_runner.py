"""
Wires the paged cache and scheduler into the actual model forward
pass -- FlashAttention-2 + INT8 kernels underneath.
"""

from core.block_manager import BlockManager
from core.scheduler import Scheduler
from core.sequence import Sequence
from core.config import configurations
import queue
from collections import deque # for the new_request dequeue

def run(q):

    new_requests = deque()

    block_manager_obj = BlockManager(configurations['num_blocks'], configurations['block_size'])

    scheduler_obj = Scheduler(block_manager_obj, configurations['block_size'], configurations['max_len'], skip_threshold=configurations['scheduler_skip_threshold'], lookahead=configurations['scheduler_lookahead'])

    seq_cnt = 0  # unique count for each new sequence id (new user)

    while True:

        try:
            while True:
                item = q.get_nowait()
                new_requests.append(item)
        except queue.Empty:
            pass

        # convert each entry from new_requests into Sequence obj and pass to scheduler to add request
        while new_requests:
            curr_req = new_requests.popleft()
            scheduler_obj.add_request(Sequence(seq_id=seq_cnt, prompt_token_ids=curr_req)) # is_finished is False by default, so we are not passing it in this call
            seq_cnt += 1

        scheduler_obj.schedule() # this will return the output for prefill and decode sequences
