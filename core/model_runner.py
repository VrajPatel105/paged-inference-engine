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
import torch

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

        prefill_seq, decode_seq = scheduler_obj.schedule()

        # Note: here, prefill_seq / decode_seq are Sequence objects that holds their own metadata (seq_id, propmt_token_id, is_finished)

        # we now have to call the model's forrward pass 
        # here i have concluded to get a 1D tensor from a combination of prefill_seq and decode_seq that will be get sent over to the forward pass.
        # but, on top of that, for PE, we need to also send position_ids tensor, then 
        # a mask tensor  : the main idea here is that we are computing everything for each and every single sequence in prefill and decode in just one forward pass.
        # since in the model, all the blocks except attention do not require any dependency on other tokens / sequence in the data.
        # so we can just send a single huge 1d tensor that has all the prefill and decode sequences and in just one forward pass, all of that is going to be comptued.
        # but here, attention is the only part that need the context of all the other tokens present in that sequence.
        # so we also need a mask that will esentially be a huge [total_tokens, total_tokens] tensor for each new sequence, that sequence can only 
        # attend to the tokens present in that sequence only. so if there's another new sequence B, it cannot look at previous sequence's tokens.

        # lets first build the mask itself . shape : [total_tokens, total_tokens]

        sequence_id = []
        length = []
        offset = []
        position_ids = []
        pos_seq_id = []
        offset_cnt = 0

        # main 1d tensor that will be flat token
        flat_tokens = []

        for seq in prefill_seq:
            num_new_token = len(seq.prompt_token_ids)
            sequence_id.append(seq.seq_id)
            length.append(num_new_token)
            offset.append(offset_cnt)
            offset_cnt = offset_cnt + len(seq.prompt_token_ids) # increasing the count for the next seq's starting pos to be recorded
            pos_seq_id.extend([seq.seq_id] * num_new_token)
            position_ids.extend(range(len(seq.token_ids)))
            flat_tokens.extend(seq.prompt_token_ids)

        # now the decode loop
        for seq in decode_seq:
            num_new_token = 1 # just for more clarity, declaring a var for num_new_token even thought we know that for decode, it's just 1 new token added.
            sequence_id.append(seq.seq_id)
            length.append(num_new_token)
            offset.append(offset_cnt)
            offset_cnt = offset_cnt + num_new_token
            pos_seq_id.extend([seq.seq_id] * num_new_token)
            position_ids.extend([len(seq.token_ids)])
            flat_tokens.append(seq.token_ids[-1])

        # converting all the lists to tensors since we are going to send it to forward pass
        sequence_id = torch.tensor(sequence_id, dtype=torch.int32)
        length = torch.tensor(length, dtype=torch.int32)
        offset = torch.tensor(offset, dtype=torch.int32)
        position_ids = torch.tensor(position_ids, dtype=torch.int32)
        pos_seq_id = torch.tensor(pos_seq_id, dtype=torch.int32)

        # now finally building the main mask using position_ids and pos_seq_id (insstead of using two loops which we could, we are just using broadcasting)
        
        # Grid 1: same sequence (symmetric, order of unsqueeze doesn't matter which is which)
        same_seq = pos_seq_id.unsqueeze(1) == pos_seq_id.unsqueeze(0)

        # Grid 2: not future (directioanl: row i is the query, col j is the key; j must be <= i)
        not_future = position_ids.unsqueeze(0) <= position_ids.unsqueeze(1)

        # Final mask: both conditions must hold
        tgt_mask = same_seq & not_future

