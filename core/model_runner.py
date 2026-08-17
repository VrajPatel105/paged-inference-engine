"""
Wires the paged cache and scheduler into the actual model forward
pass -- FlashAttention-2 + INT8 kernels underneath.
"""

from core.block_manager import BlockManager
from core.scheduler import Scheduler
from core.sequence import Sequence
from core.config import core_configurations
import queue
from collections import deque # for the new_request dequeue
import torch
from transformer.model import build_transformer
from transformer.config import transformer_configurations
from transformer.load_checkpoint import load_trained_weights
model = load_trained_weights('transformer/decoder_only.pt')

def run(q, tok, output_state=None, window_map=None):

    new_requests = deque()

    block_manager_obj = BlockManager(core_configurations['num_blocks'], core_configurations['block_size'])

    scheduler_obj = Scheduler(block_manager_obj, core_configurations['block_size'], core_configurations['max_len'], skip_threshold=core_configurations['scheduler_skip_threshold'], lookahead=core_configurations['scheduler_lookahead'])

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
            window_id, curr_req = new_requests.popleft()
            scheduler_obj.add_request(Sequence(seq_id=seq_cnt, prompt_token_ids=curr_req)) # is_finished is False by default, so we are not passing it in this call
            if window_map is not None:
                window_map[seq_cnt] = window_id
            seq_cnt += 1

        output = scheduler_obj.schedule()
        prefill_seq = output.prefill_seqs
        decode_seq = output.decode_seqs

        if not prefill_seq and not decode_seq:
            continue

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
        length = [] # can be reused as q_len_per_seq for FA kernel
        offset = []
        position_ids = []
        pos_seq_id = []
        kv_len_per_seq = [] # required for FA kernel
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
            kv_len_per_seq.append(num_new_token)
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
            kv_len_per_seq.append(len(seq.token_ids) + 1)
            flat_tokens.append(seq.token_ids[-1])

        # converting all the lists to tensors since we are going to send it to forward pass
        sequence_id = torch.tensor(sequence_id, dtype=torch.int32, device='cuda')
        length = torch.tensor(length, dtype=torch.int32, device='cuda')
        offset = torch.tensor(offset, dtype=torch.int32, device='cuda')
        position_ids = torch.tensor(position_ids, dtype=torch.int32, device='cuda')
        pos_seq_id = torch.tensor(pos_seq_id, dtype=torch.int32, device='cuda')
        flat_tokens = torch.tensor(flat_tokens, dtype=torch.int32, device='cuda')
        kv_len_per_seq = torch.tensor(kv_len_per_seq, dtype=torch.int32, device='cuda')

        # now finally building the main mask using position_ids and pos_seq_id (insstead of using two loops which we could, we are just using broadcasting)
        
        # Grid 1: same sequence (symmetric, order of unsqueeze doesn't matter which is which)
        # same_seq = pos_seq_id.unsqueeze(1) == pos_seq_id.unsqueeze(0)

        # # Grid 2: not future (directioanl: row i is the query, col j is the key; j must be <= i)
        # not_future = position_ids.unsqueeze(0) <= position_ids.unsqueeze(1)

        # # Final mask: both conditions must hold
        # tgt_mask = same_seq & not_future
        # we dont need the mask anymore since the FA already now implements it internally. 

        logits = model(flat_tokens, block_manager_obj.block_table, length, kv_len_per_seq, sequence_id, offset, length, position_ids, pos_seq_id) # adding sequence_id as well because we need it for computing the num_blocks_per_seq in attention module by making sure that the sequence ids are exactly in order

        # the logits now contain the per sequence output. logits shape : [total_tokens, vocab_size]
        # we only want the last row that was contributed to the sequence and then append it to that particular seq_id's sequence
        # one row of logits looks like : [seq_a, [0.1,0.4,0.4,0.1]] the list is probabilities
        # but if we not here, in logits, we only want that particular sequence's last logit (row)
        # so we can use (offset[k] + length[k] - 1) to exactly reach to each last indicees of each unique row
        # so we just build a matrix of the list of last indices to extract and then we can iterate over logits to get those particular rows itself.
        num_sequences = len(prefill_seq) + len(decode_seq)
        last_indices = offset + length - 1

        # now we have exact index values of tokens that we need 
        last_seq_logits = logits[last_indices] # we directly index into the 2d tensor to get those rows rather than running a loop 
        # now converting each list of probabilities to the max probabitliy
        new_tokens = torch.argmax(last_seq_logits, dim=-1).tolist() # converting tensor back to py list since it's more easier to iterate further on when we write the new tokens into sequences
        # now adding the new tokens to both prefill and decode's sequences' metadata.
        # we will again get two loops same logic as what we ddi before to make it less confusing

        #prefill 
        current_index = 0
        for seq in prefill_seq:
            if new_tokens[current_index] == tok.EOS_ID:
               seq.is_finished=True
            else: 
                 seq.token_ids.append(new_tokens[current_index])
            current_index += 1

        for seq in decode_seq:
            if new_tokens[current_index] == tok.EOS_ID:
               seq.is_finished=True
            else: 
                 seq.token_ids.append(new_tokens[current_index])
            current_index += 1


        if output_state is not None:
            for seq in prefill_seq + decode_seq:
                win_id = window_map.get(seq.seq_id, seq.seq_id) if window_map is not None else seq.seq_id
                output_state[win_id] = tok.decode_sentence(seq.token_ids)