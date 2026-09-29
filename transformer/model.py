import torch
import torch.nn as nn
import math
from core.config import core_configurations

import sys

from kernels.flash_attention import FlashAttentionFunction

# Embeddings class
class Embedding(nn.Module):

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, self.d_model)
        
    def forward(self, x):

        return self.embedding(x) * math.sqrt(self.d_model) # converting 1d flat token to 2d -> x : [total_token, d_model]


# Positional Encoding class
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        positional_encoded_tensor = torch.zeros((max_seq_len, d_model)) # shape (max_seq_len, d_model) -> (200 * 512)

        pos = torch.arange(0, max_seq_len).unsqueeze(1).float() # [200].unsqueeze(1) -> [200,1]
        i = torch.arange(0, d_model, 2).float() # [256]
        den = 10000 ** (2*i / d_model) #[256]
        final_num_den = pos / den # [200,1] / [256] -> [200,256]

        positional_encoded_tensor[:, 0::2] = torch.sin(final_num_den)
        positional_encoded_tensor[:, 1::2] = torch.cos(final_num_den)

        self.register_buffer('pe', positional_encoded_tensor.unsqueeze(0)) # (200 * 512).unsqueeze(0) -> [1, 200, 512]

    def forward(self, x, position_ids):
        x = x + self.pe[0, position_ids, :] # [total_tokens, 512] + [total_tokens, 512] -> [total_tokens, 512]
        return x # [total_token, d_model]


# Multi Head attention class
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model # 512
        self.num_heads = num_heads # 8
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads # 64
        self.W_q = nn.Linear(d_model, d_model) # [512, 512]
        self.W_k = nn.Linear(d_model, d_model) # [512, 512]
        self.W_v = nn.Linear(d_model, d_model) # [512, 512]
        self.W_o = nn.Linear(d_model, d_model) # [512, 512]
        self.block_size = core_configurations['block_size'] # 64
        self.num_blocks = core_configurations['num_blocks'] # 256
        self.register_buffer('k_cache', torch.zeros(self.num_blocks, self.num_heads, self.block_size, self.d_k, dtype=torch.int8)) # [256, 8, 64, 64]
        self.register_buffer('v_cache', torch.zeros(self.num_blocks, self.num_heads, self.block_size, self.d_k, dtype=torch.int8)) # [256, 8, 64, 64]
        self.register_buffer('k_scale', torch.zeros(self.num_blocks, self.num_heads)) # [256, 8]
        self.register_buffer('v_scale', torch.zeros(self.num_blocks, self.num_heads)) # [256, 8]

    def forward(self, q, k, v, block_table, q_len_per_seq, kv_len_per_seq, sequence_id, offset, length, position_ids, pos_seq_id):

        # q,k,v = [total_token, d_model]

        num_sequences = len(sequence_id)
        total_tokens = q.size(0) # total_tokens is the count of how many new tokens are generated. for decode it will be just one token,  for prefill it will be the prompt size

        # 1. Project
        q = self.W_q(q) # [total_tokens, 512] @ [512, 512] + bias[512] -> [total_tokens, 512]
        k = self.W_k(k) # [total_tokens, 512] @ [512, 512] + bias[512] -> [total_tokens, 512]
        v = self.W_v(v) # [total_tokens, 512] @ [512, 512] + bias[512] -> [total_tokens, 512]

        # 2. Head-split: [total_tokens, num_heads, d_k]
        q = q.view(total_tokens, self.num_heads, self.d_k) # [total_tokens, 512] -> [total_tokens, 8, 64]
        k = k.view(total_tokens, self.num_heads, self.d_k) # [total_tokens, 512] -> [total_tokens, 8, 64]
        v = v.view(total_tokens, self.num_heads, self.d_k) # [total_tokens, 512] -> [total_tokens, 8, 64]

        # 3. Write this step's new K/V into the paged cache pool
        for i in range(total_tokens):

            # here in order to understand this well, we can imagine a situation (got this great example from ai -_- )
            # there are 256 drawers  -> block_number
            # each drawer has 8 books  -> head
            # each book has 64 pages  -> block_size
            # each page has 64 numbers  -> d_k


            seq_id = pos_seq_id[i].item() # pos_seq_id[i] # here .item() converts cuda tensor to a plain python int
            p = position_ids[i].item()

            block_idx_in_seq = p // self.block_size # for example 11 / 32 -> 0 so this sequence is in block 0
            slot_in_block = p % self.block_size # what slot will this current value go in?  # 11 % 32 -> 11 so the slot is 11 meaning from 0 to book 11 is what we want.

            block_number = block_table[seq_id][block_idx_in_seq] # what current block does this sequence have? (or, which drawer does this token belong to?)
            # we finally get the block number

            # quantization
            # we will do : dequant existing -> append nwe -> rescale -> requant -> write back


            # so for example, if we are to extract [6, :, 0:8, :] => extract the value from : drawer 6 -> all book -> first 0 to 7 pages -> all numbers
            # here we are using one scale per book (per head)

            dequantized_k = (self.k_cache[block_number, :, 0:slot_in_block, :]).to(torch.float32) * self.k_scale[block_number, :].unsqueeze(-1).unsqueeze(-1)
            # shape trace :
            # initially, bufffer shape : [256, 8, 64, 64]
            # k_cache[block_number, :, 0:slot_in_block, :] -> [8, slot_in_block, 64]

            # k_scale initially : [256, 8]
            # k_scale[block_number, :]                     -> [8]

            # .unsqueeze(-1)                               -> [8, 1]
            # .unsqueeze(-1)                               -> [8, 1, 1]

            # final dequantized_k : 
            # [8, slot_in_block, 64] * [8, 1, 1]           -> [8, slot_in_block, 64]
            dequantized_v = (self.v_cache[block_number, :, 0:slot_in_block, :]).to(torch.float32) * self.v_scale[block_number, :].unsqueeze(-1).unsqueeze(-1)

            # our dequantize_k is shape [block_number, slot_in_block, d_k] and k[i] is still [num_heads, d_k] so we unsqueeze k[i] to [num_heads, 1, d_k]
            full_k = torch.cat([dequantized_k, k[i].unsqueeze(1)], dim=1) 
            # dequantized_k =  [8, slot_in_block, 64] and k[i] = [total_tokens, 8, 64][i] = [8, 64].unsqueeze(1) -> [8, 1, 64]
            # therefore,  full_k = torch.cat([8, slot_in_block, 64] , [8, 1, 64], dim=1)
            # (torch.cat joins the tensors in a way that every other dimension must match and the dimension to be concatenated will be different), and dim=1 means we are concatenating along the 1st index of that tensor which is the middle one)
            # full_k = torch.cat([8, slot_in_block, 64] , [8, 1, 64], dim=1) = [8, slot_in_block+1, 64]
            full_v = torch.cat([dequantized_v, v[i].unsqueeze(1)], dim=1)


            # lets get the scale per head now that we got the tensors
            abs_full_k_intermediate_step = torch.max(torch.abs(full_k), dim=2).values
            # shape breakdown : torch.max(torch.abs(full_k), dim=2) -> full_k = [8, slot_in_block+1, 64]
            # torch.abs will output the same tensor with just all +ve values
            # torch.max([8, slot_in_block+1, 64], dim=2)
            # finding the maximum value along that 64 index (the numbers in our pages of the books)
            # therefore,  we get torch.max([8, slot_in_block+1, 64], dim=2) = [8, slot_in_block+1]
            # abs_full_k_intermediate_step = [8, slot_in_block+1]


            # again reducing it more to num_heads shape. currently itthe abs_full_k_intermediate_step is [num_heads, slot_in_block+1] and we have to make it [num_heads]
            k_scale_new = torch.max(abs_full_k_intermediate_step, dim=1).values / 127
            # k_scale_new = [8]

            abs_full_v_intermediate_step = torch.max(torch.abs(full_v), dim=2).values
            v_scale_new = torch.max(abs_full_v_intermediate_step, dim=1).values / 127

            # requantize -> for symmetric it's roud(value/scale)
            quantized_k = torch.clamp(torch.round(full_k/k_scale_new.unsqueeze(-1).unsqueeze(-1)), -127,127).to(torch.int8)
            # k_scale_new.unsqueeze(-1).unsqueeze(-1) = [8] -> [8,1] -> [8,1,1]
            # full_k / k_scale_new = [8, slot_in_block+1, 64]  / [8, 1, 1] -> [8, slot_in_block+1, 64]
            # torch.round([8, slot_in_block+1, 64]) -> rounds the values to the nearest integers
            # torch.clamp([8, slot_in_block+1, 64], -127, 127) -> rescales the value in -127 to 127 range


            quantized_v = torch.clamp(torch.round(full_v/v_scale_new.unsqueeze(-1).unsqueeze(-1)), -127,127).to(torch.int8)

            # writing the scale back into scale buffers. k_scale is [num_blocks, num_heads] and k_scake_new/v_scale_new is [num_heads] so we index into that particular block_number in kscale and vscale
            self.k_scale[block_number] = k_scale_new
            self.v_scale[block_number] = v_scale_new

            # now finally, quaitized_k/v is shape [num_heads, slot_in_block+1, d_k]
            self.k_cache[block_number, :, 0:slot_in_block+1, :] = quantized_k
            self.v_cache[block_number, :, 0:slot_in_block+1, :] = quantized_v



        # 4. num_blocks_per_seq, derived from block_table via sequence_id (correct order)
        # this gives us how much number of block does  each sequence have as of now
        num_blocks_per_seq = []
        for seq_id in sequence_id:
            num_blocks_per_seq.append(len(block_table[seq_id.item()]))
        num_blocks_per_seq = torch.tensor(num_blocks_per_seq, dtype=torch.int32, device=q.device) # converting the python list to a tensor
        

        # 5. Pad flat Q into [num_sequences, num_heads, max_q_len, head_dim]
        max_q_len = int(length.max().item())
        Q_padded = torch.zeros(num_sequences, self.num_heads, max_q_len, self.d_k, device=q.device, dtype=q.dtype) # making an empty 4d tensor that will be sent to fa 2 kernel

        for k_idx in range(num_sequences):
            start = offset[k_idx].item()
            this_len = length[k_idx].item()
            # q[start:start+this_len] is [this_len, num_heads, d_k] -> needs [num_heads, this_len, d_k]
            Q_padded[k_idx, :, :this_len, :] = q[start:start + this_len].transpose(0, 1)
                #  q[start:start + this_len].transpose(0, 1) -> q[this_len, 8, 64].transpose(0,1) -> [8, this_len, 64]
                # Q_padded [k_idx, :, :this_len, :] = [8, this_len, 64]  (k_idx is gone)
                # shapes match -> copy fills row k_idx's first this_len slots
                # Q_padded itself stays [num_sequences, 8, max_q_len, 64]


        # due to mismatch in blocktable, we have it in py dict and kernel expects it in iterable cuda tensor 
        # Convert block_table (dict of Python lists) into a padded CUDA tensor for the kernel
        max_blocks_this_step = max(num_blocks_per_seq).item() if torch.is_tensor(num_blocks_per_seq) else max(num_blocks_per_seq)

        block_table_tensor = torch.zeros(num_sequences, max_blocks_this_step, dtype=torch.int32, device=q.device)

        for k_idx, seq_id in enumerate(sequence_id):
            seq_id_val = seq_id.item()
            blocks_for_seq = block_table[seq_id_val]
            block_table_tensor[k_idx, :len(blocks_for_seq)] = torch.tensor(blocks_for_seq, dtype=torch.int32, device=q.device)
            # this  whole step just converts the block table into tensor of shape [num_sequences, max_blocks_this_step]


        # 6. Kernel call : K/V come from the pool (self.k_cache/self.v_cache), not from this step's k/v directly
        O = FlashAttentionFunction.apply(
            Q_padded, self.k_cache, self.v_cache,
            q_len_per_seq, block_table_tensor, num_blocks_per_seq, self.block_size, kv_len_per_seq,
            self.k_scale, self.v_scale
        )

        # 7. Unpad O back to flat [total_tokens, num_heads, d_k]
        O_flat = torch.zeros(total_tokens, self.num_heads, self.d_k, device=q.device, dtype=O.dtype)
        for k_idx in range(num_sequences):
            start = offset[k_idx].item()
            this_len = length[k_idx].item()
            O_flat[start:start + this_len] = O[k_idx, :, :this_len, :].transpose(0, 1)

        # 8. Output projection
        attention_scores = O_flat.to(torch.float32)
        x = self.W_o(attention_scores.reshape(total_tokens, self.d_model))

        return x


# Feed forward class
class FeedForward(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_model * 4 ,d_model)

    def forward(self,x):
        return self.linear2(self.relu(self.linear1(x)))



# LayerNorm class
class LayerNorm(nn.Module):

    def __init__(self, d_model, eps = 0.00001):
        super().__init__()
        self.eps = eps 
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))


    def forward(self,x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * ((x - mean) / (std + self.eps)) + self.bias


# residual class
class ResidualConnections(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.norm = LayerNorm(d_model)


    def forward(self,x, sublayer):
      x = x + sublayer
      return self.norm(x)

# Decoder Class

class Decoder(nn.Module):

    def __init__(self, masked_attention: MultiHeadAttention, feed_forward: FeedForward, d_model):
        super().__init__()
        self.d_model = d_model
        self.masked_attention = masked_attention
        self.residual_connection = nn.ModuleList([ResidualConnections(self.d_model) for _ in range(2)])
        self.feed_forward = feed_forward

    def forward(self, x, block_table, q_len_per_seq, kv_len_per_seq, sequence_id, offset, length, position_ids, pos_seq_id):
        sub_layer = self.masked_attention(x, x, x, block_table, q_len_per_seq, kv_len_per_seq, sequence_id, offset, length, position_ids, pos_seq_id)
        x = self.residual_connection[0](x, sub_layer)
        sub_layer = self.feed_forward(x)
        x = self.residual_connection[1](x, sub_layer)

        return x

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.linear_layer = nn.Linear(d_model,vocab_size)

    def forward(self,x):

        return torch.log_softmax(self.linear_layer(x), dim=-1)

class Transformer(nn.Module):

    def __init__(self, tgt_embed: Embedding, tgt_pe: PositionalEncoding, 
                 decoder_blocks: nn.ModuleList, projection_layer: ProjectionLayer):
        super().__init__()
        self.tgt_embed = tgt_embed
        self.tgt_pe = tgt_pe
        self.decoder_blocks = decoder_blocks
        self.projection_layer = projection_layer
    
    def forward(self, tgt, block_table, q_len_per_seq, kv_len_per_seq, sequence_id, offset, length, position_ids, pos_seq_id):
        # compute the num_blocks_per_seq
        tgt = self.tgt_pe(self.tgt_embed(tgt), position_ids) # this returns shape [total_token, d_model]
        
        for block in self.decoder_blocks:
            tgt = block(tgt, block_table, q_len_per_seq, kv_len_per_seq, sequence_id, offset, length, position_ids, pos_seq_id)        
        return self.projection_layer(tgt)
    


def build_transformer(configurations):
# this function needs to have objects of classes: embeddings, PE, encoder_blocks, decoder_blocks and projection_Layer
    d_model = configurations['d_model']
    num_heads = configurations['num_heads']
    N = configurations['num_blocks']
    tgt_max_seq_len = configurations['tgt_max_seq_len']
    tgt_vocab_size = configurations['tgt_vocab_size']

    # embeddings 
    tgt_embed = Embedding(d_model, tgt_vocab_size)

    # positional encoding 
    tgt_pe = PositionalEncoding(d_model, tgt_max_seq_len)
    
    
    decoder_block_mdlist = nn.ModuleList([
        Decoder(MultiHeadAttention(d_model, num_heads), FeedForward(d_model), d_model)
    for _ in range(N)] )
    
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(tgt_embed, tgt_pe, decoder_block_mdlist, projection_layer)
    
    return transformer