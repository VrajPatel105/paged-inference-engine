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

        return self.embedding(x) * math.sqrt(self.d_model) 


# Positional Encoding class
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        positional_encoded_tensor = torch.zeros((max_seq_len, d_model))

        pos = torch.arange(0, max_seq_len).unsqueeze(1).float()
        i = torch.arange(0, d_model, 2).float()
        den = 10000 ** (2*i / d_model)
        final_num_den = pos / den

        positional_encoded_tensor[:, 0::2] = torch.sin(final_num_den)
        positional_encoded_tensor[:, 1::2] = torch.cos(final_num_den)

        self.register_buffer('pe', positional_encoded_tensor.unsqueeze(0))

    def forward(self, x, position_ids):
        x = x + self.pe[0, position_ids, :]
        return x


# Multi Head attention class
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.block_size = core_configurations['block_size']
        self.num_blocks = core_configurations['num_blocks']
        self.register_buffer('k_cache', torch.zeros(self.num_blocks, self.num_heads, self.block_size, self.d_k)) # buiding the actual tensors that will be used for kv cache in FA
        self.register_buffer('v_cache', torch.zeros(self.num_blocks, self.num_heads, self.block_size, self.d_k))
        

    def forward(self, q, k, v, block_table, q_len_per_seq, kv_len_per_seq, sequence_id, offset, length, position_ids, pos_seq_id):

        num_sequences = len(sequence_id)
        total_tokens = q.size(0)

        # 1. Project
        q = self.W_q(q)
        k = self.W_k(k)
        v = self.W_v(v)

        # 2. Head-split: [total_tokens, num_heads, d_k]
        q = q.view(total_tokens, self.num_heads, self.d_k)
        k = k.view(total_tokens, self.num_heads, self.d_k)
        v = v.view(total_tokens, self.num_heads, self.d_k)

        # 3. Write this step's new K/V into the paged cache pool
        for i in range(total_tokens):
            seq_id = pos_seq_id[i].item()
            p = position_ids[i].item()

            block_idx_in_seq = p // self.block_size
            slot_in_block = p % self.block_size

            block_number = block_table[seq_id][block_idx_in_seq]

            self.k_cache[block_number, :, slot_in_block, :] = k[i]
            self.v_cache[block_number, :, slot_in_block, :] = v[i]

        # 4. num_blocks_per_seq, derived from block_table via sequence_id (correct order)
        num_blocks_per_seq = []
        for seq_id in sequence_id:
            num_blocks_per_seq.append(len(block_table[seq_id.item()]))
        num_blocks_per_seq = torch.tensor(num_blocks_per_seq, dtype=torch.int32, device=q.device)

        # 5. Pad flat Q into [num_sequences, num_heads, max_q_len, head_dim]
        max_q_len = int(length.max().item())
        Q_padded = torch.zeros(num_sequences, self.num_heads, max_q_len, self.d_k, device=q.device, dtype=q.dtype)

        for k_idx in range(num_sequences):
            start = offset[k_idx].item()
            this_len = length[k_idx].item()
            # q[start:start+this_len] is [this_len, num_heads, d_k] -> needs [num_heads, this_len, d_k]
            Q_padded[k_idx, :, :this_len, :] = q[start:start + this_len].transpose(0, 1)

        # 6. Kernel call : K/V come from the pool (self.k_cache/self.v_cache), not from this step's k/v directly
        O, _ = FlashAttentionFunction.apply(
            Q_padded, self.k_cache, self.v_cache,
            q_len_per_seq, block_table, num_blocks_per_seq, self.block_size, kv_len_per_seq
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
        tgt = self.tgt_pe(self.tgt_embed(tgt), position_ids)
        
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
        Decoder(MultiHeadAttention(d_model, num_heads, flash_attention=True), FeedForward(d_model), d_model)
    for _ in range(N)] )
    
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(tgt_embed, tgt_pe, decoder_block_mdlist, projection_layer)
    
    return transformer