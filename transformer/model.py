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

    def __init__(self, d_model, num_heads, flash_attention=False, is_cross_attention=False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.flash_attention = flash_attention
        self.is_cross_attention = is_cross_attention

    @staticmethod
    def attention(q,k,v,d_k,mask):
        
        attention_scores = ((q @ k.transpose(-2,-1) ) / math.sqrt(d_k))

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        attention_scores = torch.softmax(attention_scores, dim=-1)

        return attention_scores @ v  

    # added kv_cache parameter
    def forward(self, q, k, v, mask, block_table, q_len_per_seq, kv_len_per_seq, sequence_id, kv_cache=None):
        # compute  the num_blocks_per_seq and block_size from config
        block_size = core_configurations['block_size']
        num_blocks_per_seq = []
        # upon iterating over sequence_id it makes sure that we only get the size of blocks for the sequences that are currently being processed. otherwise, the block_manager might have random prefill and decode sequences all over which might mess up our logic.
        for seq in sequence_id:
            num_blocks_per_seq.append(len(block_table[seq]))

        batch_size = q.size(0)
        q_len = q.size(1)
        k_len = k.size(1)
        
        self.q = self.W_q(q) # -> q @ W_q
        self.k = self.W_k(k) # -> k @ W_k
        self.v = self.W_v(v) # -> v @ W_v

        
        # so till now the shape is batch_size, seq_len, d_model for all q,k,v
        # now we need to convert to another tensor shape which is :
        # batch_size,seq_len,d_model => batch_size,seq_len, num_heads, d_k -> batch_size, num_heads, seq_len,d_k  
        q = q.view(batch_size, q_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, k_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, k_len, self.num_heads, self.d_k).transpose(1, 2)

        if kv_cache is not None:
            if self.is_cross_attention:
                k, v = kv_cache       
            else:
                old_k, old_v = kv_cache
                k = torch.cat([old_k, k], dim=2)
                v = torch.cat([old_v, v], dim=2)

        new_cache = (k, v)

        if self.flash_attention and kv_cache is None:
            # 1. if flash_attention = true, then we firstly cast the k q v to fp16 cuz our flash attention class is casted to fp16
            q = q.to(torch.float16)
            k = k.to(torch.float16)
            v = v.to(torch.float16)
            # 2. calling the flashattention main class
            O = FlashAttentionFunction.apply(q, k, v)
            
            # 3. cast results back to fp32 again : (
            attention_scores = O.to(torch.float32)
            
        else: 
            attention_scores = self.attention(q, k, v, self.d_k, mask=mask)

        x = self.W_o(attention_scores.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model))

        # return new_cache as well
        return x, new_cache


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

'''
We have two types of mask used here :
1. Padding mask (src_mask) : this is used to ignore the padding tokens in input sentences.  
    Because not all of the sentences are the same length, we need to add padding tokens based on the max_seq_len
    So, when calculating the attention, the padding tokens are ignored

# 1 = real token, 0 = padding
src_mask = [1, 1, 0, 0]

Then in attention, wherever mask is 0, you set the score to `-inf`. After softmax, `e^(-inf) = 0`, so those positions get zero attention weight. They're completely ignored.

'''
'''
The second type of mask is : 
2. Casual mask (tgt_mask): this is done during training when the decoder initial attention block is able to see all the future tokens in a sequence, we have to stop it.
    So we simply add a mask that makes those values infinity and turns them to 0 with softmax applied

    That's `torch.tril` Function used for the lower triangular. 
    Wherever it's 0, set to `-inf` before softmax. Same exact mechanism as padding mask, different shape and purpose.

'''
class Decoder(nn.Module):

    def __init__(self, masked_attention: MultiHeadAttention, feed_forward: FeedForward, d_model):
        super().__init__()
        self.d_model = d_model
        self.masked_attention = masked_attention
        self.residual_connection = nn.ModuleList([ResidualConnections(self.d_model) for _ in range(2)])
        self.feed_forward = feed_forward

    def forward(self, x, tgt_mask, block_table, q_len_per_seq, kv_len_per_seq, sequence_id, sa_cache=None):
        sub_layer, new_sa_cache = self.masked_attention(x, x, x, tgt_mask, block_table, q_len_per_seq, kv_len_per_seq, sequence_id,  kv_cache=sa_cache)
        x = self.residual_connection[0](x, sub_layer)
        sub_layer = self.feed_forward(x)
        x = self.residual_connection[1](x, sub_layer)

        return x, new_sa_cache

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
    
    def forward(self, tgt, tgt_mask, position_ids, block_table, q_len_per_seq, kv_len_per_seq, sequence_id):
        # compute the num_blocks_per_seq
        tgt = self.tgt_pe(self.tgt_embed(tgt), position_ids)
        
        for block in self.decoder_blocks:
            tgt, _ = block(tgt, tgt_mask, block_table, q_len_per_seq, kv_len_per_seq, sequence_id)
        
        return self.projection_layer(tgt)
    

# Now we have all these classes built but nothing that assembles them. So, we will make a `build_transformer` Function. It is just a regular 
# Python function that takes configuration parameters like `d_model`, `num_heads`, `num_blocks`, `vocab_size` 
# and creates one instance of every class you've built, wires them together, and returns a ready-to-use Transformer object.

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