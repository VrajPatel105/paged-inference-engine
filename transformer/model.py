import torch
import torch.nn as nn
import math
from config import configurations

import sys
sys.path.append("/mnt/c/dev/projects/cpp-gpu-inference/5. flash-attention")

from flash_attention import FlashAttentionFunction

# Implementation pipeline: 
# Embedding 
# Positional Encoding 
# Multi-Head Attention 
# Add & Norm
# Feedforward
# Add & Norm 
# stack into Encoder layer
#  Decoder (with cross-attention)
#  final Linear + Softmax.


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

    def forward(self,x):
        seq_len = x.shape[1]
        x = x + self.pe[:, :seq_len, :]
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
    def forward(self, q, k, v, mask, kv_cache=None):

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

# Encoder class
'''
What we have to do for encoder class:
get the tensor x that is already positional encoded.
take the tensor x and have it pass through the multiheadattention block
then add the output from multiheadattention block and the initial data x which is obtained by the residualconnections block
and then apply layer norm on this.
now we have x tensor
then apply FNN layer  -> x tensor to x_tensor
then again apply add & norm by residual connections and layernorm.
Now we have our final tensor x ready.
'''

class Encoder(nn.Module):
    def __init__(self, multi_head_attention: MultiHeadAttention, feed_forward: FeedForward, d_model):
        super().__init__()
        self.d_model = d_model
        self.multi_head_attention = multi_head_attention
        self.residual_connection = nn.ModuleList([ResidualConnections(self.d_model) for _ in range(2)])
        self.feed_forward = feed_forward

    def forward(self,x, src_mask):
        sub_layer, _ = self.multi_head_attention(x,x,x,src_mask) # x q k v
        x = self.residual_connection[0](x, sub_layer)
        sub_layer = self.feed_forward(x)
        x = self.residual_connection[1](x, sub_layer)

        # we can also write the above implementation as this : 
        #   x = self.residual_connection[0](x, lambda x: self.multi_head_attention(x, x, x, src_mask))
        #   x = self.residual_connection[1](x, self.feed_forward)

        return x


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

    def __init__(self, masked_attention : MultiHeadAttention, cross_attention: MultiHeadAttention, feed_forward: FeedForward, d_model):
        super().__init__()
        self.d_model = d_model
        self.masked_attention = masked_attention
        self.cross_attention = cross_attention
        self.residual_connection = nn.ModuleList([ResidualConnections(self.d_model) for _ in range(3)])
        self.feed_forward = feed_forward

    # KV CACHE: added sa_cache and ca_cache params (None during training)
    def forward(self,x, enc_output, src_mask, tgt_mask, sa_cache=None, ca_cache=None):
        # KV CACHE: pass and receive self-attention cache
        sub_layer, new_sa_cache = self.masked_attention(x,x,x,tgt_mask, kv_cache=sa_cache) 
        x = self.residual_connection[0](x, sub_layer)
        # KV CACHE: pass and receive cross-attention cache
        sub_layer, new_ca_cache = self.cross_attention(x, enc_output, enc_output, src_mask, kv_cache=ca_cache) 
        x = self.residual_connection[1](x, sub_layer)
        sub_layer = self.feed_forward(x)
        x = self.residual_connection[2](x, sub_layer)

        # KV CACHE: return both caches alongside output
        return x, new_sa_cache, new_ca_cache


class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.linear_layer = nn.Linear(d_model,vocab_size)

    def forward(self,x):

        return torch.log_softmax(self.linear_layer(x), dim=-1)
    #The reason is that during training you'd use `nn.NLLLoss` (negative log likelihood) which expects log probabilities as input,
    # and numerically it's more stable than raw softmax followed by log. If you use `nn.CrossEntropyLoss` instead, 
    # skip the softmax entirely because CrossEntropyLoss applies it internally.

# Combined Transformer class
'''Here's my previously written transfromer class below'''
# class Transformer(nn.Module):

#     def __init__(self, embeddings, pe, encoder_blocks: nn.ModuleList, decoder_blocks: nn.ModuleList, projection_layer):
#         super().__init__()
#         self.embeddings = embeddings
#         self.pe = pe
#         self.encoder_blocks = encoder_blocks
#         self.decoder_blocks = decoder_blocks
#         self.projection_layer = projection_layer
    
#     def forward(self,src_data, src_vocab_size, tgt_data, tgt_vocab_size, src_mask, tgt_mask, d_model):
        
#         src_embeddings = Embedding(src_vocab_size, d_model)  # language 1 vocab
#         tgt_embeddings = Embedding(tgt_vocab_size, d_model)  # Language 2 vocab

#         src_pe_embeddings = self.pe(src_embeddings)
#         tgt_pe_embeddings = self.pe(tgt_embeddings)


#         for block in self.encoder_blocks:
#             src_pe_embeddings =  block(src_pe_embeddings, src_mask)
#         encoder_output = src_pe_embeddings

#         for block in self.decoder_blocks:
#             tgt_pe_embeddings = block(tgt_pe_embeddings, encoder_output, src_mask, tgt_mask)
#         decoder_output = tgt_pe_embeddings

#         return self.projection_layer(decoder_output) # we return the logits directly to the loss function

# final transformer class

class Transformer(nn.Module):

    def __init__(self, src_embed: Embedding, tgt_embed: Embedding, src_pe: PositionalEncoding, 
                 tgt_pe : PositionalEncoding, encoder_blocks: nn.ModuleList, decoder_blocks: nn.ModuleList, 
                 projection_layer: ProjectionLayer):
        super().__init__()
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pe = src_pe
        self.tgt_pe = tgt_pe
        self.encoder_blocks = encoder_blocks
        self.decoder_blocks = decoder_blocks
        self.projection_layer = projection_layer
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        src = self.src_pe(self.src_embed(src))
        tgt = self.tgt_pe(self.tgt_embed(tgt))

        for block in self.encoder_blocks:
            src = block(src, src_mask)
        
        for block in self.decoder_blocks:
            # KV CACHE: during training no cache — ignore returned caches
            tgt, _, _ = block(tgt, src, src_mask, tgt_mask)
        
        return self.projection_layer(tgt)
    

# Now we have all these classes built but nothing that assembles them. So, we will make a `build_transformer` Function. It is just a regular 
# Python function that takes configuration parameters like `d_model`, `num_heads`, `num_blocks`, `vocab_size` 
# and creates one instance of every class you've built, wires them together, and returns a ready-to-use Transformer object.

def build_transformer(configurations):
# this function needs to have objects of classes: embeddings, PE, encoder_blocks, decoder_blocks and projection_Layer
    d_model = configurations['d_model']
    num_heads = configurations['num_heads']
    N = configurations['num_blocks']
    src_max_seq_len = configurations['src_max_seq_len']
    tgt_max_seq_len = configurations['tgt_max_seq_len']
    src_vocab_size = configurations['src_vocab_size']
    tgt_vocab_size = configurations['tgt_vocab_size']

    # embeddings 
    src_embed = Embedding(d_model, src_vocab_size)
    tgt_embed = Embedding(d_model, tgt_vocab_size)

    # positional encoding 
    src_pe = PositionalEncoding(d_model, src_max_seq_len)
    tgt_pe = PositionalEncoding(d_model, tgt_max_seq_len)

    # Encoder blocks -> needs a modulelist as parameters
    encoder_block_mdlist = nn.ModuleList([
        Encoder(MultiHeadAttention(d_model, num_heads), FeedForward(d_model), d_model)
    for _ in range(N)] )
    
    decoder_block_mdlist = nn.ModuleList([
        Decoder(MultiHeadAttention(d_model, num_heads, flash_attention=True),MultiHeadAttention(d_model, num_heads, is_cross_attention=True), FeedForward(d_model), d_model)
    for _ in range(N)] )
    
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(src_embed, tgt_embed, src_pe, tgt_pe, 
                              encoder_block_mdlist, decoder_block_mdlist, projection_layer)
    
    return transformer