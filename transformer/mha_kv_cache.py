"""
This is just a code file for learning purpose.
This file is not used anywhere in the entire project.
The only purpose of this project is to write the mha only with kv cache and not with flash attention or kv quantization.
This file has mha class with naive implementation of kv cache and mha
"""

import torch
import torch.nn as nn
import math

class NaiveMHA:


    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # self.register_buffer('k_cache', torch.zeroes(self.))



    @staticmethod
    def attention(q, k, v, d_k, mask=None):
        # (B, heads, seq, d_k) x (B, heads, d_k, seq) → (B, heads, seq, seq)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        scores = torch.softmax(scores, dim=-1)
        return scores @ v    # → (B, heads, seq, d_k)

    def forward(self, q, k, v, mask=None):
        B, q_len, _ = q.shape
        k_len = k.size(1)

        q = self.W_q(q)    # (2,4,4) @ W_q.T  →  (2,4,4)
        k = self.W_k(k)    # (2,4,4) @ W_k.T  →  (2,4,4)
        v = self.W_v(v)    # (2,4,4) @ W_v.T  →  (2,4,4)

        # Split into heads: (B, seq, d_model) → (B, heads, seq, d_k)
        q = q.view(B, q_len, self.num_heads, self.d_k).transpose(1, 2)  # (2,2,4,2)
        k = k.view(B, k_len, self.num_heads, self.d_k).transpose(1, 2)  # (2,2,4,2)
        v = v.view(B, k_len, self.num_heads, self.d_k).transpose(1, 2)  # (2,2,4,2)

        x = self.attention(q, k, v, self.d_k, mask)                     # (2,2,4,2)

        # Concat heads: (2,2,4,2) → transpose → (2,4,2,2) → view → (2,4,4)
        x = x.transpose(1, 2).contiguous().view(B, q_len, self.d_model)

        return self.W_o(x)    # (2,4,4) @ W_o.T  →  (2,4,4)