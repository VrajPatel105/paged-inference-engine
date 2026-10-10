import torch
import torch.nn as nn

from core.config import core_configurations
from transformer.model import MultiHeadAttention      # INT8 paged KV (original path)
from gpt2.paged_attn_fp16 import PagedAttentionFP16   # fp16 paged KV, vectorized write


class GPT2FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.act = nn.GELU(approximate="tanh")  # GPT-2's activation
        self.linear2 = nn.Linear(d_model * 4, d_model)

    def forward(self, x):
        return self.linear2(self.act(self.linear1(x)))


class GPT2Block(nn.Module):
    """Pre-norm block: x + attn(ln1(x)), then x + mlp(ln2(x))."""

    def __init__(self, d_model, num_heads, attn_cls=PagedAttentionFP16):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, eps=1e-5)
        self.masked_attention = attn_cls(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-5)
        self.feed_forward = GPT2FeedForward(d_model)

    def forward(self, x, block_table, q_len_per_seq, kv_len_per_seq,
                sequence_id, offset, length, position_ids, pos_seq_id):
        h = self.ln1(x)
        x = x + self.masked_attention(h, h, h, block_table, q_len_per_seq, kv_len_per_seq,
                                      sequence_id, offset, length, position_ids, pos_seq_id)
        x = x + self.feed_forward(self.ln2(x))
        return x


class GPT2Paged(nn.Module):
    def __init__(self, d_model=768, num_heads=12, num_layers=12,
                 vocab_size=50257, max_positions=1024, kv="fp16"):
        super().__init__()
        assert kv in ("fp16", "int8"), "kv must be 'fp16' or 'int8'"
        attn_cls = PagedAttentionFP16 if kv == "fp16" else MultiHeadAttention

        self.wte = nn.Embedding(vocab_size, d_model)        # no sqrt(d_model) scaling
        self.wpe = nn.Embedding(max_positions, d_model)     # learned positions
        self.decoder_blocks = nn.ModuleList(
            [GPT2Block(d_model, num_heads, attn_cls) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model, eps=1e-5)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight               # tied, as in GPT-2

    def forward(self, tgt, block_table, q_len_per_seq, kv_len_per_seq,
                sequence_id, offset, length, position_ids, pos_seq_id):
        x = self.wte(tgt) + self.wpe(position_ids)          # [total_tokens, d_model]
        for block in self.decoder_blocks:
            x = block(x, block_table, q_len_per_seq, kv_len_per_seq,
                      sequence_id, offset, length, position_ids, pos_seq_id)
        return self.lm_head(self.ln_f(x))                   # raw logits [total_tokens, vocab]


def build_gpt2(num_kv_blocks=256, kv="fp16", **kwargs):
    # The attention layers read the KV pool size from core_configurations at construction.
    # fp16 pool: 256 blocks * 12 heads * 64 slots * 64 dims * 2 bytes * 2 (K and V)
    # is about 50 MB per layer, about 600 MB across 12 layers.
    core_configurations["num_blocks"] = num_kv_blocks
    return GPT2Paged(kv=kv, **kwargs)