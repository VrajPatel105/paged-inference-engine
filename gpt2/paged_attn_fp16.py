import torch
from transformer.model import MultiHeadAttention
from kernels.flash_attention import FlashAttentionFunction


class PagedAttentionFP16(MultiHeadAttention):
    """Same interface as MultiHeadAttention, but the KV pool is fp16 (scales fixed at 1)
    and the cache write is vectorized (no per-token loop, no re-quantization)."""

    def __init__(self, d_model, num_heads):
        super().__init__(d_model, num_heads)
        shape = (self.num_blocks, self.num_heads, self.block_size, self.d_k)
        self.k_cache = torch.zeros(shape, dtype=torch.float16)
        self.v_cache = torch.zeros(shape, dtype=torch.float16)
        self.k_scale = torch.ones(self.num_blocks, self.num_heads)
        self.v_scale = torch.ones(self.num_blocks, self.num_heads)

    def forward(self, q, k, v, block_table, q_len_per_seq, kv_len_per_seq,
                sequence_id, offset, length, position_ids, pos_seq_id):
        dev = q.device
        total = q.size(0)
        num_seq = len(sequence_id)

        q = self.W_q(q).view(total, self.num_heads, self.d_k)
        k = self.W_k(k).view(total, self.num_heads, self.d_k)
        v = self.W_v(v).view(total, self.num_heads, self.d_k)

        # dense block table [num_seq, max_blocks] in the same order as sequence_id
        sid_list = sequence_id.tolist()
        row_of = {s: r for r, s in enumerate(sid_list)}
        lens = [len(block_table[s]) for s in sid_list]
        max_b = max(lens)
        bt = torch.zeros(num_seq, max_b, dtype=torch.int32, device=dev)
        for r, s in enumerate(sid_list):
            bt[r, :lens[r]] = torch.tensor(block_table[s], dtype=torch.int32, device=dev)
        num_blocks_per_seq = torch.tensor(lens, dtype=torch.int32, device=dev)

        # vectorized KV write: token -> (block, slot)
        rows = torch.tensor([row_of[s] for s in pos_seq_id.tolist()], device=dev)
        pos = position_ids.long()
        blocks = bt[rows, pos // self.block_size].long()
        slots = pos % self.block_size
        self.k_cache[blocks, :, slots, :] = k.to(torch.float16)
        self.v_cache[blocks, :, slots, :] = v.to(torch.float16)

        # pad Q to [num_seq, heads, max_q_len, d_k]
        max_q = int(length.max().item())
        Qp = torch.zeros(num_seq, self.num_heads, max_q, self.d_k, device=dev, dtype=q.dtype)
        offs, lns = offset.tolist(), length.tolist()
        for r in range(num_seq):
            Qp[r, :, :lns[r], :] = q[offs[r]:offs[r] + lns[r]].transpose(0, 1)

        O = FlashAttentionFunction.apply(
            Qp, self.k_cache, self.v_cache, q_len_per_seq, bt,
            num_blocks_per_seq, self.block_size, kv_len_per_seq,
            self.k_scale, self.v_scale,
        )

        O_flat = torch.zeros(total, self.num_heads, self.d_k, device=dev, dtype=O.dtype)
        for r in range(num_seq):
            O_flat[offs[r]:offs[r] + lns[r]] = O[r, :, :lns[r], :].transpose(0, 1)

        return self.W_o(O_flat.to(torch.float32).reshape(total, self.d_model))