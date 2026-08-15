import torch
from kernels.flash_attention import flash_attention_forward

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_paged_fa_correctness():
    torch.manual_seed(0)

    batch = 3
    num_heads = 4
    head_dim = 64
    block_size = 16

    # Deliberately different real KV lengths per sequence, including
    # one that's an exact multiple of block_size (no partial last block)
    # and ones that aren't (to exercise the padding mask).
    kv_len_per_seq = torch.tensor([20, 35, 16], device=DEVICE, dtype=torch.int32)

    # In prefill-style testing, q_len == kv_len (every query attends causally
    # over the full prefix). Keeps the reference comparison simple.
    q_len_per_seq = kv_len_per_seq.clone()

    max_q_len = q_len_per_seq.max().item()
    max_kv_len = kv_len_per_seq.max().item()
    max_blocks_per_seq = -(-max_kv_len // block_size)  # ceil div

    # --- Build a physical KV pool with blocks deliberately NOT in sequence order ---
    # Enough total blocks for all sequences plus some spare/unused ones,
    # to prove the kernel follows block_table rather than assuming order.
    num_blocks_needed = sum(-(-l.item() // block_size) for l in kv_len_per_seq)
    num_total_blocks = num_blocks_needed + 5  # a few spare blocks, never referenced

    K_pool = torch.randn(num_total_blocks, block_size, num_heads, head_dim,
                          device=DEVICE, dtype=torch.float16)
    V_pool = torch.randn(num_total_blocks, block_size, num_heads, head_dim,
                          device=DEVICE, dtype=torch.float16)

    # Assign each sequence a scrambled (non-contiguous) set of physical blocks
    all_block_ids = torch.randperm(num_total_blocks, device=DEVICE)[:num_blocks_needed]
    block_table = torch.zeros(batch, max_blocks_per_seq, device=DEVICE, dtype=torch.int32)

    cursor = 0
    seq_block_assignments = []  # keep track for building the reference tensor
    for i in range(batch):
        n_blocks = -(-kv_len_per_seq[i].item() // block_size)
        ids = all_block_ids[cursor: cursor + n_blocks]
        block_table[i, :n_blocks] = ids
        seq_block_assignments.append(ids)
        cursor += n_blocks

    num_blocks_per_seq = torch.tensor(
        [-(-l.item() // block_size) for l in kv_len_per_seq],
        device=DEVICE, dtype=torch.int32
    )

    # --- Q: padded to max_q_len, real data only in the first q_len_per_seq[i] rows ---
    Q = torch.randn(batch, num_heads, max_q_len, head_dim, device=DEVICE, dtype=torch.float16)

    # --- Run the paged kernel ---
    O_paged, _ = flash_attention_forward(
        Q, K_pool, V_pool,
        q_len_per_seq, block_table, num_blocks_per_seq, block_size, kv_len_per_seq
    )

    # --- Build the reference: reconstruct each sequence's true contiguous K/V
    #     by gathering the same physical blocks the kernel used, in block_table order ---
    O_ref = torch.zeros_like(O_paged)
    for i in range(batch):
        qlen = q_len_per_seq[i].item()
        kvlen = kv_len_per_seq[i].item()

        ids = seq_block_assignments[i]
        K_seq = K_pool[ids].reshape(-1, num_heads, head_dim)[:kvlen]   # [kvlen, heads, dim]
        V_seq = V_pool[ids].reshape(-1, num_heads, head_dim)[:kvlen]

        K_seq = K_seq.permute(1, 0, 2).unsqueeze(0)   # [1, heads, kvlen, dim]
        V_seq = V_seq.permute(1, 0, 2).unsqueeze(0)
        Q_seq = Q[i:i+1, :, :qlen, :]                  # [1, heads, qlen, dim]

        out = torch.nn.functional.scaled_dot_product_attention(
            Q_seq, K_seq, V_seq, is_causal=True
        )
        O_ref[i, :, :qlen, :] = out[0]

    # Compare only the real (non-padded) query positions per sequence
    results = []
    for i in range(batch):
        qlen = q_len_per_seq[i].item()
        diff = (O_paged[i, :, :qlen, :] - O_ref[i, :, :qlen, :]).abs()
        print(f"seq {i}: max abs diff = {diff.max().item():.5f}, mean = {diff.mean().item():.5f}")
        results.append(torch.allclose(O_paged[i, :, :qlen, :], O_ref[i, :, :qlen, :], atol=1e-2, rtol=1e-3))

    for i, ok in enumerate(results):
        assert ok, f"Mismatch on sequence {i}"

    print("All sequences match — paged FA-2 forward is correct.")


if __name__ == "__main__":
    test_paged_fa_correctness()