import torch
import triton
import triton.language as tl
import math

# device 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=2),
    ],
    key=['block_size', 'head_dim'],
)
@triton.jit
def flash_attention_kernel(
    Q, K, V, O, L,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_k_block, stride_kh, stride_ks, stride_kd,
    stride_v_block, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    block_table, num_blocks_per_seq,
    q_len_per_seq,
    stride_bt_batch, stride_bt_col,
    kv_len_per_seq,
    k_scale, v_scale,
    stride_k_b_scale, stride_k_h_scale,
    stride_v_b_scale, stride_v_h_scale,
    block_size : tl.constexpr,
    head_dim : tl.constexpr, 
    BLOCK_M : tl.constexpr,
):
    pid_batch = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)
    pid_m = tl.program_id(axis=2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M) 
    offs_d = tl.arange(0, head_dim)
    q_ptrs = Q + pid_batch * stride_qb + pid_head * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd 

    q_len_ptr = q_len_per_seq + pid_batch
    q_len_val = tl.load(q_len_ptr)
    q_mask = offs_m[:, None] < q_len_val
    q_tile = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # now initializing the running states
    m = tl.full((BLOCK_M,), value=-float('inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    o = tl.zeros((BLOCK_M, head_dim), dtype=tl.float32)

    num_blocks_val = tl.load(num_blocks_per_seq + pid_batch)

    for j in range(num_blocks_val):

        bt_ptr = block_table + pid_batch * stride_bt_batch + j * stride_bt_col
        block_number = tl.load(bt_ptr)
        offs_n = tl.arange(0, block_size)
        k_ptrs = K + block_number * stride_k_block + pid_head * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd 
        v_ptrs = V + block_number * stride_v_block + pid_head * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd 

        kv_len_val = tl.load(kv_len_per_seq + pid_batch)
        remaining_for_this_block = kv_len_val - j * block_size

        k_mask = offs_n[:, None] < remaining_for_this_block
        v_mask = offs_n[:, None] < remaining_for_this_block

        # build the k_scale and v_scale tiles and load it. 
        k_scale_ptrs = k_scale + block_number * stride_k_b_scale + pid_head * stride_k_h_scale # this is just a scalar value ptr now
        k_scale_tile = tl.load(k_scale_ptrs)

        v_scale_ptrs = v_scale + block_number * stride_v_b_scale + pid_head * stride_v_h_scale
        v_scale_tile = tl.load(v_scale_ptrs)

        k_tile = tl.load(k_ptrs, mask=k_mask, other=0.0)
        v_tile = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # now we can dequantize the k and v tiles back before sending it to dot
        dequantized_k_tile = k_tile.to(tl.float32) * k_scale_tile
        dequantized_v_tile = v_tile.to(tl.float32) * v_scale_tile


        # compute S with the 1/root(d) scailing factor 
        num = tl.dot(q_tile, tl.trans(dequantized_k_tile))
        # S = num / tl.sqrt(head_dim.to(tl.float32))
        S = num / (head_dim ** 0.5) # replaced the above one with this new one for precision issues
        S = S.to(tl.float32) # trying to force it to produce fp32 

        # S = tl.where(offs_n[None, :] < seq_len, S, float('-inf')) # by this we are replacing the mask values from 0 to -inf cuz there shoujdl be no involvement of 0
        # now we replace the above with padding + causal mask together 

        padding_ok = offs_n[None, :] < remaining_for_this_block
        causal_ok = offs_m[:, None] >= j * block_size + offs_n
        mask = padding_ok & causal_ok
        S = tl.where(mask, S, float('-inf'))
        # compute the running vars 
        # 1. rowmax 
        rowmax = tl.max(S, axis=1)
        # 2. max
        m_new = tl.maximum(m, rowmax)
        # 3. rowsum
        tilde_p = tl.exp(S - m_new[:, None])
        rowsum = tl.sum(tilde_p, axis=1) 
        l_new = tl.exp(m - m_new) * l + rowsum

        # now o
        rescale_factor = tl.exp(m - m_new)
        old_c = rescale_factor[:, None] * o # this broadcasts the per row scalar across head_dim
        new_c = tl.dot(tilde_p, dequantized_v_tile)
        o_new = old_c + new_c

        # update the new values to the running states (vars)
        m = m_new
        l = l_new
        o = o_new

    # now out of the loop
    final_o = o / l[:, None]
    final_l = m + tl.log(l)

    # finally write the computed values back to hbm
    o = final_o
    l = final_l

    # finally write o back to output 
    o_ptrs = O + pid_batch * stride_ob + pid_head * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od 
    o_mask = offs_m[:, None] < q_len_val 
    tl.store(o_ptrs, final_o, mask=o_mask)

    l_ptrs = L + pid_batch * stride_lb + pid_head * stride_lh + offs_m * stride_ls
    l_mask = offs_m < q_len_val
    tl.store(l_ptrs, final_l, mask=l_mask)


def flash_attention_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, q_len_per_seq, block_table, num_blocks_per_seq, block_size, kv_len_per_seq, k_scale_ptr, v_scale_ptr):

    # extracting the size for q k v tensors
    batch, num_heads, o_l_seq_len_helper, head_dim = Q.shape

    # q_len_per_seq : each entry is that sequence's real query length this step" (no mention of seq_id -> it's indexed by batch position, not by an ID)
    
    assert Q.is_cuda and K.is_cuda and V.is_cuda , "Not CUDA Tensors -_-"

    # output tensor
    O = torch.empty(batch, num_heads, o_l_seq_len_helper, head_dim, device=DEVICE, dtype=torch.float16)
    L = torch.empty(batch, num_heads, o_l_seq_len_helper, device=DEVICE, dtype=torch.float16)

    # define the launchpad grid
    grid = lambda META: (batch, num_heads, triton.cdiv(q_len_per_seq.max().item(), META['BLOCK_M']))

    flash_attention_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(2), K.stride(1), K.stride(3),
            V.stride(0), V.stride(2), V.stride(1), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            block_table, num_blocks_per_seq,
            q_len_per_seq, 
            block_table.stride(0), block_table.stride(1), 
            kv_len_per_seq,
            k_scale_ptr, v_scale_ptr,
            k_scale_ptr.stride(0), k_scale_ptr.stride(1),
            v_scale_ptr.stride(0), v_scale_ptr.stride(1),
            block_size, head_dim
        )
    
    return O, L


class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, q_len_per_seq, block_table, num_blocks_per_seq, block_size, kv_len_per_seq, k_scale_ptr, v_scale_ptr):
        O, L = flash_attention_forward(Q, K, V, q_len_per_seq, block_table, num_blocks_per_seq, block_size, kv_len_per_seq, k_scale_ptr, v_scale_ptr)
        ctx.save_for_backward(Q, K, V, O, L)
        return O