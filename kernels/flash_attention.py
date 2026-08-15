import torch
import triton
import triton.language as tl
import math

# device 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=8, num_stages=2),
    ],
    key=['seq_len', 'head_dim'],
)
@triton.jit
def flash_attention_kernel(
    Q, K, V, O, L,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    seq_len,
    head_dim : tl.constexpr, 
    BLOCK_M : tl.constexpr,
    BLOCK_N : tl.constexpr
):
    pid_batch = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)
    pid_m = tl.program_id(axis=2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M) 
    offs_d = tl.arange(0, head_dim)
    q_ptrs = Q + pid_batch * stride_qb + pid_head * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd 
    q_mask = offs_m[:, None] < seq_len
    # finally loading q into mem
    q_tile = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # now initializing the running states
    m = tl.full((BLOCK_M,), value=-float('inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    o = tl.zeros((BLOCK_M, head_dim), dtype=tl.float32)

    for j in range(tl.cdiv(seq_len, BLOCK_N)):
        offs_n = j * BLOCK_N + tl.arange(0, BLOCK_N)
        k_ptrs = K + pid_batch * stride_kb + pid_head * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd 
        v_ptrs = V + pid_batch * stride_vb + pid_head * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd 

        k_mask = offs_n[:, None] < seq_len
        v_mask = offs_n[:, None] < seq_len

        k_tile = tl.load(k_ptrs, mask=k_mask, other=0.0)
        v_tile = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # compute S with the 1/root(d) scailing factor 
        num = tl.dot(q_tile, tl.trans(k_tile))
        # S = num / tl.sqrt(head_dim.to(tl.float32))
        S = num / (head_dim ** 0.5) # replaced the above one with this new one for precision issues
        S = S.to(tl.float32) # trying to force it to produce fp32 

        # S = tl.where(offs_n[None, :] < seq_len, S, float('-inf')) # by this we are replacing the mask values from 0 to -inf cuz there shoujdl be no involvement of 0
        # now we replace the above with padding + causal mask together 

        padding_ok = offs_n[None, :] < seq_len
        causal_ok = offs_m[:, None] >= offs_n[None, :]
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
        new_c = tl.dot(tilde_p, v_tile.to(tl.float32)) # casting to fp32
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
    o_mask = offs_m[:, None] < seq_len
    tl.store(o_ptrs, final_o, mask=o_mask)

    l_ptrs = L + pid_batch * stride_lb + pid_head * stride_lh + offs_m * stride_ls
    l_mask = offs_m < seq_len
    tl.store(l_ptrs, final_l, mask=l_mask)


def flash_attention_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):

    # extracting the size for q k v tensors
    batch, num_heads, seq_len, head_dim = Q.shape
    assert K.shape == Q.shape and V.shape == Q.shape, "Q, K, V shape mismatch"

    assert Q.is_cuda and K.is_cuda and V.is_cuda , "Not CUDA Tensors -_-"

    # output tensor
    O = torch.empty(batch, num_heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)
    L = torch.empty(batch, num_heads, seq_len, device=DEVICE, dtype=torch.float16)

    # define the launchpad grid
    grid = lambda META: (batch, num_heads, triton.cdiv(seq_len, META['BLOCK_M']))

    flash_attention_kernel[grid](
        Q, K, V, O, L,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        seq_len, head_dim,
    )

    return O, L


def run_fa_fwd(batch, heads, seq_len, head_dim):

    torch.manual_seed(42)

    Q = torch.rand(batch, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)
    K = torch.rand(batch, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)
    V = torch.rand(batch, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)

    output_flash_attention_o, _ = flash_attention_forward(Q, K, V)

    # causal reference — PyTorch's built-in, is_causal=True applies the
    # same "key position <= query position" rule you just added
    output_torch = torch.nn.functional.scaled_dot_product_attention(
        Q, K, V, is_causal=True
    )

    diff = (output_flash_attention_o - output_torch).abs()
    print("max abs diff:", diff.max().item())
    print("mean abs diff:", diff.mean().item())

    result = torch.allclose(output_flash_attention_o, output_torch, atol=1e-2, rtol=1e-3)
    return result


class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V):
        O, L = flash_attention_forward(Q, K, V)
        ctx.save_for_backward(Q, K, V, O, L)
        return O