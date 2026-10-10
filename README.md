# Paged Inference Engine

A mini inference server built from scratch: paged KV cache, continuous batching, and a custom FlashAttention-2 kernel in Triton, all written without looking at vLLM's source.

The transformer underneath is my own decoder-only build, adapted from an earlier project: [cpp-gpu-inference/en-de-transformer](https://github.com/VrajPatel105/cpp-gpu-inference/tree/main/en-de-transformer). I built this to understand how LLM serving works under the hood, not just how to train a model. The design follows the PagedAttention paper.

A serving layer for this engine (FastAPI, Docker, CI/CD to AWS ECR) lives in [paged-engine-serving](https://github.com/VrajPatel105/paged-engine-serving).

## What it does

- **Paged KV cache.** Keys and values live in fixed-size blocks instead of one contiguous buffer per sequence, so many requests can share GPU memory without fragmentation.
- **Continuous batching scheduler.** Requests are admitted roughly in order, but the scheduler can look a few requests ahead and admit whichever ones fit. If too many requests get skipped in a row, it falls back to blocking mode until memory frees up. New requests join a batch that is already running instead of waiting for it to finish.
- **FlashAttention-2 kernel (Triton).** Reads K and V straight from the paged block cache through a block table.
- **INT8 KV cache option.** The engine can store keys and values in int8 (symmetric, one scale per block per head) instead of fp16, which roughly halves KV memory. I wrote this before the kernel fixes described below and have not re-validated it since, so treat it as unverified. All GPT-2 results below use the fp16 KV path.

## Correctness

The first version of the kernel was tested against PyTorch `scaled_dot_product_attention` on three sequences of different lengths stored in scrambled blocks. That test only covered prefill, where the query length equals the key length.

Running a real model exposed two bugs it missed:

- **Stride order in the kernel launcher.** After I changed the cache layout, the launcher still passed the head stride and the slot stride in the old order. Head 0 read the right memory and every other head read the wrong memory. Prefill logits were badly wrong, even for a one token prompt.
- **Causal mask during decode.** The mask compared key positions against the query's index inside the padded query tensor. For a single decode token that index is 0, so the token could only attend to the first key. The mask now uses the query's real position, `kv_len - q_len + offset`.

To catch this kind of thing, the repo now has a GPT-2 small parity harness. It loads the HuggingFace weights into the paged model and compares against HF:

- `tests/test_gpt2_parity.py` checks last-position logits for single-sequence prefill.
- `tests/test_gpt2_engine_parity.py` runs five prompts through the paged scheduler and compares generation with HF greedy decoding.
- `gpt2/diagnose.py` prints per-layer hidden-state error, which is how the stride bug was found.

After the fixes, prefill matches HF to within about 0.005 in log-probability at prompt lengths from 1 to 21 tokens (fp32 weights), and the relative error of each layer's hidden state stays around 1e-4.

## Benchmarks

Hardware: RTX 5080 Laptop (16 GB), WSL2. PyTorch 2.11.0 (CUDA 12.8 build), Triton 3.6.0. Model: GPT-2 small (124M), fp32 weights, fp16 KV pool. Every request generates exactly 64 new tokens. Numbers are the median of 3 runs for the HF loops and my engine, and the median of 5 for vLLM. All systems get the same pre-tokenized prompts.

Throughput in tokens per second:

| Concurrent requests | HF, no cache | HF, KV cache | Paged (this repo) | vLLM 0.21.0 |
|---|---|---|---|---|
| 1 | 253 | 297 | 82 | 949 |
| 4 | 255 | 294 | 297 | 3,663 |
| 8 | 252 | 297 | 493 | 6,667 |
| 16 | 248 | 295 | 753 | 13,058 |
| 32 | 251 | 295 | 1,030 | 23,247 |

The two HF baselines handle requests one at a time, so part of my engine's lead over them is just batching. vLLM batches too, which makes it the fair comparison.

What the numbers say:

- My engine matches HF with a KV cache at 4 concurrent requests and reaches 3.5x its throughput at 32. It is slower at a single request.
- vLLM is about 12x faster than my engine at 1 request and about 23x faster at 32.
- Dividing wall time by the 64 steps gives roughly 12 ms per step for my engine at 1 request and 31 ms at 32. vLLM takes about 1 ms at 1 request and 1.4 ms at 32. So my engine has a large fixed cost per step plus a cost of about 0.6 ms for each extra sequence, and vLLM has almost neither.
- Scheduling, batch assembly and the final `.tolist()` account for only 1 to 2 percent of step time, so the cost is inside the model call. My guess is the per-layer host work in the attention wrapper (rebuilding the block table, padding and unpadding the query in Python, a few GPU syncs per layer). I have not profiled it, so that is a hypothesis.

Things that make the vLLM comparison imperfect: vLLM runs fp16 weights while mine are fp32, and it uses CUDA graphs, fused kernels and asynchronous scheduling. It also ran in a separate environment (vLLM 0.21.0 built for CUDA 12.9, FlashAttention 2 backend, FlashInfer sampler disabled, `pin_memory=False` because of WSL). Same GPU, same prompts, same token counts.

The first benchmark in this repo used a small model I trained myself, and showed the paged engine below the naive loop at every concurrency level. That sweep ran before the two kernel fixes above, so I have removed it rather than keep numbers from a broken kernel.

I also tried to measure a memory ceiling, but under WSL2 the reported GPU memory was physically impossible for this card, so that measurement is inconclusive and not included.

## Limitations

- Greedy decoding only. No speculative decoding or beam search.
- No CPU offloading or recomputation when memory runs out. The scheduler waits.
- The INT8 KV path has not been re-tested since the kernel fixes.
- The live demo and the standalone runner (`core/model_runner.py`) predate those fixes and have not been re-validated. The GPT-2 harness uses its own runner in `gpt2/paged_run.py`.
- Per-step host overhead is the main thing holding throughput back (see above).

## Next

Compute the block table and batch metadata once per step instead of once per layer, vectorize the query padding, and try CUDA graphs. Then re-test the INT8 KV path against the same GPT-2 harness, and only after that look at CPU offload.

## Running it

```bash
python -m gpt2.reference              # save HF reference outputs (run once)
pytest -s tests/test_gpt2_parity.py
pytest -s tests/test_gpt2_engine_parity.py
python -m bench.benchmark_gpt2        # HF vs paged
python bench/benchmark_vllm.py        # run in a separate vLLM environment
```

Run everything from the repo root. vLLM needs its own environment because it pins a different PyTorch build.