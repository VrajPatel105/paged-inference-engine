import json
import statistics
import time

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from gpt2.load_weights import load_gpt2_into_paged
from gpt2.paged_run import paged_generate

HF = "openai-community/gpt2"
NEW_TOKENS = 64
LEVELS = [1, 4, 8, 16, 32]
REPEATS = 3
DEV = "cuda"


@torch.no_grad()
def hf_nocache(model, prompt_ids, n_new):
    ids = torch.tensor([prompt_ids], device=DEV)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ts = []
    for _ in range(n_new):
        nxt = model(ids).logits[:, -1].argmax(-1)
        nxt.item()
        ts.append(time.perf_counter())
        ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
    return t0, ts


@torch.no_grad()
def hf_cached(model, prompt_ids, n_new):
    ids = torch.tensor([prompt_ids], device=DEV)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ts, past = [], None
    for _ in range(n_new):
        out = model(input_ids=ids, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1)
        nxt.item()
        ts.append(time.perf_counter())
        ids = nxt.view(1, 1)
    return t0, ts


def run_sequential(fn, model, prompts):
    start = time.perf_counter()
    ttfts, tpots = [], []
    for p in prompts:
        t0, ts = fn(model, p, NEW_TOKENS)
        gaps = [b - a for a, b in zip(ts, ts[1:])]
        ttfts.append(ts[0] - t0)
        tpots.append(sum(gaps) / len(gaps))
    wall = time.perf_counter() - start
    return {"wall": wall, "tokens": len(prompts) * NEW_TOKENS,
            "ttft": statistics.mean(ttfts), "tpot": statistics.mean(tpots)}


def run_paged(model, prompts):
    r = paged_generate(model, prompts, NEW_TOKENS)
    return {"wall": r["wall"], "tokens": r["tokens"],
            "ttft": statistics.mean(s["ttft"] for s in r["per_seq"]),
            "tpot": statistics.mean(s["tpot"] for s in r["per_seq"]),
            "overhead_pct": 100 * (1 - r["forward_s"] / r["busy_s"])}


def median_run(fn):
    runs = sorted((fn() for _ in range(REPEATS)), key=lambda r: r["wall"])
    r = runs[len(runs) // 2]
    r["tok_s"] = r["tokens"] / r["wall"]
    return r


if __name__ == "__main__":
    tok = GPT2Tokenizer.from_pretrained(HF)
    golden = json.load(open("gpt2/golden_tokens.json"))
    base = [g["prompt_ids"] for g in golden]

    hf_model = GPT2LMHeadModel.from_pretrained(HF).to(DEV).eval()
    paged_model = load_gpt2_into_paged(num_kv_blocks=512)

    # warmup: CUDA init, Triton autotune, allocator
    hf_nocache(hf_model, base[0], 4)
    hf_cached(hf_model, base[0], 4)
    paged_generate(paged_model, base[:2], 4)

    results = {}
    for n in LEVELS:
        prompts = [base[i % len(base)] for i in range(n)]
        results[n] = {
            "hf_nocache": median_run(lambda: run_sequential(hf_nocache, hf_model, prompts)),
            "hf_cached": median_run(lambda: run_sequential(hf_cached, hf_model, prompts)),
            "paged": median_run(lambda: run_paged(paged_model, prompts)),
        }

    print(f"\nGPT-2 small, {NEW_TOKENS} new tokens per request, fp32, median of {REPEATS}\n")
    print(f"{'N':>3} | {'nocache tok/s':>13} | {'cached tok/s':>12} | {'paged tok/s':>11} | {'paged overhead %':>16}")
    for n in LEVELS:
        r = results[n]
        print(f"{n:>3} | {r['hf_nocache']['tok_s']:>13.1f} | {r['hf_cached']['tok_s']:>12.1f} | "
              f"{r['paged']['tok_s']:>11.1f} | {r['paged']['overhead_pct']:>16.1f}")

    r = results[1]
    print("\nN=1 latency (ms)   [sequential systems: isolated; paged: includes scheduling]")
    for name in ("hf_nocache", "hf_cached", "paged"):
        print(f"{name:>11}: TTFT {r[name]['ttft']*1e3:8.1f}   TPOT {r[name]['tpot']*1e3:7.2f}")

    json.dump({str(k): v for k, v in results.items()},
              open("bench/results_gpt2.json", "w"), indent=2)