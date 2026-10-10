import json
import statistics
import time

from vllm import LLM, SamplingParams

GOLDEN = json.load(open("gpt2/golden_tokens.json"))
BASE = [g["prompt_ids"] for g in GOLDEN]
MAX_NEW = 64
LEVELS = [1, 4, 8, 16, 32]
REPEATS = 5


def main():
    llm = LLM("openai-community/gpt2", dtype="float16", gpu_memory_utilization=0.5)
    sp = SamplingParams(temperature=0, max_tokens=MAX_NEW, ignore_eos=True)

    def run(n):
        batch = [{"prompt_token_ids": BASE[i % len(BASE)]} for i in range(n)]
        t0 = time.perf_counter()
        outs = llm.generate(batch, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        return sum(len(o.outputs[0].token_ids) for o in outs), dt

    results = {}
    for n in LEVELS:
        run(n)  # warmup for this batch shape (JIT, graphs)
        runs = sorted((run(n) for _ in range(REPEATS)), key=lambda r: r[1])
        toks, dt = runs[len(runs) // 2]
        results[n] = {"tokens": toks, "seconds": dt, "tok_per_s": toks / dt}
        print(f"n={n:3d}  {toks / dt:9.1f} tok/s  ({dt * 1e3:.0f} ms, {toks} tokens, median of {REPEATS})")

    json.dump(results, open("bench/results_vllm.json", "w"), indent=2)


if __name__ == "__main__":
    main()