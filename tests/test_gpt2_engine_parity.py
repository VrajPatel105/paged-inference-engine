import json

from gpt2.load_weights import load_gpt2_into_paged
from gpt2.paged_run import paged_generate

GOLDEN = "gpt2/golden_tokens.json"


def first_diff(a, b):
    return next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), None)


def test_stage2_engine_matches_hf():
    model = load_gpt2_into_paged(num_kv_blocks=256)
    golden = json.load(open(GOLDEN))
    prompts = [g["prompt_ids"] for g in golden]
    ref = [g["generated_ids"] for g in golden]
    targets = [len(r) for r in ref]

    paged_generate(model, prompts[:1], 2)  # compile / warm up

    summary = {}
    for legacy in (True, False):
        # teacher forced: feed HF's tokens back in, check each step's argmax.
        # Separates per-step correctness from INT8 drift compounding.
        res = paged_generate(model, prompts, targets, forced=ref, legacy_positions=legacy)
        rates = []
        for i, r in enumerate(ref):
            match = sum(int(a == b) for a, b in zip(res["predicted"][i], r))
            rates.append(match / len(r))
            print(f"[legacy={legacy}] prompt {i}: teacher-forced match {match}/{len(r)}, "
                  f"first miss at {first_diff(res['predicted'][i], r)}")
        summary[legacy] = sum(rates) / len(rates)

        # free running: exact token parity with HF greedy
        free = paged_generate(model, prompts, targets, legacy_positions=legacy)
        for i, r in enumerate(ref):
            print(f"[legacy={legacy}] prompt {i}: free-running first divergence at "
                  f"{first_diff(free['generated'][i], r)} (None = identical)")

    print("mean teacher-forced match:", summary)
    assert summary[False] > 0.8, "correct-position run should mostly agree with HF"