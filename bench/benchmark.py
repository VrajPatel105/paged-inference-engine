


"""
Please note that benchmark was mostly done by claude since there was nothing to learn here, 
it was just gather all the pieces and connecting them once again so i had claude do it  :) 
"""




"""
Benchmark harness comparing naive (no cache, no batching) generation against
the paged engine (block cache + continuous batching + INT8 KV quantization).
Standalone module — does not modify train.py or model_runner.py.
"""

import time
import pickle
import torch
from transformer.config import transformer_configurations
from transformer.load_checkpoint import load_trained_weights
from transformer.train import build_train_transformer
from core.block_manager import BlockManager
from core.scheduler import Scheduler
from core.sequence import Sequence
from core.config import core_configurations


def naive_generate_with_timing(model, prompt, tok, device, max_len):
    model.eval()
    with torch.no_grad():
        tokens = tok.encode_sentence(prompt, add_sos=True, add_eos=False)
        generated_ids = list(tokens)

        input_ids = torch.tensor([generated_ids], dtype=torch.long, device=device)
        torch.cuda.synchronize()
        start = time.time()
        logits = model(input_ids)
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()
        torch.cuda.synchronize()
        ttft = time.time() - start
        generated_ids.append(next_token_id)

        decode_times = []
        for _ in range(max_len - 1):
            if next_token_id == tok.EOS_ID:
                break

            input_ids = torch.tensor([generated_ids], dtype=torch.long, device=device)
            torch.cuda.synchronize()
            start = time.time()
            logits = model(input_ids)
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()
            torch.cuda.synchronize()
            decode_times.append(time.time() - start)

            generated_ids.append(next_token_id)

        tpot = sum(decode_times) / len(decode_times) if decode_times else 0.0

        return {
            "text": tok.decode_sentence(generated_ids[1:]),
            "ttft": ttft,
            "tpot": tpot,
            "num_tokens": len(generated_ids),
        }

def paged_generate_with_timing(model, prompts, tok, max_steps=None):
    """
    prompts: list of already-tokenized prompt_token_ids lists, submitted
    concurrently, so they compete for scheduling the same way N simultaneous
    requests would.

    max_steps: if set, raises RuntimeError once schedule() has been called
    this many times without every sequence finishing. Needed because the
    scheduler blocks/waits rather than erroring when it runs out of blocks
    (v1 design), so an under-provisioned request count would otherwise spin
    forever instead of failing loudly.

    Returns per-sequence TTFT/TPOT plus total wall-clock time for the batch.
    """
    block_manager_obj = BlockManager(core_configurations['num_blocks'], core_configurations['block_size'])
    scheduler_obj = Scheduler(
        block_manager_obj,
        core_configurations['block_size'],
        core_configurations['max_len'],
        skip_threshold=core_configurations['scheduler_skip_threshold'],
        lookahead=core_configurations['scheduler_lookahead'],
    )

    submit_time = {}
    token_times = {}

    batch_start = time.time()
    for seq_id, prompt_token_ids in enumerate(prompts):
        scheduler_obj.add_request(Sequence(seq_id=seq_id, prompt_token_ids=prompt_token_ids))
        submit_time[seq_id] = batch_start
        token_times[seq_id] = []

    num_finished = 0
    total_sequences = len(prompts)
    steps = 0

    while num_finished < total_sequences:
        if max_steps is not None and steps >= max_steps:
            raise RuntimeError(f"no progress after {max_steps} schedule() calls — likely out of blocks for {total_sequences} concurrent sequences")
        steps += 1

        output = scheduler_obj.schedule()
        prefill_seq = output.prefill_seqs
        decode_seq = output.decode_seqs

        if not prefill_seq and not decode_seq:
            continue

        sequence_id, length, offset, position_ids, pos_seq_id, kv_len_per_seq = [], [], [], [], [], []
        flat_tokens = []
        offset_cnt = 0

        for seq in prefill_seq:
            num_new_token = len(seq.prompt_token_ids)
            sequence_id.append(seq.seq_id)
            length.append(num_new_token)
            offset.append(offset_cnt)
            offset_cnt += num_new_token
            pos_seq_id.extend([seq.seq_id] * num_new_token)
            position_ids.extend(range(len(seq.token_ids)))
            kv_len_per_seq.append(num_new_token)
            flat_tokens.extend(seq.prompt_token_ids)

        for seq in decode_seq:
            sequence_id.append(seq.seq_id)
            length.append(1)
            offset.append(offset_cnt)
            offset_cnt += 1
            pos_seq_id.append(seq.seq_id)
            position_ids.append(len(seq.token_ids))
            kv_len_per_seq.append(len(seq.token_ids) + 1)
            flat_tokens.append(seq.token_ids[-1])

        sequence_id = torch.tensor(sequence_id, dtype=torch.int32, device='cuda')
        length = torch.tensor(length, dtype=torch.int32, device='cuda')
        offset = torch.tensor(offset, dtype=torch.int32, device='cuda')
        position_ids = torch.tensor(position_ids, dtype=torch.int32, device='cuda')
        pos_seq_id = torch.tensor(pos_seq_id, dtype=torch.int32, device='cuda')
        flat_tokens = torch.tensor(flat_tokens, dtype=torch.int32, device='cuda')
        kv_len_per_seq = torch.tensor(kv_len_per_seq, dtype=torch.int32, device='cuda')

        torch.cuda.synchronize()
        logits = model(flat_tokens, block_manager_obj.block_table, length, kv_len_per_seq,
                        sequence_id, offset, length, position_ids, pos_seq_id)
        torch.cuda.synchronize()
        step_time = time.time()

        last_indices = offset + length - 1
        new_tokens = torch.argmax(logits[last_indices], dim=-1).tolist()

        current_index = 0
        for seq in prefill_seq + decode_seq:
            if new_tokens[current_index] == tok.EOS_ID:
                seq.is_finished = True
                num_finished += 1
            else:
                seq.token_ids.append(new_tokens[current_index])
                token_times[seq.seq_id].append(step_time)
                if len(seq.token_ids) >= core_configurations['max_len']:
                    num_finished += 1
            current_index += 1

    total_wall_time = time.time() - batch_start

    results = {}
    for seq_id in submit_time:
        times = token_times[seq_id]
        if not times:
            continue
        ttft = times[0] - submit_time[seq_id]
        gaps = [times[i] - times[i - 1] for i in range(1, len(times))]
        tpot = sum(gaps) / len(gaps) if gaps else 0.0
        results[seq_id] = {"ttft": ttft, "tpot": tpot, "num_tokens": len(times)}

    total_tokens = sum(r["num_tokens"] for r in results.values())
    return {
        "per_sequence": results,
        "total_wall_time": total_wall_time,
        "aggregate_throughput": total_tokens / total_wall_time if total_wall_time > 0 else 0.0,
    }

def warmup_naive(model, tok, device):
    naive_generate_with_timing(model, "warmup", tok, device, max_len=10)


def warmup_paged(model, tok):
    warmup_prompt = tok.encode_sentence("warmup", add_sos=True, add_eos=False)
    paged_generate_with_timing(model, [warmup_prompt], tok)


def run_throughput_sweep(naive_model, paged_model, tok, device, base_prompts, concurrency_levels, max_len=50):
    print(f"{'N':>4} | {'naive total (s)':>16} | {'paged total (s)':>16} | {'naive tok/s':>12} | {'paged tok/s':>12}")

    for n in concurrency_levels:
        prompts_text = [base_prompts[i % len(base_prompts)] for i in range(n)]

        naive_start = time.time()
        naive_tokens = 0
        for p in prompts_text:
            result = naive_generate_with_timing(naive_model, p, tok, device, max_len=max_len)
            naive_tokens += result["num_tokens"]
        naive_total = time.time() - naive_start
        naive_throughput = naive_tokens / naive_total

        prompt_token_ids = [tok.encode_sentence(p, add_sos=True, add_eos=False) for p in prompts_text]
        paged_result = paged_generate_with_timing(paged_model, prompt_token_ids, tok)
        paged_total = paged_result["total_wall_time"]
        paged_throughput = paged_result["aggregate_throughput"]

        print(f"{n:>4} | {naive_total:>16.2f} | {paged_total:>16.2f} | {naive_throughput:>12.1f} | {paged_throughput:>12.1f}")

def find_max_concurrent_naive(model, tok, device, prompt, max_len, ceiling=128, step=16):
    n = step
    last_ok = 0
    peak_mem = 0.0
    while n <= ceiling:
        print(f"  naive: trying n={n}...", flush=True)
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            for _ in range(n):
                naive_generate_with_timing(model, prompt, tok, device, max_len=max_len)
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
            last_ok = n
            n += step
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return last_ok, peak_mem
    return last_ok, peak_mem


def find_max_concurrent_paged(model, tok, prompt_text, max_len, ceiling=128, step=16):
    n = step
    last_ok = 0
    peak_mem = 0.0
    while n <= ceiling:
        print(f"  paged: trying n={n}...", flush=True)
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            prompt_token_ids = [tok.encode_sentence(prompt_text, add_sos=True, add_eos=False) for _ in range(n)]
            paged_generate_with_timing(model, prompt_token_ids, tok, max_steps=max_len * 3)
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
            last_ok = n
            n += step
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            return last_ok, peak_mem
    return last_ok, peak_mem


def run_memory_ceiling_test(naive_model, paged_model, tok, device, prompt_text, max_len=50):
    naive_max, naive_mem = find_max_concurrent_naive(naive_model, tok, device, prompt_text, max_len)
    print(f"naive:  max concurrent sequences = {naive_max}, peak GPU memory = {naive_mem:.1f} MB")

    paged_max, paged_mem = find_max_concurrent_paged(paged_model, tok, prompt_text, max_len)
    print(f"paged:  max concurrent sequences = {paged_max}, peak GPU memory = {paged_mem:.1f} MB")


if __name__ == "__main__":
    device = torch.device("cuda")

    with open("transformer/tokenizer.pkl", "rb") as f:
        tok = pickle.load(f)

    base_prompts = ["Once upon a time", "The jungle was", "A king ruled", "hi there"]

    naive_model = build_train_transformer(transformer_configurations, tok.vocab_size())
    naive_model.load_state_dict(torch.load("transformer/decoder_only.pt"))
    naive_model.to(device)

    paged_model = load_trained_weights("transformer/decoder_only.pt")
    paged_model.to(device)

    warmup_naive(naive_model, tok, device)
    warmup_paged(paged_model, tok)

    print("=== Throughput / latency sweep ===")
    run_throughput_sweep(naive_model, paged_model, tok, device, base_prompts, concurrency_levels=[1, 4, 8, 16, 32])

    print("\n=== Memory ceiling test (naive) ===")
    naive_max, naive_mem = find_max_concurrent_naive(naive_model, tok, device, "Once upon a time", max_len=50)
    print(f"naive: max concurrent sequences = {naive_max}, peak GPU memory = {naive_mem:.1f} MB")

    del naive_model
    torch.cuda.empty_cache()

    print("\n=== Memory ceiling test (paged) ===")
    paged_max, paged_mem = find_max_concurrent_paged(paged_model, tok, "Once upon a time", max_len=50)
    print(f"paged: max concurrent sequences = {paged_max}, peak GPU memory = {paged_mem:.1f} MB")

# output : because of wsl, the memory ceiling test for paged was way out of bounds
# (mlenv) vraj@Vraj:/mnt/c/dev/projects/paged-inference-engine$ python -m bench.benchmark

# === Throughput / latency sweep ===
#    N |  naive total (s) |  paged total (s) |  naive tok/s |  paged tok/s
#    1 |             0.11 |             1.00 |        490.9 |         95.1
#    4 |             0.40 |             2.53 |        542.1 |        151.8
#    8 |             0.79 |             4.64 |        550.3 |        165.5
#   16 |             1.59 |             8.88 |        542.1 |        172.9
#   32 |             3.18 |            17.44 |        542.8 |        176.1

# === Memory ceiling test (naive) ===
#   naive: trying n=16...
#   naive: trying n=32...
#   naive: trying n=48...
#   naive: trying n=64...
#   naive: trying n=80...
#   naive: trying n=96...
#   naive: trying n=112...
#   naive: trying n=128...
# naive: max concurrent sequences = 128, peak GPU memory = 9389.7 MB

# === Memory ceiling test (paged) ===
#   paged: trying n=16...
#   paged: trying n=32...
#   paged: trying n=48...
#   paged: trying n=64...
#   paged: trying n=80...
# paged: max concurrent sequences = 64, peak GPU memory = 32146.6 MB