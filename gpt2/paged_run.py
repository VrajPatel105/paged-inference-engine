import time
import torch

from core.block_manager import BlockManager
from core.scheduler import Scheduler
from core.sequence import Sequence
from core.config import core_configurations

DEV = "cuda"


def _t(xs):
    return torch.tensor(xs, dtype=torch.int32, device=DEV)


@torch.no_grad()
def paged_generate(model, prompts, new_tokens, forced=None,
                   legacy_positions=False, max_steps=200000):
    """
    prompts: list of token-id lists. new_tokens: int or per-prompt list.
    forced: optional per-prompt list of token ids to feed back instead of the
            model's own argmax (teacher forcing, isolates per-step correctness).
    legacy_positions: reproduce model_runner.py's decode position convention.
    EOS is ignored: every sequence generates exactly its target length.
    """
    n = len(prompts)
    targets = [new_tokens] * n if isinstance(new_tokens, int) else list(new_tokens)
    block_size = core_configurations["block_size"]

    bm = BlockManager(core_configurations["num_blocks"], block_size)
    sched = Scheduler(
        bm, block_size, max(len(p) + t for p, t in zip(prompts, targets)) + 2,
        skip_threshold=core_configurations["scheduler_skip_threshold"],
        lookahead=core_configurations["scheduler_lookahead"],
    )

    start = time.perf_counter()
    for i, p in enumerate(prompts):
        sched.add_request(Sequence(seq_id=i, prompt_token_ids=list(p),
                                   max_token_to_generate=targets[i]))

    gen = [[] for _ in range(n)]
    preds = [[] for _ in range(n)]
    times = [[] for _ in range(n)]
    done = steps = 0
    t_forward = t_busy = 0.0

    while done < n:
        steps += 1
        if steps > max_steps:
            raise RuntimeError("no progress: likely out of KV blocks for this concurrency")

        t0 = time.perf_counter()
        out = sched.schedule()
        prefill, decode = out.prefill_seqs, out.decode_seqs
        if not prefill and not decode:
            continue

        seq_ids, length, offset, pos_ids, pos_seq, kv_len, flat = [], [], [], [], [], [], []
        cnt = 0
        for seq in prefill:
            m = len(seq.prompt_token_ids)
            seq_ids.append(seq.seq_id); length.append(m); offset.append(cnt); cnt += m
            pos_seq.extend([seq.seq_id] * m)
            pos_ids.extend(range(m))
            kv_len.append(m)
            flat.extend(seq.prompt_token_ids)
        for seq in decode:
            L = len(seq.token_ids)
            seq_ids.append(seq.seq_id); length.append(1); offset.append(cnt); cnt += 1
            pos_seq.append(seq.seq_id)
            pos_ids.append(L if legacy_positions else L - 1)
            kv_len.append(L + 1 if legacy_positions else L)
            flat.append(seq.token_ids[-1])

        sid, ln, off = _t(seq_ids), _t(length), _t(offset)
        torch.cuda.synchronize()
        tf0 = time.perf_counter()
        logits = model(_t(flat), bm.block_table, ln, _t(kv_len), sid, off, ln,
                       _t(pos_ids), _t(pos_seq))
        torch.cuda.synchronize()
        tf1 = time.perf_counter()
        pred = torch.argmax(logits[off + ln - 1], dim=-1).tolist()
        t_step = time.perf_counter()

        t_forward += tf1 - tf0
        t_busy += t_step - t0

        for k, seq in enumerate(prefill + decode):
            i = seq.seq_id
            step = len(gen[i])
            preds[i].append(pred[k])
            tok = forced[i][step] if forced is not None else pred[k]
            gen[i].append(tok)
            seq.token_ids.append(tok)
            times[i].append(t_step)
            if len(gen[i]) >= targets[i]:
                seq.is_finished = True
                done += 1

    wall = time.perf_counter() - start
    per_seq = []
    for i in range(n):
        ts = times[i]
        gaps = [b - a for a, b in zip(ts, ts[1:])]
        per_seq.append({"ttft": ts[0] - start,
                        "tpot": sum(gaps) / len(gaps) if gaps else 0.0})
    return {"generated": gen, "predicted": preds, "per_seq": per_seq, "wall": wall,
            "tokens": sum(len(g) for g in gen), "forward_s": t_forward, "busy_s": t_busy}