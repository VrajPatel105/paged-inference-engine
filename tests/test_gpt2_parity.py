import json
import torch
import torch.nn.functional as F

from gpt2.load_weights import load_gpt2_into_paged

GOLDEN_TOKENS = "gpt2/golden_tokens.json"
GOLDEN_LOGITS = "gpt2/golden_last_logits.pt"
BLOCK_SIZE = 64


def paged_last_logits(model, prompt_ids):
    dev = "cuda"
    T = len(prompt_ids)
    n_blocks = (T + BLOCK_SIZE - 1) // BLOCK_SIZE
    kw = dict(device=dev)
    with torch.no_grad():
        logits = model(
            torch.tensor(prompt_ids, dtype=torch.long, **kw),          # flat tokens
            {0: list(range(n_blocks))},                                # block_table: seq 0 -> blocks 0..n-1
            torch.tensor([T], dtype=torch.int32, **kw),                # q_len_per_seq
            torch.tensor([T], dtype=torch.int32, **kw),                # kv_len_per_seq
            torch.tensor([0], dtype=torch.long, **kw),                 # sequence_id
            torch.tensor([0], dtype=torch.long, **kw),                 # offset
            torch.tensor([T], dtype=torch.long, **kw),                 # length
            torch.arange(T, dtype=torch.long, **kw),                   # position_ids
            torch.zeros(T, dtype=torch.long, **kw),                    # pos_seq_id
        )
    return logits[-1].float().cpu()


def test_stage1_logits_match_hf():
    model = load_gpt2_into_paged()
    golden = json.load(open(GOLDEN_TOKENS))
    ref_logits = torch.load(GOLDEN_LOGITS)

    for i, g in enumerate(golden):
        mine, ref = paged_last_logits(model, g["prompt_ids"]), ref_logits[i]
        cos = F.cosine_similarity(mine, ref, dim=0).item()
        mx = (mine - ref).abs().max().item()
        t_mine, t_ref = mine.argmax().item(), ref.argmax().item()
        print(f"[{i}] cos={cos:.5f} max_abs_diff={mx:.3f} top1 mine={t_mine} ref={t_ref}")
        assert t_mine == t_ref, f"prompt {i}: top-1 differs"
        assert cos > 0.99, f"prompt {i}: cos={cos}"