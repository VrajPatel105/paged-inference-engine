import json
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel

from gpt2.load_weights import load_gpt2_into_paged, HF_NAME

DEV, BLOCK_SIZE = "cuda", 64


def call_paged(model, ids):
    T = len(ids)
    nb = (T + BLOCK_SIZE - 1) // BLOCK_SIZE
    i32 = lambda x: torch.tensor(x, dtype=torch.int32, device=DEV)
    i64 = lambda x: torch.tensor(x, dtype=torch.long, device=DEV)
    with torch.no_grad():
        return model(i64(ids), {0: list(range(nb))}, i32([T]), i32([T]),
                     i64([0]), i64([0]), i64([T]), i64(list(range(T))), i64([0] * T))


def paged_with_hidden(model, ids):
    outs = []
    hooks = [b.register_forward_hook(lambda m, i, o: outs.append(o.detach().float().cpu()))
             for b in model.decoder_blocks]
    logits = call_paged(model, ids)
    for h in hooks:
        h.remove()
    return logits[-1].float().cpu(), outs


def hf_with_hidden(hf, ids):
    outs = []
    hooks = [b.register_forward_hook(
        lambda m, i, o: outs.append((o[0] if isinstance(o, tuple) else o).detach().float().reshape(-1, 768)))
        for b in hf.transformer.h]
    with torch.no_grad():
        logits = hf(torch.tensor([ids])).logits[0, -1].float()
    for h in hooks:
        h.remove()
    return logits, outs


def compare(mine, ref):
    lp_m, lp_r = F.log_softmax(mine, -1), F.log_softmax(ref, -1)
    kl = F.kl_div(lp_m, lp_r, log_target=True, reduction="sum").item()  # KL(ref || mine)
    top5_m, top5_r = set(mine.topk(5).indices.tolist()), set(ref.topk(5).indices.tolist())
    return {
        "logprob_maxdiff": (lp_m - lp_r).abs().max().item(),
        "kl": kl,
        "top1": mine.argmax().item() == ref.argmax().item(),
        "top5_overlap": len(top5_m & top5_r),
    }


if __name__ == "__main__":
    model = load_gpt2_into_paged()
    hf = GPT2LMHeadModel.from_pretrained(HF_NAME).eval()
    golden = json.load(open("gpt2/golden_tokens.json"))

    # A) does the error grow with prompt length? (k=1 has no re-rounding)
    ids_long = golden[4]["prompt_ids"]
    print("== A) error vs prompt length (fox prompt prefixes) ==")
    for k in [1, 2, 4, 8, 12, 16, len(ids_long)]:
        k = min(k, len(ids_long))
        ids = ids_long[:k]
        m, _ = paged_with_hidden(model, ids)
        r, _ = hf_with_hidden(hf, ids)
        c = compare(m, r)
        print(f"k={k:3d}  logprob_maxdiff={c['logprob_maxdiff']:7.3f}  KL={c['kl']:8.4f}  "
              f"top1={c['top1']}  top5_overlap={c['top5_overlap']}/5")

    # B) where does the error enter? relative error of each block's output
    ids = golden[2]["prompt_ids"]
    _, mh = paged_with_hidden(model, ids)
    _, rh = hf_with_hidden(hf, ids)
    print(f"\n== B) per-layer relative hidden-state error (prompt 2, {len(ids)} tokens) ==")
    for i, (a, b) in enumerate(zip(mh, rh)):
        a, b = a.reshape(-1, 768), b.reshape(-1, 768)
        rel_all = ((a - b).norm() / b.norm()).item()
        rel_last = ((a[-1] - b[-1]).norm() / b[-1].norm()).item()
        print(f"layer {i:2d}  rel_err_all={rel_all:.4f}  rel_err_last_token={rel_last:.4f}")