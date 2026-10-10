import torch
from transformers import GPT2LMHeadModel

from gpt2.gpt2_model import build_gpt2

HF_NAME = "openai-community/gpt2"
NUM_LAYERS, D = 12, 768


def _put(dst, key, tensor, ref_model_state):
    assert tensor.shape == ref_model_state[key].shape, (
        f"{key}: got {tuple(tensor.shape)}, model expects {tuple(ref_model_state[key].shape)}"
    )
    dst[key] = tensor.clone()


def map_gpt2_to_paged(hf, model_state):
    out = {}
    put = lambda k, t: _put(out, k, t, model_state)

    put("wte.weight", hf["transformer.wte.weight"])
    put("wpe.weight", hf["transformer.wpe.weight"])
    put("lm_head.weight", hf["transformer.wte.weight"])      # tied
    put("ln_f.weight", hf["transformer.ln_f.weight"])
    put("ln_f.bias", hf["transformer.ln_f.bias"])

    for i in range(NUM_LAYERS):
        h = f"transformer.h.{i}"
        m = f"decoder_blocks.{i}"

        put(f"{m}.ln1.weight", hf[f"{h}.ln_1.weight"])
        put(f"{m}.ln1.bias", hf[f"{h}.ln_1.bias"])
        put(f"{m}.ln2.weight", hf[f"{h}.ln_2.weight"])
        put(f"{m}.ln2.bias", hf[f"{h}.ln_2.bias"])

        # HF Conv1D stores (in, out); nn.Linear stores (out, in) -> transpose.
        # c_attn is [q | k | v] along the output dim, heads contiguous within each.
        w = hf[f"{h}.attn.c_attn.weight"]            # (768, 2304)
        b = hf[f"{h}.attn.c_attn.bias"]              # (2304,)
        wq, wk, wv = w.split(D, dim=1)
        bq, bk, bv = b.split(D, dim=0)
        put(f"{m}.masked_attention.W_q.weight", wq.t()); put(f"{m}.masked_attention.W_q.bias", bq)
        put(f"{m}.masked_attention.W_k.weight", wk.t()); put(f"{m}.masked_attention.W_k.bias", bk)
        put(f"{m}.masked_attention.W_v.weight", wv.t()); put(f"{m}.masked_attention.W_v.bias", bv)
        put(f"{m}.masked_attention.W_o.weight", hf[f"{h}.attn.c_proj.weight"].t())
        put(f"{m}.masked_attention.W_o.bias", hf[f"{h}.attn.c_proj.bias"])

        put(f"{m}.feed_forward.linear1.weight", hf[f"{h}.mlp.c_fc.weight"].t())
        put(f"{m}.feed_forward.linear1.bias", hf[f"{h}.mlp.c_fc.bias"])
        put(f"{m}.feed_forward.linear2.weight", hf[f"{h}.mlp.c_proj.weight"].t())
        put(f"{m}.feed_forward.linear2.bias", hf[f"{h}.mlp.c_proj.bias"])

    return out


def load_gpt2_into_paged(num_kv_blocks=256, device="cuda"):
    hf = GPT2LMHeadModel.from_pretrained(HF_NAME).state_dict()
    model = build_gpt2(num_kv_blocks=num_kv_blocks)

    mapped = map_gpt2_to_paged(hf, model.state_dict())
    missing, unexpected = model.load_state_dict(mapped, strict=False)

    missing = [k for k in missing if not any(s in k for s in ("k_cache", "v_cache", "k_scale", "v_scale"))]
    assert not missing, f"missing weights: {missing}"
    assert not unexpected, f"unexpected keys: {unexpected}"

    return model.to(device).eval()   # fp32 on purpose: your attention casts O to fp32 before W_o


if __name__ == "__main__":
    load_gpt2_into_paged()
    print("GPT-2 weights loaded cleanly")