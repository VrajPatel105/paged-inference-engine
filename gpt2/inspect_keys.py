from transformers import GPT2LMHeadModel
from transformer.model import build_transformer
from transformer.config import transformer_configurations

# --- GPT-2 side ---
hf = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").state_dict()
print("=== GPT-2 (embeddings, layer 0, tail) ===")
for k, v in hf.items():
    if ".h." in k and not k.split(".h.")[1].startswith("0."):
        continue
    print(f"{k:40s} {tuple(v.shape)}")

cfg = dict(transformer_configurations)
cfg["tgt_vocab_size"] = 50257
cfg["tgt_max_seq_len"] = 1024

mine = build_transformer(cfg).state_dict()
print("\n=== My existing model (embeddings, layer 0, tail) ===")
for k, v in mine.items():
    if "decoder_blocks." in k and not k.split("decoder_blocks.")[1].startswith("0."):
        continue
    if "k_cache" in k or "v_cache" in k or "scale" in k:
        continue
    print(f"{k:40s} {tuple(v.shape)}")

# (mlenv) vraj@Vraj:/mnt/c/dev/projects/paged-inference-engine$ python -m gpt2.inspect_keys
# Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 148/148 [00:00<00:00, 1096.38it/s]
# === GPT-2 (embeddings, layer 0, tail) ===
# transformer.wte.weight                   (50257, 768)
# transformer.wpe.weight                   (1024, 768)
# transformer.h.0.ln_1.weight              (768,)
# transformer.h.0.ln_1.bias                (768,)
# transformer.h.0.attn.c_attn.weight       (768, 2304)
# transformer.h.0.attn.c_attn.bias         (2304,)
# transformer.h.0.attn.c_proj.weight       (768, 768)
# transformer.h.0.attn.c_proj.bias         (768,)
# transformer.h.0.ln_2.weight              (768,)
# transformer.h.0.ln_2.bias                (768,)
# transformer.h.0.mlp.c_fc.weight          (768, 3072)
# transformer.h.0.mlp.c_fc.bias            (3072,)
# transformer.h.0.mlp.c_proj.weight        (3072, 768)
# transformer.h.0.mlp.c_proj.bias          (768,)
# transformer.ln_f.weight                  (768,)
# transformer.ln_f.bias                    (768,)
# lm_head.weight                           (50257, 768)

# === My existing model (embeddings, layer 0, tail) ===
# tgt_embed.embedding.weight               (50257, 512)
# tgt_pe.pe                                (1, 1024, 512)
# decoder_blocks.0.masked_attention.W_q.weight (512, 512)
# decoder_blocks.0.masked_attention.W_q.bias (512,)
# decoder_blocks.0.masked_attention.W_k.weight (512, 512)
# decoder_blocks.0.masked_attention.W_k.bias (512,)
# decoder_blocks.0.masked_attention.W_v.weight (512, 512)
# decoder_blocks.0.masked_attention.W_v.bias (512,)
# decoder_blocks.0.masked_attention.W_o.weight (512, 512)
# decoder_blocks.0.masked_attention.W_o.bias (512,)
# decoder_blocks.0.residual_connection.0.norm.alpha (512,)
# decoder_blocks.0.residual_connection.0.norm.bias (512,)
# decoder_blocks.0.residual_connection.1.norm.alpha (512,)
# decoder_blocks.0.residual_connection.1.norm.bias (512,)
# decoder_blocks.0.feed_forward.linear1.weight (2048, 512)
# decoder_blocks.0.feed_forward.linear1.bias (2048,)
# decoder_blocks.0.feed_forward.linear2.weight (512, 2048)
# decoder_blocks.0.feed_forward.linear2.bias (512,)
# projection_layer.linear_layer.weight     (50257, 512)
# projection_layer.linear_layer.bias       (50257,)
# (mlenv) vraj@Vraj:/mnt/c/dev/projects/paged-inference-engine$ 