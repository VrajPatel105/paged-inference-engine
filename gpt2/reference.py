import json
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

PROMPTS = [
    "Once upon a time",
    "The capital of France is",
    "Boston is very well know for its ",
    "def fibonacci(n):",
    "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The",
]
MAX_NEW_TOKENS = 50

tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval()

golden = []
logits_first = []
with torch.no_grad():
    for p in PROMPTS:
        ids = tokenizer(p, return_tensors="pt").input_ids  # one prompt at a time, no padding
        # logits at the last prompt position, for the Stage 1 logit comparison
        logits_first.append(model(ids).logits[0, -1].clone())
        out = model.generate(
            ids,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
        )
        golden.append({
            "prompt": p,
            "prompt_ids": ids[0].tolist(),
            "generated_ids": out[0, ids.shape[1]:].tolist(),
        })
        print(repr(p), "->", repr(tokenizer.decode(golden[-1]["generated_ids"])))

with open("gpt2/golden_tokens.json", "w") as f:
    json.dump(golden, f, indent=2)
torch.save(torch.stack(logits_first), "gpt2/golden_last_logits.pt")