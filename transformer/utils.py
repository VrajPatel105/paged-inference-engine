import torch

def make_masks(encoder_input, decoder_input, pad_id, device):
  
  tgt_len = decoder_input.size(1)
  src_mask = (encoder_input != pad_id).unsqueeze(1).unsqueeze(1)
  tgt_pad_mask = (decoder_input != pad_id).unsqueeze(1).unsqueeze(1)
  causal_m = torch.tril(torch.ones(tgt_len, tgt_len, device=device)).bool()
  tgt_mask = (tgt_pad_mask & causal_m).int()

  return src_mask.int(), tgt_mask

# just making another casual mask seperate function that will be in help during translate()
def causal_mask(size, device):
    return torch.tril(torch.ones(size, size, device=device)).int().unsqueeze(0).unsqueeze(0)
