import torch

from transformer.model import build_transformer
from transformer.config import transformer_configurations


def map_train_keys_to_paged(train_state):
    new_state = {}

    name_map = {
        'embed': 'tgt_embed',
        'pe': 'tgt_pe',
    }

    for key, value in train_state.items():
        new_key = key

        for old, new in name_map.items():
            if new_key.startswith(old + '.'):
                new_key = new + new_key[len(old):]
                break

        new_key = new_key.replace('.attention.', '.masked_attention.')
        new_state[new_key] = value

    return new_state


def load_trained_weights(checkpoint_path):
    train_state = torch.load(checkpoint_path, map_location='cpu')

    paged_model = build_transformer(transformer_configurations)

    mapped_state = map_train_keys_to_paged(train_state)
    paged_model.load_state_dict(mapped_state, strict=False)
    paged_model = paged_model.to('cuda')

    missing, unexpected = paged_model.load_state_dict(mapped_state, strict=False)

    if missing:
        print("missing keys (in paged model, not found in checkpoint):")
        for k in missing:
            print(" ", k)

    if unexpected:
        print("unexpected keys (in checkpoint, not found in paged model):")
        for k in unexpected:
            print(" ", k)

    if not missing and not unexpected:
        print("all keys matched cleanly")

    return paged_model


if __name__ == "__main__":
    model = load_trained_weights('transformer/decoder_only.pt')