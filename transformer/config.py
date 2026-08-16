# updated for the larger 200k-row Tatoeba dataset
transformer_configurations = {
    'd_model': 512,
    'num_heads': 8,
    'num_blocks': 6,
    'tgt_max_seq_len': 200,
    'tgt_vocab_size': 12849,
    'max_len': 200,
    'batch_size': 32,
    'learning_rate': 0.0001,
    'epochs': 50
}