# updated for the larger 200k-row Tatoeba dataset
configurations = {
    'd_model' : 512,
    'num_heads' : 8,
    'num_blocks' : 6,
    'src_max_seq_len' : 200,   # was 323 -> reduced but still comfortably above observed max (165)
    'tgt_max_seq_len' : 200,   # was 123 -> bumped up, observed German max was 159, added margin
    'src_vocab_size' : 29240,  # was 10000 -> actual English vocab is 29,240
    'tgt_vocab_size' : 52239,  # was 3883  -> actual German vocab is 52,239
    'path' : '/mnt/c/dev/projects/cpp-gpu-inference/en-de-transformer/English-German.tsv',
    'max_len': 200,
    'batch_size' : 32,         
    'learning_rate' : 0.0001,
    'epochs' : 50            
}