"""
This is the high level file that manages the Queue and handes the Queue to model_runner and benchmark.
"""

import queue
import pickle
from core.model_runner import run

q = queue.Queue()

with open('transformer/tokenizer.pkl', 'rb') as f:
    tok = pickle.load(f)

test_prompt = tok.encode_sentence("Once upon a time", add_sos=True, add_eos=False)
q.put(test_prompt)

run(q)