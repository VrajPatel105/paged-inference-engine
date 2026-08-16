"""
This is the high level file that manages the Queue and handes the Queue to model_runner and benchmark.
"""

import queue
from core.model_runner import run
from transformer.tokenizer import Tokenizer
import pickle

q = queue.Queue()

with open('transformer/tokenizer.pkl', 'rb') as f:
    tok = pickle.load(f)

run(q, tok)