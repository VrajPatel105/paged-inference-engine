"""
This is the high level file that manages the Queue and handes the Queue to model_runner and benchmark.
"""

import queue
from core.model_runner import run
# build the queue
q = queue.Queue()


# using the already build queue, call the runner now.

run(q)