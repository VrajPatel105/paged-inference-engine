from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import threading
import queue
import pickle

from core.model_runner import run

app = FastAPI()

with open('transformer/tokenizer.pkl', 'rb') as f:
    tok = pickle.load(f)

q = queue.Queue()
output_state = {}
window_map = {}


def run_with_error_logging(*args):
    try:
        run(*args)
    except Exception:
        import traceback
        traceback.print_exc()


thread = threading.Thread(target=run_with_error_logging, args=(q, tok, output_state, window_map), daemon=True)
thread.start()


class PromptRequest(BaseModel):
    prompt: str


@app.post("/submit/{window_id}")
def submit(window_id: int, req: PromptRequest):
    tokens = tok.encode_sentence(req.prompt, add_sos=True, add_eos=False)
    q.put((window_id, tokens))
    return {"status": "submitted"}


@app.get("/output")
def get_output():
    return output_state


app.mount("/", StaticFiles(directory="demo/static", html=True), name="static")