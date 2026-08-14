

class Sequence:

    def __init__(self, seq_id, prompt_token_ids):
        self.seq_id = seq_id
        self.prompt_token_ids = prompt_token_ids
        self.is_finished = False
        self.token_ids = list(prompt_token_ids)    