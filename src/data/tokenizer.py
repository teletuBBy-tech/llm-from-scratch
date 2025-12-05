# src/data/tokenizer.py

import tiktoken

class TiktokenWrapper:
    def __init__(self, name="gpt2"):
        self.enc = tiktoken.get_encoding(name)

    def encode(self, text, allowed_special=None):
        return self.enc.encode(text, allowed_special=allowed_special or set())

    def decode(self, tokens):
        return self.enc.decode(tokens)

    @property
    def vocab_size(self):
        return self.enc.n_vocab
