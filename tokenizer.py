"""Simple character-level tokenizer (byte-level would need BPE; char-level
keeps vocab small (~90 symbols) which matters a lot given we train on CPU
with a hand-rolled NumPy backward pass, not cuDNN kernels)."""
import json


class CharTokenizer:
    def __init__(self, text=None, vocab=None):
        if vocab is not None:
            self.stoi = vocab
        else:
            chars = sorted(set(text))
            self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self):
        return len(self.stoi)

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.stoi, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            stoi = json.load(f)
        return cls(vocab=stoi)
