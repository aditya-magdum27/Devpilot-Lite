"""
PHASE 1: PolicyGPT inference, standalone.

This must work correctly on its own, with no LangGraph or Streamlit
involved, before anything is wired to persistence. Test with:

    python model.py

What's happening: loads the trained from-scratch transformer weights and
vocab, and exposes a single generate(prompt_text) function that the
LangGraph node (graph.py) will call. Nothing here knows about threads,
messages, or SQLite -- it's a pure text-in, text-out function.
"""
import numpy as np
from tokenizer import CharTokenizer
from transformer import GPTFromScratch

VOCAB_PATH = "data/policy_vocab.json"
WEIGHTS_PATH = "data/policygpt_weights.npz"

MODEL_CFG = dict(d_model=128, n_heads=4, n_layers=3, d_ff=384, max_len=256)

_TOK = None
_MODEL = None


def _param_names(model):
    names = ["tok_emb", "pos_emb", "ln_f.gamma", "ln_f.beta", "head.W", "head.b"]
    for i in range(len(model.blocks)):
        b = f"blocks.{i}."
        names += [b + "ln1.gamma", b + "ln1.beta",
                  b + "attn.q_proj.W", b + "attn.q_proj.b",
                  b + "attn.k_proj.W", b + "attn.k_proj.b",
                  b + "attn.v_proj.W", b + "attn.v_proj.b",
                  b + "attn.out_proj.W", b + "attn.out_proj.b",
                  b + "ln2.gamma", b + "ln2.beta",
                  b + "ffn.fc1.W", b + "ffn.fc1.b",
                  b + "ffn.fc2.W", b + "ffn.fc2.b"]
    return names


def load_model():
    global _TOK, _MODEL
    if _MODEL is not None:
        return _TOK, _MODEL
    tok = CharTokenizer.load(VOCAB_PATH)
    model = GPTFromScratch(vocab_size=tok.vocab_size, **MODEL_CFG)
    weights = np.load(WEIGHTS_PATH)
    for name, p in zip(_param_names(model), model.params()):
        p.data = weights[name].astype(p.data.dtype)
    _TOK, _MODEL = tok, model
    return tok, model


def generate(prompt_text: str, max_new_tokens: int = 150, temperature: float = 0.7) -> str:
    """Pure function: text in, continuation text out. No message objects,
    no state, no persistence -- that's added in later phases."""
    tok, model = load_model()
    ctx = MODEL_CFG["max_len"]
    ids = tok.encode(prompt_text) or [0]
    generated_ids = []
    for _ in range(max_new_tokens):
        window = np.array([ids[-ctx:]])
        logits = model(window)
        last_logits = logits.data[0, -1] / max(temperature, 1e-6)
        last_logits -= last_logits.max()
        probs = np.exp(last_logits)
        probs /= probs.sum()
        next_id = int(np.random.choice(len(probs), p=probs))
        ids.append(next_id)
        generated_ids.append(next_id)
        # stop early if the model produces a newline followed by "Human:"
        # style turn marker -- crude but good enough for this toy model to
        # avoid running on into a hallucinated next turn
        if len(generated_ids) > 10 and tok.decode(generated_ids[-7:]) == "\nHuman:":
            generated_ids = generated_ids[:-7]
            break
    return tok.decode(generated_ids)


if __name__ == "__main__":
    tok, model = load_model()
    print(f"loaded PolicyGPT: vocab={tok.vocab_size}, params={sum(p.data.size for p in model.params()):,}")
    test_prompt = "Human: What is fiscal policy?\nAI:"
    out = generate(test_prompt, max_new_tokens=120, temperature=0.6)
    print("--- prompt ---")
    print(test_prompt)
    print("--- completion ---")
    print(out)
