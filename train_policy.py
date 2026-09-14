import os
import time
import numpy as np
from tokenizer import CharTokenizer
from transformer import GPTFromScratch, cross_entropy_loss
from optim import Adam

CORPUS_PATH = "data/policy_corpus.txt"
CKPT_PATH = "data/policygpt_weights.npz"
VOCAB_PATH = "data/policy_vocab.json"

CONTEXT_LEN = 256     # longer than CodeGPT's 64 -- conversations need more context
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 3
D_FF = 384
BATCH_SIZE = 8        # smaller batch since T is much larger now (attention is O(T^2))
LR = 3e-3
STEPS = int(os.environ.get("STEPS", 500))
EVAL_EVERY = 20


def get_batch(data, batch_size, context_len):
    ix = np.random.randint(0, len(data) - context_len - 1, size=batch_size)
    x = np.stack([data[i:i + context_len] for i in ix])
    y = np.stack([data[i + 1:i + context_len + 1] for i in ix])
    return x, y


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


def main():
    with open(CORPUS_PATH, encoding="utf-8") as f:
        text = f.read()
    tok = CharTokenizer(text)
    tok.save(VOCAB_PATH)
    data = np.array(tok.encode(text), dtype=np.int64)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    print(f"corpus: {len(text)} chars, vocab size: {tok.vocab_size}, "
          f"train tokens: {len(train_data)}, val tokens: {len(val_data)}")

    model = GPTFromScratch(vocab_size=tok.vocab_size, d_model=D_MODEL, n_heads=N_HEADS,
                            n_layers=N_LAYERS, d_ff=D_FF, max_len=CONTEXT_LEN)
    n_params = sum(p.data.size for p in model.params())
    print(f"model params: {n_params:,}")

    names = _param_names(model)
    if os.environ.get("RESUME", "0") == "1" and os.path.exists(CKPT_PATH):
        saved = np.load(CKPT_PATH)
        for name, p in zip(names, model.params()):
            p.data = saved[name].astype(p.data.dtype)
        print(f"resumed weights from {CKPT_PATH}")

    opt = Adam(model.params(), lr=LR)

    import gc
    t0 = time.time()
    losses = []
    for step in range(1, STEPS + 1):
        xb, yb = get_batch(train_data, BATCH_SIZE, CONTEXT_LEN)
        logits = model(xb)
        loss = cross_entropy_loss(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.data))
        del logits, loss
        if step % 2 == 0:
            gc.collect()

        if step % EVAL_EVERY == 0 or step == 1:
            xb_v, yb_v = get_batch(val_data, min(BATCH_SIZE, len(val_data) - CONTEXT_LEN - 1), CONTEXT_LEN)
            val_loss = cross_entropy_loss(model(xb_v), yb_v).data
            avg_train = np.mean(losses[-EVAL_EVERY:])
            elapsed = time.time() - t0
            print(f"step {step:5d} | train_loss {avg_train:.4f} | val_loss {val_loss:.4f} "
                  f"| {elapsed:.1f}s elapsed")

    param_dict = {}
    for name, p in zip(names, model.params()):
        param_dict[name] = p.data
    np.savez(CKPT_PATH, **param_dict)
    print(f"saved weights to {CKPT_PATH}")


if __name__ == "__main__":
    main()
