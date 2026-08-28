# CodeGPT-Scratch
A GPT-style transformer built entirely from scratch (PyTorch) for Python code autosuggest — hand-written self-attention, multi-head attention, and a custom-trained BPE tokenizer, with an attention visualizer and ablation study.

# CodeGPT-from-scratch

A small GPT-style transformer, built entirely from scratch (no pretrained
models, no HuggingFace model classes), trained to predict the next tokens
of Python code given a prefix — i.e. autosuggest/autocomplete.

**Current scope**: general Python syntax (trained on Python's own
standard library source code), not yet specialized to any particular
library. This is a deliberate phase-1 scope, not an oversight — see
"Known limitations" below.

## Why this project

Most fresher "built a transformer" projects either fine-tune an existing
model or train on generic prose. This one:
- implements self-attention, multi-head attention, and the full
  transformer block **from scratch** in PyTorch — every tensor operation
  is hand-written, not a call to `nn.MultiheadAttention`
- is trained on **code**, not prose — gives a genuine autosuggest use
  case and structurally interesting attention patterns to visualize
  (bracket matching, indentation, function signatures)
- uses **Python's own standard library** as training data — real,
  high-quality, syntactically consistent code, no download or scraping
  needed
- trains its **own BPE tokenizer from scratch** on that corpus, rather
  than reusing GPT-2's or any pretrained vocabulary

## Architecture

Standard decoder-only transformer (same family as GPT):

\`\`\`
token embedding + positional embedding
  -> N x [ self-attention -> residual + layernorm -> feedforward -> residual + layernorm ]
  -> final layernorm
  -> linear head -> vocabulary logits
\`\`\`

- **Self-attention**: each token computes a Query, Key, and Value.
  Attention scores are `Q @ K^T`, scaled by `1/sqrt(head_size)` (keeps
  softmax gradients from vanishing at larger head sizes), masked
  causally (a token can't attend to future tokens), softmax'd into
  weights, then used to compute a weighted sum of Values.
- **Multi-head attention**: several heads run in parallel and are
  concatenated — each can specialize in a different relationship (e.g.
  one head attending to matching brackets, another to the most recent
  variable assignment).
- **Positional embeddings**: a learned vector per position, added to the
  token embedding — without this, self-attention is permutation-invariant
  and the model has no notion of token order.
- **Residual connections + pre-norm layernorm**: let gradients flow
  through the network even as blocks are stacked.

## Pipeline

\`\`\`bash
pip install -r requirements.txt

python prepare_data.py          # build training corpus from Python's stdlib
python train_tokenizer.py       # train a BPE tokenizer from scratch
python train.py                 # train the model (GPU recommended)
python generate.py --prefix "def add(a, b):" --max-tokens 30
python visualize_attention.py --code "def add(a, b):\n    return a + b" --layer 0 --head 0
python ablation.py              # positional encoding on/off, single vs multi-head
\`\`\`

## Ablation study

Runs short, controlled training runs (same seed, same data, same
iteration count) under three configurations and reports real validation
loss — concrete numbers, not just claims:

1. Multi-head (4 heads) + positional encoding (baseline)
2. Single-head + positional encoding
3. Multi-head, **no** positional encoding

## Known limitations (honest scope)

- Trained on general Python syntax only — not yet extended to a
  specific library's usage patterns (phase 2 candidate).
- BPE vocab (8000) is corpus-specific to the stdlib.
- Model is intentionally small (a few million parameters) — a
  from-scratch learning project, not a production-scale system.
- Generation uses simple temperature-scaled sampling, no beam search or
  top-k/top-p decoding.
\`\`\`
