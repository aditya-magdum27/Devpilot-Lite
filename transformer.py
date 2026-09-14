"""
Decoder-only (GPT-style) transformer, built entirely on autograd.Tensor.
No torch / tf / jax. Every matrix multiply, attention head, layernorm and
gelu here runs through the Tensor class in autograd.py, so calling
.backward() computes real gradients through the whole network.
"""
import numpy as np
from autograd import Tensor


def param(*shape, scale=0.02):
    return Tensor(np.random.randn(*shape) * scale, requires_grad=True)


class Linear:
    def __init__(self, d_in, d_out):
        self.W = param(d_in, d_out)
        self.b = Tensor(np.zeros(d_out), requires_grad=True)

    def __call__(self, x):
        return x.matmul(self.W) + self.b

    def params(self):
        return [self.W, self.b]


class LayerNorm:
    def __init__(self, d):
        self.gamma = Tensor(np.ones(d), requires_grad=True)
        self.beta = Tensor(np.zeros(d), requires_grad=True)

    def __call__(self, x):
        return x.layernorm(self.gamma, self.beta)

    def params(self):
        return [self.gamma, self.beta]


def embedding_lookup(table: Tensor, idx: np.ndarray):
    """idx: (B, T) int array. table: (V, D) Tensor. Returns (B, T, D) Tensor."""
    out_data = table.data[idx]
    out = Tensor(out_data, requires_grad=table.requires_grad, _children=(table,), _op="embed")

    def _backward():
        if table.requires_grad:
            np.add.at(table.grad, idx, out.grad)
    out._backward = _backward
    return out


class CausalSelfAttention:
    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.out_proj = Linear(d_model, d_model)

    def __call__(self, x):
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head

        q = self.q_proj(x).reshape(B, T, H, dh).transpose(1, 2)  # (B,H,T,dh)
        k = self.k_proj(x).reshape(B, T, H, dh).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, H, dh).transpose(1, 2)

        scores = q.matmul(k.transpose(-1, -2)) * (1.0 / np.sqrt(dh))  # (B,H,T,T)

        # causal mask: future positions -> -inf before softmax
        mask = np.triu(np.ones((T, T)), k=1).astype(bool)
        masked_data = scores.data.copy()
        masked_data[..., mask] = -1e9
        scores_masked = Tensor(masked_data, requires_grad=scores.requires_grad,
                                _children=(scores,), _op="causal_mask")

        def _mask_backward():
            if scores.requires_grad:
                g = scores_masked.grad.copy()
                g[..., mask] = 0.0
                scores.grad += g
        scores_masked._backward = _mask_backward

        attn = scores_masked.softmax(axis=-1)          # (B,H,T,T)
        out = attn.matmul(v)                            # (B,H,T,dh)
        out = out.transpose(1, 2).reshape(B, T, D)      # (B,T,D)
        return self.out_proj(out)

    def params(self):
        p = []
        for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            p += m.params()
        return p


class FeedForward:
    def __init__(self, d_model, d_ff):
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)

    def __call__(self, x):
        return self.fc2(self.fc1(x).gelu())

    def params(self):
        return self.fc1.params() + self.fc2.params()


class Block:
    """Pre-norm transformer block: x + Attn(LN(x)), then x + FFN(LN(x))."""
    def __init__(self, d_model, n_heads, d_ff):
        self.ln1 = LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

    def params(self):
        return self.ln1.params() + self.attn.params() + self.ln2.params() + self.ffn.params()


class GPTFromScratch:
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2, d_ff=256, max_len=128):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.tok_emb = param(vocab_size, d_model)
        self.pos_emb = param(max_len, d_model)
        self.blocks = [Block(d_model, n_heads, d_ff) for _ in range(n_layers)]
        self.ln_f = LayerNorm(d_model)
        self.head = Linear(d_model, vocab_size)

    def __call__(self, idx: np.ndarray):
        B, T = idx.shape
        tok = embedding_lookup(self.tok_emb, idx)                 # (B,T,D)
        pos_idx = np.arange(T)[None, :].repeat(B, axis=0)
        pos = embedding_lookup(self.pos_emb, pos_idx)              # (B,T,D)
        x = tok + pos
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)                                      # (B,T,V)
        return logits

    def params(self):
        p = [self.tok_emb, self.pos_emb] + self.ln_f.params() + self.head.params()
        for b in self.blocks:
            p += b.params()
        return p


def cross_entropy_loss(logits: Tensor, targets: np.ndarray):
    """logits: (B,T,V) Tensor, targets: (B,T) int array.

    NOTE: this fuses softmax+NLL and back-props the combined shortcut
    gradient (probs - one_hot) straight into the pre-softmax logits node.
    Softmax is only ever used to compute forward probabilities here (via
    plain numpy) — it is NOT a separate graph node, precisely because the
    shortcut gradient already accounts for the softmax Jacobian. Wiring it
    in as a graph node too would double-apply that Jacobian, which is
    exactly the bug an earlier version of this file had (caught by a
    finite-difference gradient check before training).
    """
    B, T, V = logits.shape
    flat_logits = logits.reshape(B * T, V)          # still a graph node
    flat_targets = targets.reshape(-1)

    x = flat_logits.data - np.max(flat_logits.data, axis=-1, keepdims=True)
    e = np.exp(x)
    probs = e / e.sum(axis=-1, keepdims=True)         # plain numpy, forward-only

    picked = probs[np.arange(B * T), flat_targets]
    nll = -np.log(np.clip(picked, 1e-12, None))
    loss_val = nll.mean()

    out = Tensor(loss_val, requires_grad=True, _children=(flat_logits,), _op="softmax_ce")

    def _backward():
        grad = probs.copy()
        grad[np.arange(B * T), flat_targets] -= 1.0
        grad /= (B * T)
        flat_logits.grad += grad * out.grad
    out._backward = _backward
    return out
