"""
Minimal reverse-mode autograd engine, NumPy backend.
This is the "PyTorch-under-the-hood" piece: a Tensor that tracks the graph
of ops applied to it and can run backward() to populate .grad on every
leaf tensor that requires it. Everything in transformer.py is built on
top of this — no torch/tf/jax anywhere in the model.
"""
import numpy as np


class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    def _wrap(self, other):
        return other if isinstance(other, Tensor) else Tensor(other)

    # ---------- core ops ----------
    def __add__(self, other):
        other = self._wrap(other)
        out = Tensor(self.data + other.data, self.requires_grad or other.requires_grad,
                     (self, other), "+")

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = self._wrap(other)
        out = Tensor(self.data * other.data, self.requires_grad or other.requires_grad,
                     (self, other), "*")

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad * self.data, other.data.shape)
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-self._wrap(other))

    def __truediv__(self, other):
        other = self._wrap(other)
        return self * (other ** -1.0)

    def __pow__(self, power):
        out = Tensor(self.data ** power, self.requires_grad, (self,), f"**{power}")

        def _backward():
            if self.requires_grad:
                self.grad += (power * self.data ** (power - 1)) * out.grad
        out._backward = _backward
        return out

    def matmul(self, other):
        other = self._wrap(other)
        out = Tensor(self.data @ other.data, self.requires_grad or other.requires_grad,
                     (self, other), "matmul")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ np.swapaxes(other.data, -1, -2)
            if other.requires_grad:
                self.grad_contrib = np.swapaxes(self.data, -1, -2) @ out.grad
                other.grad += _unbroadcast(self.grad_contrib, other.data.shape)
        out._backward = _backward
        return out

    def transpose(self, axis1=-1, axis2=-2):
        out = Tensor(np.swapaxes(self.data, axis1, axis2), self.requires_grad, (self,), "T")

        def _backward():
            if self.requires_grad:
                self.grad += np.swapaxes(out.grad, axis1, axis2)
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), self.requires_grad, (self,), "relu")

        def _backward():
            if self.requires_grad:
                self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def gelu(self):
        # tanh approximation, same as GPT-2
        c = np.sqrt(2 / np.pi)
        x = self.data
        inner = c * (x + 0.044715 * x ** 3)
        t = np.tanh(inner)
        out_data = 0.5 * x * (1 + t)
        out = Tensor(out_data, self.requires_grad, (self,), "gelu")

        def _backward():
            if self.requires_grad:
                sech2 = 1 - t ** 2
                dinner_dx = c * (1 + 3 * 0.044715 * x ** 2)
                dgelu_dx = 0.5 * (1 + t) + 0.5 * x * sech2 * dinner_dx
                self.grad += dgelu_dx * out.grad
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), self.requires_grad, (self,), "sum")

        def _backward():
            if self.requires_grad:
                g = out.grad
                if not keepdims and axis is not None:
                    g = np.expand_dims(g, axis)
                self.grad += np.ones_like(self.data) * g
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def softmax(self, axis=-1):
        x = self.data - np.max(self.data, axis=axis, keepdims=True)
        e = np.exp(x)
        s = e / e.sum(axis=axis, keepdims=True)
        out = Tensor(s, self.requires_grad, (self,), "softmax")

        def _backward():
            if self.requires_grad:
                g = out.grad
                dot = np.sum(g * s, axis=axis, keepdims=True)
                self.grad += s * (g - dot)
        out._backward = _backward
        return out

    def layernorm(self, gamma, beta, eps=1e-5):
        x = self.data
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        xhat = (x - mu) / np.sqrt(var + eps)
        out_data = xhat * gamma.data + beta.data
        out = Tensor(out_data, True, (self, gamma, beta), "layernorm")
        N = x.shape[-1]

        def _backward():
            g = out.grad
            if gamma.requires_grad:
                gamma.grad += _unbroadcast((g * xhat), gamma.data.shape)
            if beta.requires_grad:
                beta.grad += _unbroadcast(g, beta.data.shape)
            if self.requires_grad:
                dxhat = g * gamma.data
                std_inv = 1.0 / np.sqrt(var + eps)
                dvar = np.sum(dxhat * (x - mu) * -0.5 * std_inv ** 3, axis=-1, keepdims=True)
                dmu = np.sum(dxhat * -std_inv, axis=-1, keepdims=True) + \
                    dvar * np.mean(-2.0 * (x - mu), axis=-1, keepdims=True)
                self.grad += dxhat * std_inv + dvar * 2.0 * (x - mu) / N + dmu / N
        out._backward = _backward
        return out

    def reshape(self, *shape):
        old_shape = self.data.shape
        out = Tensor(self.data.reshape(*shape), self.requires_grad, (self,), "reshape")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(old_shape)
        out._backward = _backward
        return out

    # ---------- backward pass ----------
    def backward(self):
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()


def _unbroadcast(grad, target_shape):
    """Sum-reduce grad down to target_shape after a numpy broadcast."""
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(target_shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad
