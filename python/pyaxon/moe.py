"""Neural Mixture-of-Experts layer (the gated MoE, DeepSeek/Switch style).

The modular router (`pyaxon.modular`) is MoE at the SYSTEM level (a classifier picks a
compartment). This is MoE INSIDE the network: a feed-forward layer replaced by N expert
FFNs plus a **gating network** that activates only the **top-k** experts per token. You
get the capacity of N experts at the compute of k -- the core trick behind DeepSeek-V3.

This is a clear NumPy reference implementation (forward pass + gating + top-k + weighted
combine + load-balancing aux loss). It shows the mechanism; wiring it into the C++
autograd for GPU training is the production step.
"""

import numpy as np


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _gelu(x):
    return 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


class MoELayer:
    """N expert FFNs + a gating network; only top_k experts run per token.

    forward(X) with X of shape (tokens, d_model) returns (tokens, d_model). Also exposes
    the routing (which experts fired) and the load-balancing loss.
    """

    def __init__(self, d_model, d_hidden, num_experts=8, top_k=2, seed=0):
        self.num_experts = num_experts
        self.top_k = top_k
        rng = np.random.default_rng(seed)
        s1, s2 = np.sqrt(2 / d_model), np.sqrt(2 / d_hidden)
        # one FFN (W1: d_model->d_hidden, W2: d_hidden->d_model) per expert
        self.W1 = rng.normal(0, s1, (num_experts, d_model, d_hidden))
        self.W2 = rng.normal(0, s2, (num_experts, d_hidden, d_model))
        self.W_gate = rng.normal(0, s1, (d_model, num_experts))   # gating network
        self.last_load_ = None       # fraction of tokens routed to each expert

    def _expert(self, e, x):
        return _gelu(x @ self.W1[e]) @ self.W2[e]

    def forward(self, X):
        X = np.asarray(X, dtype=np.float64)
        gate = _softmax(X @ self.W_gate)                      # (tokens, num_experts)
        # top-k experts per token, renormalized weights
        topk = np.argsort(gate, axis=1)[:, ::-1][:, :self.top_k]
        out = np.zeros_like(X)
        counts = np.zeros(self.num_experts)
        for i in range(X.shape[0]):
            experts = topk[i]
            w = gate[i, experts]
            w = w / w.sum()
            for e, we in zip(experts, w):
                out[i] += we * self._expert(e, X[i])
                counts[e] += 1
        self.last_load_ = counts / max(1, X.shape[0] * self.top_k)
        return out

    def load_balance_loss(self, X):
        """Auxiliary loss that penalizes uneven expert usage (keeps experts balanced,
        avoiding collapse to a few). Lower is better; ideal ~1.0 when perfectly even."""
        X = np.asarray(X, dtype=np.float64)
        gate = _softmax(X @ self.W_gate)
        importance = gate.mean(axis=0)                        # mean gate prob per expert
        # coefficient of variation squared -> 0 when perfectly balanced
        return float(self.num_experts * (importance ** 2).sum())

    def __call__(self, X):
        return self.forward(X)


class SoftMoE:
    """TRAINABLE Mixture-of-Experts (soft gating) built on the pyaxon autograd.

    All experts run, weighted by a softmax gate -- fully differentiable, so it trains
    end-to-end with `.backward()` + an optimizer. (Hard top-k routing, as in `MoELayer`,
    is not differentiable through the selection; use it as an inference-time speedup.)

    forward(x) with x a pyaxon Tensor (tokens, d_model) -> (tokens, d_model).
    """

    def __init__(self, d_model, d_hidden, num_experts=4, seed=0):
        from ._axon import nn as _nn
        self.num_experts = num_experts
        self.gate = _nn.Linear(d_model, num_experts, seed=seed)
        self.w1 = [_nn.Linear(d_model, d_hidden, seed=seed + 1 + 2 * e) for e in range(num_experts)]
        self.w2 = [_nn.Linear(d_hidden, d_model, seed=seed + 2 + 2 * e) for e in range(num_experts)]

    def forward(self, x):
        from ._axon import add as _add
        from ._axon import gelu as _gelu
        from ._axon import mul as _mul
        from ._axon import softmax as _softmax
        g = _softmax(self.gate(x))                     # (tokens, num_experts)
        out = None
        for e in range(self.num_experts):
            ge = g.slice(1, e, 1)                      # gate column e -> (tokens, 1)
            expert_out = self.w2[e](_gelu(self.w1[e](x)))
            weighted = _mul(ge, expert_out)           # broadcast (tokens,1)*(tokens,d)
            out = weighted if out is None else _add(out, weighted)
        return out

    def parameters(self):
        ps = list(self.gate.parameters())
        for e in range(self.num_experts):
            ps += list(self.w1[e].parameters()) + list(self.w2[e].parameters())
        return ps

    def __call__(self, x):
        return self.forward(x)
