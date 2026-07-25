"""Extra loss functions: binary cross-entropy and Huber (NumPy).

Complements the autograd losses in the C++ core (`mse_loss`, `cross_entropy`). These
NumPy versions compute the loss value and its gradient w.r.t. the prediction, so they
can drive training loops / classical ML. Wiring them into the C++ autograd graph (like
mse_loss/cross_entropy) is a follow-up (needs a `log` op node in the core).
"""

import numpy as np


def bce_loss(y_true, y_pred, eps=1e-7, from_logits=False):
    """Binary cross-entropy. If from_logits, y_pred are logits (sigmoid applied here)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    if from_logits:
        p = 1.0 / (1.0 + np.exp(-p))
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def bce_grad(y_true, y_pred, from_logits=False):
    """Gradient of BCE w.r.t. y_pred (or logits, which simplifies to sigmoid(x)-y)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    if from_logits:
        p = 1.0 / (1.0 + np.exp(-p))
        return (p - y_true) / y_true.size
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return ((p - y_true) / (p * (1 - p))) / y_true.size


def huber_loss(y_true, y_pred, delta=1.0):
    """Huber loss: quadratic for small errors, linear for large (robust to outliers)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    d = y_pred - y_true
    a = np.abs(d)
    quad = 0.5 * d ** 2
    lin = delta * (a - 0.5 * delta)
    return float(np.mean(np.where(a <= delta, quad, lin)))


def huber_grad(y_true, y_pred, delta=1.0):
    d = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    g = np.where(np.abs(d) <= delta, d, delta * np.sign(d))
    return g / d.size
