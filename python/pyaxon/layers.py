"""Extra NN layers. Conv2d wraps the autograd C++ `conv2d` op, so it TRAINS via
`.backward()` like the other layers (Linear, Embedding, ...).
"""

import numpy as np

from ._axon import conv2d as _conv2d
from ._axon import from_numpy


class Conv2d:
    """Trainable 2D convolution. Input (N, Cin, H, W) -> (N, Cout, Hout, Wout).

    Weights/bias are pyaxon Tensors with requires_grad -- pass `parameters()` to an
    optimizer and train with the C++ autograd (`conv2d` backward computes dx/dw/db).
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, seed=0):
        self.stride = stride
        self.padding = padding
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))  # He init
        w = rng.normal(0, scale, (out_channels, in_channels, kernel_size, kernel_size))
        self.weight = from_numpy(np.ascontiguousarray(w, dtype=np.float32))
        self.weight.requires_grad_(True)
        self.bias = from_numpy(np.zeros(out_channels, dtype=np.float32))
        self.bias.requires_grad_(True)

    def forward(self, x):
        return _conv2d(x, self.weight, self.bias, self.stride, self.padding)

    def parameters(self):
        return [self.weight, self.bias]

    def __call__(self, x):
        return self.forward(x)
