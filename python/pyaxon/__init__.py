"""pyaxon -- educational deep learning library (Python API + C++ core).

Typical usage::

    import pyaxon as ax

    x = ax.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = ax.tensor([[0], [1], [1], [0]])

    model = ax.nn.Sequential([
        ax.nn.Linear(2, 8), ax.nn.ReLU(), ax.nn.Linear(8, 1),
    ])
    opt = ax.optim.Adam(model.parameters(), lr=0.05)

    for _ in range(2000):
        loss = ax.mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
"""

import os

# The _axon core is built with MinGW (MSYS2); make sure the C++ runtime DLLs are
# found on import (harmless if the path does not exist).
_MINGW_BIN = r"C:\msys64\mingw64\bin"
if os.name == "nt" and os.path.isdir(_MINGW_BIN):
    os.add_dll_directory(_MINGW_BIN)

from . import _axon  # noqa: E402
from contextlib import contextmanager  # noqa: E402

from ._axon import (  # noqa: E402,F401
    BPETokenizer,
    Tensor,
    WordPieceTokenizer,
    add,
    attention,
    backward,
    bce_loss,
    conv2d,
    cross_entropy,
    cuda_available,
    cuda_device_name,
    cuda_matmul_threshold,
    is_cuda_enabled,
    set_cuda_enabled,
    set_cuda_matmul_threshold,
    huber_loss,
    from_numpy,
    gelu,
    is_grad_enabled,
    load,
    log_softmax,
    matmul,
    max_axis,
    mean,
    mean_axis,
    min_axis,
    mse_loss,
    mul,
    ones,
    rand,
    randn,
    relu,
    save,
    set_grad_enabled,
    sigmoid,
    softmax,
    sub,
    sum,
    sum_axis,
    tanh,
    tensor,
    zeros,
)

# Submodules exposed as pyaxon.nn and pyaxon.optim.
nn = _axon.nn
optim = _axon.optim

from . import data  # noqa: E402,F401  (pyaxon.data: windows/batches)
from . import lm  # noqa: E402,F401  (pyaxon.lm: train/generate/save AxonLM)


# pyaxon.pre and pyaxon.ml use NumPy -- loaded on demand to keep the core light
# (importing pyaxon does not require NumPy; only accessing ax.pre / ax.ml does).
_LAZY = ("pre", "ml", "corpus", "langid", "router", "collect", "rag", "embed", "compress",
         "modular", "vindex", "manifest", "generate", "metrics", "model_selection", "moe",
         "losses", "layers", "system")


def __getattr__(name):
    if name in _LAZY:
        import importlib
        mod = importlib.import_module("." + name, __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'pyaxon' has no attribute '{name}'")


@contextmanager
def no_grad():
    """Disable autograd inside the block (faster inference, no graph).

    Usage::

        with ax.no_grad():
            logits = model(x)
    """
    prev = is_grad_enabled()
    set_grad_enabled(False)
    try:
        yield
    finally:
        set_grad_enabled(prev)


__all__ = [
    "Tensor", "tensor", "from_numpy", "zeros", "ones", "randn",
    "add", "sub", "mul", "matmul", "sum", "mean", "rand",
    "sum_axis", "mean_axis", "max_axis", "min_axis",
    "relu", "sigmoid", "tanh", "gelu", "softmax", "log_softmax", "attention",
    "mse_loss", "cross_entropy", "bce_loss", "huber_loss", "conv2d",
    "backward", "no_grad", "set_grad_enabled", "is_grad_enabled",
    "cuda_available", "cuda_device_name", "is_cuda_enabled", "set_cuda_enabled",
    "set_cuda_matmul_threshold", "cuda_matmul_threshold",
    "save", "load", "BPETokenizer", "WordPieceTokenizer",
    "nn", "optim", "data", "lm",
]

__version__ = "0.1.0"
