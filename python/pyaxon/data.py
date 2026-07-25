"""Data utilities for language modeling: sliding windows and batches.

For an autoregressive model, each training example is a window of tokens `x` and
its shift by 1 `y` (predict the next token).
"""

import random
import struct


def load_text(path, encoding="utf-8"):
    """Read a whole text file (real dataset: book, article, source code...)."""
    with open(path, "r", encoding=encoding, errors="replace") as f:
        return f.read()


def iter_text_chunks(path, chunk_chars=100000, encoding="utf-8"):
    """Iterate a large file in pieces (streaming, for corpora that do not fit in
    memory at once)."""
    with open(path, "r", encoding=encoding, errors="replace") as f:
        while True:
            chunk = f.read(chunk_chars)
            if not chunk:
                break
            yield chunk


def load_idx(path):
    """Read an IDX file (used by MNIST): images or labels.

    Returns (flat_data, shape). Downloads nothing -- point it at a local .idx.
    """
    with open(path, "rb") as f:
        zero, dtype, ndim = struct.unpack(">HBB", f.read(4))
        assert zero == 0, "invalid IDX file"
        assert dtype == 0x08, "only byte (uint8) IDX is supported"
        dims = [struct.unpack(">I", f.read(4))[0] for _ in range(ndim)]
        n = 1
        for d in dims:
            n *= d
        raw = f.read(n)
        return [float(b) for b in raw], dims


def make_windows(ids, context_len):
    """Generate all sliding (x, y) windows of size `context_len`.

    x = ids[i : i+context_len]     (input)
    y = ids[i+1 : i+context_len+1] (target = next token of each position)
    """
    if context_len < 1:
        raise ValueError("context_len must be >= 1")
    windows = []
    for i in range(len(ids) - context_len):
        windows.append((ids[i:i + context_len], ids[i + 1:i + context_len + 1]))
    return windows


def iter_batches(windows, batch_size, rng=None, shuffle=True):
    """Iterate over the windows in batches of size `batch_size` (shuffled)."""
    order = list(range(len(windows)))
    if shuffle:
        (rng or random).shuffle(order)
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        yield [windows[j] for j in chunk]
