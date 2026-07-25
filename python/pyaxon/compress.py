"""Storage compression: int8 quantization + gzip.

Two independent, composable tools used to shrink the knowledge base on disk:

- **int8 quantization** of float matrices (embeddings, LSA components): each row is
  scaled so its largest magnitude maps to 127, then stored as signed bytes plus one
  float scale per row. Reconstruction is `code * scale`. This is ~4x smaller than
  float32 with negligible cosine-similarity loss (the vectors are L2-normalized).
- **gzip** of the JSON payload: text chunks dominate the size and compress ~4-5x.

Unlike the C++ `quantize_int8` op (which returns a dequantized tensor for compute),
here we keep the raw int8 codes -- that is what makes the file small.
"""

import base64
import gzip
import json

import numpy as np


def quantize_int8(matrix):
    """Quantize a 2D float matrix to int8 with a per-row scale.

    Returns a JSON-friendly dict: {shape, scale (per row), codes (base64 int8)}.
    Reconstruct with `dequantize_int8`. Per-row scaling keeps rows of very
    different magnitude accurate (scale = max|row| / 127)."""
    mat = np.ascontiguousarray(matrix, dtype=np.float32)
    if mat.ndim != 2:
        raise ValueError("quantize_int8 expects a 2D matrix")
    scale = np.abs(mat).max(axis=1) / 127.0
    scale[scale == 0] = 1.0                              # all-zero row -> avoid /0
    codes = np.round(mat / scale[:, None]).astype(np.int8)
    return {
        "shape": list(mat.shape),
        "scale": [float(s) for s in scale],
        "codes": base64.b64encode(codes.tobytes()).decode("ascii"),
    }


def dequantize_int8(d):
    """Reconstruct the float32 matrix from `quantize_int8`'s dict."""
    rows, cols = d["shape"]
    codes = np.frombuffer(base64.b64decode(d["codes"]), dtype=np.int8).reshape(rows, cols)
    scale = np.asarray(d["scale"], dtype=np.float32)[:, None]
    return codes.astype(np.float32) * scale


def save_json_gz(obj, path):
    """Serialize `obj` to JSON and gzip it to `path` (utf-8)."""
    raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    with gzip.open(path, "wb") as f:
        f.write(raw)


def load_json_gz(path):
    """Load a gzip-compressed JSON file written by `save_json_gz`."""
    with gzip.open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))
