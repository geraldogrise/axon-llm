# pyaxon

> An educational deep learning library, in the style of **PyTorch / TensorFlow**, with a
> **Python API** and a **high-performance C++ core**. Built from scratch, phase by phase,
> to *truly understand* how an AI framework works under the hood — and to practice
> performance optimization (SIMD, multithreading, GPU).

The name comes from **Py** (Python) + **axon** (the axon, the fiber that connects neurons). The goal
is to have Python's ergonomics with C++'s speed, exactly as PyTorch does with
`libtorch`/ATen.

---

## Why this project exists

The goal is **not** to compete with PyTorch. It is to **learn** by building:

- How a **Tensor** is represented in memory and why *layout* matters for performance.
- How **autograd** (automatic differentiation) works with a dynamic graph.
- How layers, optimizers, tokenizers, and Transformers are actually implemented.
- How to extract performance with **SIMD (AVX2/AVX-512)**, **OpenMP**, and later **CUDA**.

This repository is the evolution of the roadmap described in [`plan.md`](./plan.md), reorganized as
a reusable library instead of a loose set of exercises.

---

## Layered architecture

```
┌───────────────────────────────────────────────────────────┐
│  Python layer  (import pyaxon as ax)                       │  ← ergonomics
│  ax.tensor, ax.nn, ax.optim, ax.tokenizer, ax.transformer   │
├───────────────────────────────────────────────────────────┤
│  Bindings  (pybind11)                                       │  ← bridge
├───────────────────────────────────────────────────────────┤
│  C++ core  "libaxon"                                        │  ← performance
│  Tensor · Autograd · Ops · nn · optim · tokenizer · transf. │
├───────────────────────────────────────────────────────────┤
│  Execution backends                                         │  ← hardware
│  CPU (SIMD + OpenMP)  │  GPU (CUDA / future)                 │
└───────────────────────────────────────────────────────────┘
```

**Core idea:** all the heavy lifting (matrix multiplication, forward, backward)
happens in C++. Python only orchestrates. This way you learn both sides: the friendly API
and the fast engine.

---

## Directory structure (target)

```
pyaxon/
├── include/axon/            # public headers of the C++ core
├── src/
│   ├── core/                # Tensor, memory, dtype, device
│   ├── autograd/            # dynamic graph, Function, backward
│   ├── ops/                 # kernels: matmul, add, softmax, relu...
│   │   ├── cpu/             #   CPU + SIMD implementation
│   │   └── cuda/            #   GPU kernels (advanced phases)
│   ├── nn/                  # Linear, ReLU, GELU, LayerNorm, Module
│   ├── optim/               # SGD, Adam
│   ├── tokenizer/           # BPE, WordPiece, vocabulary
│   └── transformer/         # attention, embedding, decoder
├── bindings/                # pybind11 -> Python module "_axon"
├── python/pyaxon/           # Python package (public API)
├── tests/                   # C++ tests (gtest) and Python tests (pytest)
├── benchmarks/              # performance measurement vs. baseline
├── examples/                # XOR, MNIST, mini-GPT...
├── docs/                    # documentation and site (arquitetura.html)
├── CMakeLists.txt
└── README.md
```

---

## Roadmap (from `plan.md` to a library)

| Phase | Deliverable                    | pyaxon module                      |
|------:|--------------------------------|------------------------------------|
| 1    | Fundamentals + tooling         | CMake, CI, repo layout             |
| 2    | Linear algebra / Tensor        | `core/`, `ops/cpu/`                |
| 3    | Autograd + first network       | `autograd/`, `nn/`, `optim/`       |
| 4    | Text processing                | `tokenizer/`                       |
| 5    | Transformers                   | `transformer/`                     |
| 6    | Training (loader, ckpt)        | `optim/`, data utilities           |
| 7    | Optimized inference            | SIMD, OpenMP, CUDA                 |
| 8    | Complete engine + examples     | `examples/`, packaging             |

> The order prioritizes having **something runnable early**: already in Phase 3 you train an XOR end to end.

---

## What it will look like (target API)

```python
import pyaxon as ax
import pyaxon.nn as nn

# Tensors with autograd, like in PyTorch
x = ax.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = ax.tensor([[0.0], [1.0], [1.0], [0.0]])

model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
)

opt = ax.optim.Adam(model.parameters(), lr=0.05)

for epoch in range(1000):
    pred = model(x)
    loss = ax.mse_loss(pred, y)
    opt.zero_grad()
    loss.backward()        # autograd in C++
    opt.step()
```

---

## Performance philosophy

1. **Measure before optimizing.** Every kernel has a benchmark in `benchmarks/`.
2. **Contiguous layout** (row-major) and memory reuse to maximize cache.
3. **Vectorization**: AVX2 → AVX-512 in the `ops/cpu/` kernels.
4. **Parallelism**: OpenMP to split loops across cores.
5. **GPU** (final phases): CUDA with cuBLAS/cuDNN as a comparison reference.

Each optimization comes with a number: "how many times faster did it get?".

---

## Status

✅ **Already works end to end** — pyaxon trains neural networks, in C++ **and** in Python:

- **C++ core** (`libaxon`): Tensor with views/broadcasting, `matmul`, reductions, softmax
- **Dynamic autograd** (PyTorch style) with topological `backward()` and *gradient check*
- **Activations**: `ReLU`, `Sigmoid`, `Tanh`, `GELU`, `softmax` (all with autograd)
- **`nn`**: `Linear`, activations, `Dropout`, `LayerNorm`, `Embedding`, `RNN`, `LSTM`, `MultiHeadAttention`, `TransformerBlock`, `AxonLM`; **`optim`**: `SGD`, `Adam`/`AdamW` (weight decay)
- **Complete Transformer**: multi-head attention + causal masking + positional encoding → **AxonLM** (decoder-only language model that **generates text**)
- **Losses**: `mse_loss`, `cross_entropy` (stable, log-sum-exp)
- **BPE tokenizer**: training + encode/decode + save/load, special tokens `<unk>/<bos>/<eos>/<pad>`; **checkpoints**; **`no_grad`**; **top-k/top-p** sampling
- **Python bindings** (pybind11): `import pyaxon as ax` with a PyTorch-style API
- **Performance**: `matmul` with cache-blocking, SIMD (AVX2/FMA), and OpenMP — up to **~26× faster** than the naive version ([`docs/BENCHMARKS.md`](./docs/BENCHMARKS.md)), with dispatch that avoids threads on small matrices
- **Training**: `pyaxon.data` (windows/batches) + gradient accumulation; top-k sampling
- **3D batch training** (`forward_batch`): several sequences per step (~1.6× faster)
- **CLI + wheel**: `python -m pyaxon train/generate/serve` and `pip install` of the wheel ([`docs/CLI.md`](./docs/CLI.md))
- **Serving**: REST API + web demo in Python ([`examples/serve.py`](./examples/serve.py)) **and an HTTP server in C++** ([`examples/cpp_server/`](./examples/cpp_server), Winsock) — ([`docs/SERVING.md`](./docs/SERVING.md))
- **Optional GPU**: backend structure (CUDA/OpenCL/Vulkan) ready and disabled by default ([`docs/CUDA.md`](./docs/CUDA.md))
- **Inference**: KV-cache (incremental generation), bias+activation fusion, int8 quantization, `no_grad`
- **Interop**: NumPy (`from_numpy`/`.numpy()`); **BPE** and **WordPiece** tokenizers
- **Classic ML** (`pyaxon.ml`): LinearRegression, LogisticRegression, GaussianNB, KMeans, KNN, DecisionTree; **preprocessing** (`pyaxon.pre`): scalers, normalize, one-hot, split — scikit-learn style ([`docs/ML.md`](./docs/ML.md))
- **axon-lang** (compartmentalized): identifies **Portuguese** and routes **text → area → sector → subsector**, activating only that path; real data pipeline (Wikipedia/files/PDF, with dedup and auto-discovery of subsectors) ([`docs/AXON_LANG.md`](./docs/AXON_LANG.md))
- **Tests**: 70 C++ tests (gtest) + 35 Python tests (pytest), all green
- Examples: **XOR**, **3-class classifier** (100% accuracy), **char-level AxonLM** ([`axonlm.py`](./examples/axonlm.py)), **text training with BPE** ([`train_text.py`](./examples/train_text.py)), and **web server** ([`serve.py`](./examples/serve.py))

```python
import pyaxon as ax
x = ax.tensor([[0,0],[0,1],[1,0],[1,1]])
y = ax.tensor([[0],[1],[1],[0]])
model = ax.nn.Sequential([ax.nn.Linear(2,8), ax.nn.ReLU(), ax.nn.Linear(8,1)])
opt = ax.optim.Adam(model.parameters(), lr=0.05)
for _ in range(2000):
    loss = ax.mse_loss(model(x), y)
    opt.zero_grad(); loss.backward(); opt.step()
```

How to build: [`docs/BUILD.md`](./docs/BUILD.md) · design: [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md)
· roadmap: [`docs/CHECKLIST.md`](./docs/CHECKLIST.md) · visual overview: [`docs/architecture.html`](./docs/architecture.html)

**Next:** Transformer (softmax/GELU/LayerNorm/attention) → mini-GPT, and `matmul` optimization (SIMD/OpenMP) with benchmarks.
