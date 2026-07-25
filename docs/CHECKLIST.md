# pyaxon — Build Checklist

A complete list of everything that will be done, in order. Mark `[x]` as you finish.
Each "hot-path" item is only considered done with a **test** and, when applicable,
a **benchmark**.

Legend: 🔥 = hot path (requires test + benchmark) · 🧪 = needs a test · 📏 = needs a benchmark

> **Hardware/SDK-dependent items** (real GPU: working CUDA, OpenCL, Vulkan,
> cuBLAS/cuDNN) were made **optional** — build flags off, with the structure
> ready (see [`CUDA.md`](CUDA.md)). Everything that can be done with CPU alone has been implemented.

---

## Phase 0 — Project setup ✅
- [x] Directory structure (`include/`, `src/`, `bindings/`, `python/`, `tests/`, `benchmarks/`, `examples/`, `docs/`)
- [x] Root `CMakeLists.txt` (C++20, optimization flags `-O3 -march=native`)
- [x] `.gitignore` (build/, __pycache__, *.so, .venv)
- [x] `LICENSE` (MIT)
- [x] gtest integration (C++ tests) — 65 tests passing
- [x] pytest integration (Python tests) — 21 tests passing
- [x] CI (GitHub Actions: build + tests + benchmarks)
- [x] `.clang-format` and style standard
- [x] Build README (how to build on Windows/Linux) — `docs/BUILD.md`

## Phase 1 — Core: Tensor and memory ✅
- [x] `Storage` (raw buffer, owns the data, `shared_ptr`) 🧪
- [x] `DType` (F32, F64, I32) and `Device` (CPU, CUDA)
- [x] `Tensor`: shape, strides, offset 🧪
- [x] Constructors: `zeros`, `ones`, `full`, `from_data`, `randn`, `rand` 🧪
- [x] Copy-free views: `reshape`, `transpose`, `permute`, `slice` 🧪
- [x] Indexing and element access (`at`)
- [x] `contiguous()` (materializes a view)
- [x] Tensor printing/`repr`
- [x] Broadcasting (NumPy-style rules) 🧪 — see Phase 1.5

## Phase 1.5 — Broadcasting and reductions (done)
- [x] Elementwise broadcasting (NumPy-style rules) 🧪
- [x] `sum_to` (reduces the gradient to the original shape — undoes broadcasting) 🧪
- [x] `randn` (random init with a fixed seed), `clone`, `item`

## Phase 2 — Operations (CPU kernels)
- [x] Elementwise with broadcasting: `add`, `sub`, `mul`, `neg`, `mul_scalar` 🔥
- [x] Reduction: `sum_all`, `mean_all`, `sum/mean/max/min_axis` (per axis) 🧪
- [x] `matmul` — 5 variants (naive/ikj/blocked/omp/avx2), validated against each other 🔥📏
  - [x] cache-friendly ikj + auto-vectorization (`-O3 -march=native`)
  - [x] blocking/tiling (L1/L2)
  - [x] OpenMP (multithread) — up to **26.5×** vs. naive
  - [x] explicit AVX2 SIMD (FMA) — up to **23.8×** vs. naive
  - [x] explicit **AVX-512** SIMD (guarded by CPU; falls back to OMP if unavailable) 🧪
  - [x] benchmark measuring GFLOP/s and speedup ([`BENCHMARKS.md`](BENCHMARKS.md))
- [x] Per-axis reduction: `sum_axis`, `mean_axis`, `max_axis`, `min_axis` 🧪
- [x] Activations: `relu`, `sigmoid`, `tanh`, `gelu` 🧪
- [x] `softmax` / `log_softmax` (numerically stable, log-sum-exp) 🧪
- [x] `relu` / `relu_mask` 🧪
- [x] Validation: every fast kernel compared to the naive one (tolerance) 🧪

## Phase 3 — Autograd (automatic differentiation) ✅
- [x] `Node` base (stores inputs, defines `backward`)
- [x] `requires_grad`, `grad`, `grad_fn` on the Tensor (meta shared via `shared_ptr`)
- [x] Graph construction in the forward pass (`axon::fn` layer)
- [x] `backward()` with topological ordering 🧪
- [x] Gradient accumulation (`grad += ...`, an input used twice accumulates) 🧪
- [x] Backward for: add(broadcast), sub, mul, matmul, relu, sum, mean 🧪
- [x] Backward for: sigmoid, tanh, gelu, softmax, cross_entropy 🧪
- [x] `no_grad()` / `NoGradGuard` (graph-free context, for inference) 🧪
- [x] **Numerical gradient check** (finite differences) 🧪 — covers all activations
- [x] `zero_grad`

## Phase 4 — nn module (in progress)
- [x] `Module` base (`forward`, `parameters`, `operator()`)
- [x] `Linear` (with bias) 🧪
- [x] Weight initialization: He, **Xavier**, **Uniform** (`nn.Init`) 🧪
- [x] Activations as Modules: `ReLU`, `Sigmoid`, `Tanh`, `GELU` 🧪
- [x] `Sequential`
- [x] `LayerNorm` (with ε, fused + backward) 🧪 — grad check on x and gamma
- [x] `Embedding` (scatter-add in the backward) 🧪 — grad accumulates on a repeated index
- [x] `Dropout` (training scales by 1/(1-p); inference = identity) 🧪
- [x] `RNN` (Elman) and `LSTM` (nn modules, backprop through time) 🧪
- [x] Losses: `mse_loss`, `cross_entropy` (stable, log-sum-exp) 🧪
- [x] **Milestone: train XOR end to end** ✅ (loss ~0, predictions 0/1/1/0)
- [x] **Milestone: multiclass classifier** ✅ ([`examples/classify.py`](../examples/classify.py), 100% accuracy on 3 blobs)

## Phase 5 — Optimizers ✅ (base)
- [x] `Optimizer` base (`step`, `zero_grad`)
- [x] `SGD` (with momentum and **weight decay**) 🧪
- [x] `Adam` / **AdamW** (bias correction + decoupled weight decay) 🧪
- [x] `set_lr`/`lr` + **warmup + cosine** scheduler (`lm.cosine_lr`) 🧪

## Phase 6 — Python bindings (pybind11) ✅ (base)
- [x] pybind11 setup (build via `scripts/build_python.ps1`) — TODO: integrate into CMake
- [x] Expose `Tensor` (+ operators `+ - * @`, `backward`, `tolist`) 🧪
- [x] Expose `nn` (Module/Linear/ReLU/Sequential), `optim` (SGD/Adam), `mse_loss`
- [x] Python list ⇆ Tensor conversion (`ax.tensor`, `.tolist()`) and **NumPy** (`ax.from_numpy`, `.numpy()`) 🧪
- [x] Python package `pyaxon/` (friendly public API)
- [x] `pyproject.toml` + **installable wheel** (`pip install`, tested in a venv) 🧪
- [x] XOR example running in Python 🧪 (5 pytest tests passing)

## Phase 7 — Tokenizer ✅ (base)
- [x] `Vocab` (token ⇆ id, save/load) 🧪
- [x] BPE (Byte-Pair Encoding): training + encode/decode + round-trip 🧪
- [x] Special tokens `<unk>`, `<bos>`, `<eos>`, `<pad>` (reserved ids) 🧪
- [x] **WordPiece** (subwords with `##`, greedy longest-match) 🧪
- [ ] Priority-optimized merges (heap) — **optional** (perf; current BPE is correct and tested)

## Phase 8 — Transformer ✅
- [x] Stable softmax (base of attention) 🧪
- [x] Scaled dot-product attention 🧪 — grad check
- [x] Causal masking 🧪 — `ops::causal_mask`
- [x] `LayerNorm`, `Embedding` (structural blocks, in Phase 4) 🧪
- [x] Single-head `SelfAttention` (nn module, trains) 🧪
- [x] Multi-head attention (split/concat across heads) 🔥🧪
- [x] Sinusoidal positional encoding 🧪
- [x] Feed-Forward block (Linear→GELU→Linear)
- [x] Transformer block (pre-norm: x+attn(LN(x)), x+FFN(LN(x))) 🧪
- [x] Decoder-only model (**AxonLM**) — memorizes a sequence, logits shape 🧪
- [x] Generation (sampling with temperature) — [`examples/axonlm.py`](../examples/axonlm.py)
- [x] Top-k sampling — [`examples/train_text.py`](../examples/train_text.py)
- [x] Training on larger text with BPE + DataLoader (learns patterns, not just memorizing) ✅
- [x] Stopping at `<eos>` during generation
- [x] **Top-p** (nucleus) sampling + combinable with top-k 🧪

## Phase 9 — Training
- [x] `DataLoader` (`pyaxon.data`: sliding windows, batches, shuffle) 🧪
- [x] Training with gradient accumulation (mini-batch) — [`examples/train_text.py`](../examples/train_text.py)
- [x] Reusable training loop (`lm.train_lm`, with schedule and tokens/s) 🧪
- [x] Weight serialization (`save`/`load`, custom binary format) 🧪
- [x] Model **checkpoints** (round-trip validated in C++ and Python) 🧪
- [x] Real dataset loaders (`load_text`, `iter_text_chunks` streaming, `load_idx`/MNIST) 🧪
- [x] Checkpoints including epoch + **optimizer state** (Adam's m/v via `state_dict`) 🧪
- [x] Logging of loss, **lr**, and **tokens/s** per epoch
- [x] **Real 3D batch** (`forward_batch`: B sequences in one call) 🧪 — ~1.6× faster, validated equal to isolated sequences
- [x] **Milestone: train the AxonLM on text** ✅ (char-level, generates text)

## Phase 10 — Optimized inference
- [x] `no_grad` on the inference path (used in generation) 🧪
- [x] Multithreading (OpenMP) + SIMD (AVX2/AVX-512) in `matmul` 📏
- [x] **KV-cache** (incremental attention) 🧪 — logits identical to the full forward
- [x] **Operation fusion** (`bias_relu`/`bias_gelu` in one pass) 🧪
- [x] **int8 quantization** (symmetric per tensor: quantize/dequantize) 🧪
- [x] SIMD/OpenMP extended to **elementwise** and **reductions** (fast path) 🧪

## Phase 11 — GPU backend — OPTIONAL (depends on hardware/SDK)
- [x] `Device { CPU, CUDA }` enum on the Tensor
- [x] Build options `AXON_ENABLE_CUDA` / `AXON_ENABLE_OPENCL` / `AXON_ENABLE_VULKAN` (OFF by default, don't break the CPU build)
- [x] Reference CUDA kernel (`src/ops/cuda/matmul_cuda.cu`) — skeleton
- [x] `cuda_available()` / `ax.cuda_available()` — support query
- [ ] Tensor allocation on `Device::CUDA` + host ⇆ device copy — **requires GPU** 🔒
- [ ] CUDA/OpenCL/Vulkan kernels + cuBLAS/cuDNN — **requires GPU/SDK** 🔒
- See [`docs/CUDA.md`](CUDA.md). 🔒 = blocked by hardware unavailable on this machine

## Phase 12 — Product / Complete engine
- [x] Examples: XOR, classifier, AxonLM (char-level and text+BPE), **Markov**, **MNIST** 🧪
- [x] Digit recognition: IDX loader (`load_idx`) + MNIST example (runs with local data) 🧪
- [x] **HTTP server in C++** (`examples/cpp_server/`, Winsock, optional target) — tested ✅
- [x] **HTTP server + REST API** (`examples/serve.py`, stdlib only) — `/generate`
- [x] **Web interface** demo (`examples/web/index.html`) — prompt + controls
- [x] Serving documentation ([`docs/SERVING.md`](SERVING.md))
- [x] Tools (`tools/inspect_ckpt.py` — inspects checkpoints)
- [x] **CLI** (`python -m pyaxon train/generate/serve`) 🧪 — [`docs/CLI.md`](CLI.md)
- [x] Model bundle (weights + tokenizer + config) with round-trip between processes 🧪
- [x] `pyaxon` entry point in `pyproject.toml` (after `pip install .`)
- [x] **Wheel** (`pyaxon-0.1.0-cp313-win_amd64.whl`) with `.pyd` + web embedded — installed and tested in a clean venv ✅

---

## Phase 13 — Classic ML + preprocessing (scikit-learn / pandas style) ✅
- [x] NumPy interop (`from_numpy` / `.numpy()`) 🧪
- [x] `pyaxon.pre`: StandardScaler, MinMaxScaler, normalize, one_hot, train_test_split 🧪
- [x] `LinearRegression` (normal equation, R²) 🧪
- [x] `LogisticRegression` (dogfood: nn.Linear + cross_entropy + Adam) 🧪
- [x] `GaussianNB` (Naive Bayes) 🧪
- [x] `KMeans` (k-means++ + n_init) 🧪
- [x] `KNeighborsClassifier` 🧪
- [x] `DecisionTreeClassifier` (CART, Gini) 🧪
- [x] Example `examples/ml_demo.py` + doc [`ML.md`](ML.md)
- [ ] More models (SVM, RandomForest, GradientBoosting) and more of pandas — optional/future

## Quality rail (continuous, all phases)
- [x] Benchmarks recorded with a speedup number on the hot paths ([`BENCHMARKS.md`](BENCHMARKS.md))
- [x] Naive version always available as a correctness oracle (`matmul_naive`) 🧪
- [x] Numerical stability verified (no NaN/Inf) in softmax/log_softmax/cross_entropy 🧪
- [x] CI configured (GitHub Actions build+tests) — [ ] execution verified on the remote runner (needs a push)
- [~] No memory leaks: consistent use of `shared_ptr`/RAII; the `AXON_ENABLE_ASAN` flag
      exists, but **this machine's MinGW does not provide `libasan`** (ASan won't link). Verifiable
      on a toolchain with ASan (Clang/GCC Linux).

---

## Main milestones
1. ✅ **Design docs** — README, ARCHITECTURE, CHECKLIST, HTML
2. ✅ **Tensor + CPU ops** with tests — broadcasting, reductions, matmul, relu
3. ✅ **Autograd + XOR training** — 24 tests green, XOR converges (loss ~0)
4. ✅ **Python bindings** — `import pyaxon`, XOR trains in Python, 5 pytest tests green
5. ✅ **Transformer + AxonLM generating text** — char-level, trains and generates (43 C++ tests green)
6. ✅ **Engine + REST API + web demo** — AxonLM served over HTTP with an interactive page
7. ✅ **Distributable product** — CLI + installable wheel + 3D batch + special tokens
8. 🔧 **CUDA backend** — structure ready and optional (missing NVIDIA hardware)
