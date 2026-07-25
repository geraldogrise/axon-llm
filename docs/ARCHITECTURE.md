# pyaxon — Architecture Document

This document details **how** each piece of pyaxon will be built, the reasons behind the decisions, and
the points where performance is attacked. It is the technical map that accompanies the roadmap in
[`plan.md`](../plan.md).

---

## 1. Design principles

1. **Thin Python, fat C++.** The Python layer is only ergonomics; all the heavy computation runs in C++.
2. **Dynamic autograd (define-by-run).** The graph is built during the forward pass, like in
   PyTorch — easier to debug than a static graph.
3. **One Tensor, several backends.** The same `Tensor` can live on the CPU or the GPU; the operation
   picks the right kernel based on the `device`.
4. **Performance is a measured feature.** No "I optimized by guesswork": every kernel has a
   benchmark and a numerical comparison.
5. **From simple to complex.** There is always a *naive*, correct version before the
   *fast* version, to serve as a reference (test oracle).

---

## 2. The Tensor

The fundamental building block. An N-dimensional tensor stored in a **contiguous** buffer (row-major).

```cpp
// include/axon/tensor.h  (sketch)
enum class DType { F32, F64, I32 };
enum class Device { CPU, CUDA };

class Tensor {
public:
    std::shared_ptr<Storage> storage;  // raw buffer (owns the data)
    std::vector<int64_t> shape;        // dimensions
    std::vector<int64_t> strides;      // steps per dimension (enables views)
    int64_t offset = 0;
    DType dtype = DType::F32;
    Device device = Device::CPU;

    // autograd
    bool requires_grad = false;
    std::shared_ptr<Tensor> grad;              // accumulated gradient
    std::shared_ptr<Function> grad_fn;         // who created me (graph node)
};
```

**Why `strides`?** They enable *views* (transpose, slice, reshape) without copying data — the same
technique NumPy/PyTorch use. Transposing a matrix becomes just swapping two numbers.

**Storage separate from Tensor:** several tensors (views) can share the same `Storage`.
The buffer is only freed when the last view dies (`shared_ptr`).

---

## 3. Autograd (automatic differentiation)

The heart of a modern framework. Each operation records **how to compute its gradient**.

```
   a ──┐
       ├─► [MatMul] ──► c ──► [ReLU] ──► d ──► loss
   b ──┘
```

- **Forward:** computes the value AND creates a `Function` node storing the necessary inputs.
- **Backward:** starting from the `loss`, traverses the graph in reverse applying the chain rule.

```cpp
// each differentiable operation inherits from Function
struct Function {
    std::vector<std::shared_ptr<Tensor>> inputs;   // for the backward pass
    // given the output's gradient, returns the gradient of each input
    virtual std::vector<Tensor> backward(const Tensor& grad_out) = 0;
};
```

Example — the sum: `c = a + b`. In the backward pass, `grad_a = grad_c` and `grad_b = grad_c` (the sum just
distributes the gradient). Matrix multiplication `C = A·B` has
`grad_A = grad_C · Bᵀ` and `grad_B = Aᵀ · grad_C`.

**Topological order:** `backward()` visits the nodes in the reverse order of creation, accumulating
gradients in `tensor.grad`.

---

## 4. Ops and kernels (where the performance lives)

Each mathematical operation has two or three implementations:

| Level        | Where               | Goal                                  |
|--------------|---------------------|---------------------------------------|
| Naive        | `ops/cpu/naive/`    | Correct and readable (reference)      |
| SIMD         | `ops/cpu/`          | AVX2 / AVX-512, vectorization         |
| Parallel     | `ops/cpu/` + OpenMP | Split across cores                    |
| GPU          | `ops/cuda/`         | CUDA (final phases)                   |

MVP operations: `matmul`, `add`, `mul`, `sum`, `transpose`, `softmax`, `relu`, `gelu`,
`sigmoid`, `layernorm`.

### Case study: `matmul`

1. **Naive:** three nested loops `i,j,k`. Correct, slow.
2. **Cache-aware:** swap the loop order (`i,k,j`) for sequential access on `B`.
3. **Blocking (tiling):** processes blocks that fit in the L1/L2 cache.
4. **SIMD:** loads 8 (AVX2) or 16 (AVX-512) floats per instruction.
5. **OpenMP:** parallelizes the outer loop across cores.

Each step is measured in `benchmarks/matmul_bench.cpp` — the goal is to record the *speedup*.

---

## 5. The `nn` module

Layers composed of parameters (tensors with `requires_grad = true`).

```cpp
struct Module {
    virtual Tensor forward(const Tensor& x) = 0;
    virtual std::vector<Tensor*> parameters() = 0;
};
```

Components: `Linear`, `ReLU`, `GELU`, `Sigmoid`, `LayerNorm`, `Sequential`, `Embedding`.
Loss functions: `mse_loss`, `cross_entropy`.

---

## 6. The `optim` module

Takes the parameters and adjusts them using the gradients computed by autograd.

- **SGD** (with momentum): `w ← w − lr · grad`.
- **Adam:** keeps first- and second-order moving averages of the gradient (more robust).

`opt.zero_grad()` clears gradients; `opt.step()` applies the update.

---

## 7. Tokenizer

Turns text into integer IDs the model understands.

- **Word-level:** splits by whitespace (simple, large vocabulary).
- **BPE (Byte-Pair Encoding):** merges the most frequent byte pairs — this is what GPT uses.
- **WordPiece:** variant used by BERT.
- **Vocabulary:** a `token ⇆ id` map, saved to disk.

---

## 8. Transformer

The blocks that make up a language model (decoder-only, GPT style):

```
tokens → Embedding → (+ Positional Encoding)
       → [ Self-Attention → Add&Norm → FeedForward → Add&Norm ] × N
       → Linear → logits
```

Attention formula:

```
Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V
```

- **Multi-head:** runs several attentions in parallel and concatenates.
- **Positional encoding:** injects the notion of word order.
- **Causal masking:** prevents a token from "seeing the future" (essential in the decoder).

---

## 9. Training and checkpoints

- **DataLoader:** reads datasets, assembles *batches*, shuffles.
- **Checkpoint:** saves `{ weights, epoch, optimizer state }` to disk to resume.
- **Logs:** loss per epoch, time, throughput (tokens/s).

```cpp
struct Checkpoint {
    std::vector<float> weights;
    int epoch;
};
```

---

## 10. Python bindings (pybind11)

pybind11 exposes the C++ classes to Python with operator overloading, so that
`a + b` in Python calls the C++ kernel.

```cpp
// bindings/module.cpp (sketch)
PYBIND11_MODULE(_axon, m) {
    py::class_<Tensor>(m, "Tensor")
        .def("backward", &Tensor::backward)
        .def("__add__", [](const Tensor& a, const Tensor& b){ return a + b; });
}
```

The Python package `pyaxon/` wraps `_axon` in a nice API.

---

## 11. Build and tests

- **CMake** compiles the core, the bindings, and the tests.
- **gtest** for C++, **pytest** for Python.
- Each fast kernel is validated against the naive version (numerical tolerance).
- **CI** runs tests + benchmarks on every commit.

---

## 12. Suggested implementation order

1. `core/Tensor` + `Storage` (CPU, F32) and tests.
2. Naive ops: `add`, `matmul`, `relu`, `sum`.
3. Minimal autograd + end-to-end **XOR** training. ✅ first milestone.
4. `nn.Linear`, `optim.SGD`/`Adam`, `Sequential`.
5. `matmul` optimization (SIMD + OpenMP) with a benchmark.
6. Tokenizer (word-level → BPE).
7. Transformer block + mini-GPT.
8. Training with a real dataset + checkpoints.
9. CUDA backend.

> Golden rule: **nothing lands without a test, and without a benchmark when it's a hot path.**
