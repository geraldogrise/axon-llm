# pyaxon — Benchmarks

> "No optimizing by guesswork." Every hot-path optimization comes with a
> number. Run it yourself: `matmul_bench.exe` (build with `-DAXON_BUILD_BENCHMARKS=ON`).

The numbers below are from **one** machine (they vary with CPU, cores, and cache).
What matters is the **speedup relative** to the naive version, and the **method** of getting there.

## matmul — speedup vs. the naive version (ijk)

Reference machine: Windows, g++ 15.2 (MSYS2), `-O3 -march=native`, OpenMP.

| Size           | naive | ikj (cache) | blocked | omp (threads) | avx2 (simd+omp) |
|----------------|:-----:|:-----------:|:-------:|:-------------:|:---------------:|
| 128×128        | 1.0×  | 5.9×        | 6.7×    | 5.1×          | 6.0×            |
| 256×256        | 1.0×  | 12.7×       | 11.3×   | **26.5×**     | 23.8×           |
| 512×512        | 1.0×  | 8.8×        | 6.3×    | 19.1×         | **19.3×**       |

(Measured GFLOP/s: naive ~1.7–3.6; best variants ~33–71 GFLOP/s.)

## What each step does

1. **naive (ijk)** — element-by-element dot product. Accesses `B` column-wise → terrible
   cache use. It's the reference (correctness oracle).
2. **ikj (cache)** — reorders the loops; the inner loop scans `B` and `C`
   sequentially. On its own it already gives ~6–13×, and `-O3 -march=native` **auto-vectorizes** that
   loop (SIMD "for free").
3. **blocked (tiling)** — processes blocks that fit in L1/L2. Helps more on large
   matrices; at medium sizes plain ikj already saturates the bandwidth.
4. **omp (threads)** — parallelizes the rows with OpenMP across cores → the biggest jump.
5. **avx2 (simd+omp)** — explicit SIMD (8 floats/instr. with FMA) + OpenMP.

## An honest reading of the results

- The biggest gain comes from **cache (ikj)** + **multithreading (OpenMP)**.
- **Explicit AVX2** yields little **on top of** omp here: since `-march=native` already
  auto-vectorizes the inner loop, the "scalar" omp version in practice also runs in SIMD.
  On large matrices the bottleneck becomes **memory bandwidth**, not computation — which is
  why GFLOP/s don't grow indefinitely.
- Next steps to go further: block *packing* (BLAS style), registered
  microkernels, and comparison with a GPU backend (CUDA/cuBLAS) in the final phases.

## How to reproduce

```powershell
powershell -File scripts\build.ps1                 # ensures libaxon
$cmake -S . -B build -G Ninja -DAXON_BUILD_BENCHMARKS=ON
$cmake --build build
build\benchmarks\matmul_bench.exe
```
