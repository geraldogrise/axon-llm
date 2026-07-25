# Contributing to pyaxon — Best Practices

This document defines the project's quality standard. The general rule: **clear code,
tested and measured**. No "optimizing by guesswork".

## Principles

1. **Correctness before speed.** Every hot-path kernel has a *naive*
   (reference) version, and the fast versions are validated against it.
2. **Nothing lands without a test.** Every public function has a test (gtest for C++, pytest for Python).
3. **Measure performance.** Optimizations come with a benchmark and a *speedup* number.
4. **RAII and clear ownership.** Memory is managed by `shared_ptr`/`unique_ptr`; no raw `new`/`delete`.
5. **`const`-correctness.** Methods that don't mutate are `const`; use `[[nodiscard]]` on getters.
6. **No warnings.** The build uses `-Wall -Wextra -Wpedantic` (`/W4` on MSVC). A warning = a potential bug.

## Code style

- **C++20**, formatted with `clang-format` (config in `.clang-format`, Google base, 4 spaces, 100 columns).
- Headers with `#pragma once`. Namespace `axon` (sub-namespaces: `axon::ops`, `axon::nn`, ...).
- Names: `PascalCase` for types, `snake_case` for functions/variables, `snake_case_` for private members.
- Comments explain **why**, not the obvious.

## Workflow (Git)

- Work on a branch (`feat/...`, `fix/...`), never directly on `main`.
- Small, descriptive commits (imperative: "add matmul with SIMD").
- Open a PR; the CI (build + tests) must pass before merging.

## How to run locally

See [`docs/BUILD.md`](docs/BUILD.md). Summary:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
cd build && ctest --output-on-failure
```

## Definition of Done

An item in the [checklist](docs/CHECKLIST.md) is only `[x]` when:

- [ ] Implemented and compiling without warnings.
- [ ] With a test covering normal cases **and** error cases.
- [ ] If it's a hot path: with a benchmark and a recorded speedup number.
- [ ] Validated against the reference version (when applicable).
- [ ] `clang-format` applied.
