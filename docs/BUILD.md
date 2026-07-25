# How to build pyaxon

## Requirements

- C++20 compiler: **g++ ≥ 13** or **clang++ ≥ 16** or **MSVC 2022**.
- **CMake ≥ 3.20** and a generator (**Ninja** recommended, or Make/MSBuild).
- (Optional) OpenMP for parallelism.

## This machine's environment (Windows)

Here the toolchain is set up as follows (reference for what was used):

- **g++/gcc 15.2** from MSYS2 in `C:\msys64\mingw64\bin`
- **CMake 4.4** and **Ninja** installed via `pip` (`python -m pip install cmake ninja`)

For CMake to find the compiler and Ninja, put them on the session's `PATH`:

```powershell
$env:PATH = "C:\msys64\mingw64\bin;" +
            "C:\Users\<you>\AppData\Local\Programs\Python\Python313\Scripts;" +
            $env:PATH
```

> Tip: use the ready-made script [`scripts/build.ps1`](../scripts/build.ps1), which already adjusts the PATH.

## Steps (any platform)

```bash
# 1. Configure (Release = optimized with -O3 -march=native)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

# 2. Build the library + tests
cmake --build build

# 3. Run the tests
cd build && ctest --output-on-failure
```

## Build options (CMake `-D...`)

| Option                 | Default | What it does                                |
|------------------------|:------:|---------------------------------------------|
| `AXON_BUILD_TESTS`     | ON     | Builds the tests (downloads GoogleTest)     |
| `AXON_BUILD_BENCHMARKS`| OFF    | Builds the benchmarks                       |
| `AXON_ENABLE_NATIVE`   | ON     | `-march=native` (uses your CPU's SIMD)      |
| `AXON_ENABLE_OPENMP`   | ON     | Enables OpenMP if available                 |
| `AXON_ENABLE_ASAN`     | OFF    | AddressSanitizer (to hunt memory bugs)      |

Example in debug mode with the sanitizer:

```bash
cmake -S . -B build-debug -G Ninja -DCMAKE_BUILD_TYPE=Debug -DAXON_ENABLE_ASAN=ON
cmake --build build-debug && (cd build-debug && ctest --output-on-failure)
```

## Python extension (pybind11 bindings)

The Python API (`import pyaxon`) is a C++ extension (`_axon.pyd`) built with
pybind11 on top of `libaxon`.

On this machine (Windows, without MSVC), the extension is compiled with MinGW g++ against
the official CPython, with the C++ runtime linked statically (self-contained extension):

```powershell
# 1. Build libaxon (produces build/libaxon.a)
powershell -File scripts\build.ps1

# 2. Install pybind11 and build the _axon.pyd extension
python -m pip install pybind11 pytest
powershell -File scripts\build_python.ps1

# 3. Use / test (point PYTHONPATH at the python/ folder)
$env:PYTHONPATH = "$PWD\python"
python examples\xor.py
python -m pytest tests\test_pyaxon.py -v
```

> On platforms with MSVC or Linux/GCC, the default path `pip install .`
> (via `pyproject.toml`) tends to work directly. The script above exists because
> here the toolchain is MinGW for a CPython compiled with MSVC.
