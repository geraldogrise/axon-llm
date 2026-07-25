cmake_minimum_required(VERSION 3.20)

project(pyaxon
    VERSION 0.1.0
    DESCRIPTION "Educational deep learning library (Python API + C++ core)"
    LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Standards and options
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build in Release by default (with -O3). Use -DCMAKE_BUILD_TYPE=Debug for debugging.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(AXON_BUILD_TESTS "Build tests (gtest)" ON)
option(AXON_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(AXON_ENABLE_NATIVE "Enable -march=native (host SIMD)" ON)
option(AXON_ENABLE_OPENMP "Enable OpenMP" ON)
option(AXON_ENABLE_ASAN "Enable AddressSanitizer (debug)" OFF)
option(AXON_ENABLE_CUDA "Enable CUDA backend (requires CUDA Toolkit)" OFF)
option(AXON_USE_CUBLAS "Link cuBLAS for production-speed GPU matmul" OFF)
option(AXON_ENABLE_OPENCL "Enable OpenCL backend (requires OpenCL SDK)" OFF)
option(AXON_ENABLE_VULKAN "Enable Vulkan backend (requires Vulkan SDK)" OFF)
option(AXON_BUILD_CPP_SERVER "Build the C++ HTTP server (example)" OFF)
# Build the Python extension via CMake (used on Linux/Colab; Windows uses
# scripts/build_python.ps1 with MinGW). CUDA/cuBLAS propagate from the axon target,
# so this produces a GPU-enabled _axon.so with one configure line.
option(AXON_BUILD_PYTHON "Build the _axon Python extension (pybind11)" OFF)

# Export compile_commands.json for clangd / tooling.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Main core library: libaxon
# ---------------------------------------------------------------------------
add_library(axon
    src/core/storage.cpp
    src/core/tensor.cpp
    src/core/serialize.cpp
    src/backend/backend.cpp
    src/tokenizer/bpe.cpp
    src/tokenizer/wordpiece.cpp
    src/ops/elementwise.cpp
    src/ops/activation.cpp
    src/ops/softmax.cpp
    src/ops/layernorm.cpp
    src/ops/embedding.cpp
    src/ops/shape_ops.cpp
    src/ops/quantize.cpp
    src/ops/reduce.cpp
    src/ops/conv.cpp
    src/ops/matmul.cpp
    src/autograd/autograd.cpp
    src/autograd/functional.cpp
    src/nn/module.cpp
    src/optim/optim.cpp
)

target_include_directories(axon PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# `axon` is a static library, but it gets linked into the `_axon` Python extension,
# which is a shared object. On ELF platforms every object inside a .so must be
# position-independent, otherwise ld fails with "relocation R_X86_64_PC32 against
# symbol ... can not be used when making a shared object; recompile with -fPIC".
# No-op on Windows, where code is position-independent already.
set_target_properties(axon PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Strict warnings (best practice: treat code with care).
if(MSVC)
    target_compile_options(axon PRIVATE /W4 /permissive-)
else()
    target_compile_options(axon PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(AXON_ENABLE_NATIVE AND NOT MSVC)
    target_compile_options(axon PRIVATE $<$<CONFIG:Release>:-O3 -march=native>)
endif()

if(AXON_ENABLE_OPENMP)
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(axon PUBLIC OpenMP::OpenMP_CXX)
        target_compile_definitions(axon PUBLIC AXON_HAVE_OPENMP=1)
    endif()
endif()

if(AXON_ENABLE_ASAN AND NOT MSVC)
    target_compile_options(axon PUBLIC -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(axon PUBLIC -fsanitize=address)
endif()

# CUDA backend (optional): compiled only when enabled and the toolkit exists.
if(AXON_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        target_sources(axon PRIVATE src/ops/cuda/matmul_cuda.cu)
        target_compile_definitions(axon PUBLIC AXON_HAVE_CUDA=1)
        set_target_properties(axon PROPERTIES CUDA_ARCHITECTURES native)
        # Optional: link cuBLAS for production-speed matmul (-DAXON_USE_CUBLAS=ON).
        if(AXON_USE_CUBLAS)
            target_compile_definitions(axon PRIVATE AXON_USE_CUBLAS=1)
            target_link_libraries(axon PUBLIC cublas)
        endif()
        message(STATUS "pyaxon: CUDA backend ENABLED")
    else()
        message(WARNING "AXON_ENABLE_CUDA=ON but no CUDA compiler was found; continuing CPU-only")
    endif()
endif()

# OpenCL/Vulkan backends: optional, depend on an SDK. Only flagged here;
# the kernel implementation lands when the SDK/driver is available.
if(AXON_ENABLE_OPENCL)
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
        target_link_libraries(axon PUBLIC OpenCL::OpenCL)
        target_compile_definitions(axon PUBLIC AXON_HAVE_OPENCL=1)
        message(STATUS "pyaxon: OpenCL found (backend to be implemented)")
    else()
        message(WARNING "AXON_ENABLE_OPENCL=ON but OpenCL was not found; ignoring")
    endif()
endif()
if(AXON_ENABLE_VULKAN)
    find_package(Vulkan QUIET)
    if(Vulkan_FOUND)
        target_link_libraries(axon PUBLIC Vulkan::Vulkan)
        target_compile_definitions(axon PUBLIC AXON_HAVE_VULKAN=1)
        message(STATUS "pyaxon: Vulkan found (backend to be implemented)")
    else()
        message(WARNING "AXON_ENABLE_VULKAN=ON but Vulkan was not found; ignoring")
    endif()
endif()

# ---------------------------------------------------------------------------
# Python extension (pybind11) -- optional, for Linux/Colab GPU builds.
# ---------------------------------------------------------------------------
if(AXON_BUILD_PYTHON)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(_axon bindings/module.cpp)
    target_link_libraries(_axon PRIVATE axon)
    # Drop _axon.so straight into the package so `import pyaxon` finds it.
    set_target_properties(_axon PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/pyaxon)
    message(STATUS "pyaxon: building Python extension _axon")
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(AXON_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(AXON_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# ---------------------------------------------------------------------------
# Exemplos
# ---------------------------------------------------------------------------
option(AXON_BUILD_EXAMPLES "Compilar exemplos" ON)
if(AXON_BUILD_EXAMPLES)
    add_executable(xor examples/xor.cpp)
    target_link_libraries(xor PRIVATE axon)
endif()

# Servidor HTTP em C++ (opcional). No Windows linka o Winsock (ws2_32).
if(AXON_BUILD_CPP_SERVER)
    add_executable(axon_server examples/cpp_server/server.cpp)
    target_link_libraries(axon_server PRIVATE axon)
    if(WIN32)
        target_link_libraries(axon_server PRIVATE ws2_32)
    endif()
endif()
