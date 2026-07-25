# Builds the Python extension _axon.pyd (pybind11 bindings) with MinGW g++.
#
# Requirements: libaxon.a already built (run scripts\build.ps1 first) and the
# pybind11 package installed (pip install pybind11).
#
# Usage: powershell -File scripts\build_python.ps1

$ErrorActionPreference = "Stop"

$MingwBin = "C:\msys64\mingw64\bin"
$env:PATH = "$MingwBin;$env:PATH"

$Root = Split-Path -Parent $PSScriptRoot
$PyRoot = Split-Path -Parent (Get-Command python).Source
$PyBind = python -c "import pybind11; print(pybind11.get_include())"
$PyInc = python -c "import sysconfig; print(sysconfig.get_path('include'))"
$Tag = python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')"
$PyLib = Join-Path $PyRoot "libs\python$Tag.lib"

$LibAxon = Join-Path $Root "build\libaxon.a"
if (-not (Test-Path $LibAxon)) {
    throw "libaxon.a not found. Run scripts\build.ps1 first."
}

$Pyd = Join-Path $Root "python\pyaxon\_axon.pyd"

Write-Host "==> Building _axon.pyd..." -ForegroundColor Cyan
# -fopenmp: libaxon uses OpenMP in matmul (needs libgomp at link time).
# -static-libgcc/-static-libstdc++ reduce dependencies; libgomp-1.dll and
# libwinpthread-1.dll stay dynamic and are found via os.add_dll_directory
# (see python/pyaxon/__init__.py).
$buildArgs = @(
    "-O3", "-std=c++20", "-shared", "-fopenmp",
    "-I", (Join-Path $Root "include"), "-I", $PyBind, "-I", $PyInc,
    (Join-Path $Root "bindings\module.cpp"), $LibAxon, $PyLib,
    "-static-libgcc", "-static-libstdc++",
    "-o", $Pyd
)
& g++ @buildArgs

if (Test-Path $Pyd) {
    Write-Host "==> OK: $Pyd" -ForegroundColor Green
} else {
    throw "Failed to generate _axon.pyd"
}
