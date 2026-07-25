# pyaxon build script (Windows / PowerShell).
# Sets the PATH for this machine's toolchain and builds + runs the tests.
#
# Usage:
#   powershell -File scripts\build.ps1            # Release + tests
#   powershell -File scripts\build.ps1 -Debug     # Debug + AddressSanitizer

param(
    [switch]$Debug,
    [switch]$NoTests
)

$ErrorActionPreference = "Stop"

# --- Toolchain (adjust the paths if your environment differs) ---
$MingwBin = "C:\msys64\mingw64\bin"
$PyScripts = "$env:LOCALAPPDATA\Programs\Python\Python313\Scripts"
$CMake = "$env:LOCALAPPDATA\Programs\Python\Python313\Lib\site-packages\cmake\data\bin\cmake.exe"
$CTest = "$env:LOCALAPPDATA\Programs\Python\Python313\Lib\site-packages\cmake\data\bin\ctest.exe"

$env:PATH = "$MingwBin;$PyScripts;$env:PATH"

$Root = Split-Path -Parent $PSScriptRoot
$BuildType = if ($Debug) { "Debug" } else { "Release" }
$BuildDir = Join-Path $Root ("build-" + $BuildType.ToLower())

$configArgs = @(
    "-S", $Root, "-B", $BuildDir, "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=$BuildType",
    "-DCMAKE_CXX_COMPILER=g++"
)
if ($NoTests) { $configArgs += "-DAXON_BUILD_TESTS=OFF" }
if ($Debug)   { $configArgs += "-DAXON_ENABLE_ASAN=ON" }

Write-Host "==> Configuring ($BuildType)..." -ForegroundColor Cyan
& $CMake @configArgs

Write-Host "==> Building..." -ForegroundColor Cyan
& $CMake --build $BuildDir

if (-not $NoTests) {
    Write-Host "==> Running tests..." -ForegroundColor Cyan
    Push-Location $BuildDir
    & $CTest --output-on-failure
    Pop-Location
}

Write-Host "==> Done." -ForegroundColor Green
