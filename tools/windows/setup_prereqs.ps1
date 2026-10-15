# Setup script for Gufo build prerequisites on Windows (Strix Halo gfx1151)
$ErrorActionPreference = "Stop"

Write-Host "=== Setting up Gufo Build Prerequisites for Windows ===" -ForegroundColor Cyan

# 1. Ensure directories exist
$rockDir = "C:\TheRock"
$rockBuild = "C:\TheRock\build"
$tarball = "therock-dist-windows-gfx1151-10.0.0.tar.gz"
$tarballPath = Join-Path $rockDir $tarball
$url = "https://stable.repo.amd.com/rocm/core/tarball/$tarball"

if (-not (Test-Path $rockDir)) {
    New-Item -ItemType Directory -Path $rockDir -Force | Out-Null
}
if (-not (Test-Path $rockBuild)) {
    New-Item -ItemType Directory -Path $rockBuild -Force | Out-Null
}

# 2. Download TheRock ROCm 10.0.0 if not already extracted
$versionFile = Join-Path $rockBuild ".info\version"
if (-not (Test-Path $versionFile)) {
    if (-not (Test-Path $tarballPath) -or (Get-Item $tarballPath).Length -lt 1600000000) {
        Write-Host "Downloading TheRock ROCm 10.0.0 (~1.57 GB)..." -ForegroundColor Yellow
        & curl.exe -L -C - --retry 5 -o $tarballPath $url
        if ($LASTEXITCODE -ne 0) { throw "Failed to download TheRock tarball" }
    }
    Write-Host "Extracting TheRock ROCm 10.0.0 to $rockBuild..." -ForegroundColor Yellow
    & tar.exe -xzf $tarballPath -C $rockBuild
    if ($LASTEXITCODE -ne 0) { throw "Failed to extract TheRock tarball" }
    Write-Host "TheRock ROCm 10.0.0 extracted successfully." -ForegroundColor Green
} else {
    Write-Host "TheRock ROCm is already present at $rockBuild." -ForegroundColor Green
}

# 3. Setup vcpkg
$vcpkgDir = "C:\vcpkg"
if (-not (Test-Path $vcpkgDir)) {
    Write-Host "Cloning vcpkg into $vcpkgDir..." -ForegroundColor Yellow
    & git clone https://github.com/microsoft/vcpkg.git $vcpkgDir
    if ($LASTEXITCODE -ne 0) { throw "Failed to clone vcpkg" }
}

$vcpkgExe = Join-Path $vcpkgDir "vcpkg.exe"
if (-not (Test-Path $vcpkgExe)) {
    Write-Host "Bootstrapping vcpkg..." -ForegroundColor Yellow
    Push-Location $vcpkgDir
    try {
        & .\bootstrap-vcpkg.bat -disableMetrics
        if ($LASTEXITCODE -ne 0) { throw "Failed to bootstrap vcpkg" }
    } finally {
        Pop-Location
    }
}

# 4. Install required libraries
Write-Host "Installing vcpkg packages (icu, curl, openssl, libpng, libjpeg-turbo:x64-windows)..." -ForegroundColor Yellow
& $vcpkgExe install icu curl openssl libpng libjpeg-turbo --triplet x64-windows
if ($LASTEXITCODE -ne 0) { throw "Failed to install vcpkg packages" }

Write-Host "=== All prerequisites are installed! ===" -ForegroundColor Green
