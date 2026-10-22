# 1-Click GitHub Release Publisher for Gufo Windows (gfx1151)
# Reads version from version.txt, builds package, and uploads to GitHub release.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File tools\windows\publish_release.ps1 [-Version <version>] [-Force]
param(
    [string]$Version = "",
    [string]$Repo = "bidyut12345/gufo-windows",
    [switch]$Force = $false,
    [switch]$SkipBuild = $false
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot\..\..").Path

# 1. Determine release version
if (-not $Version) {
    $versionFile = Join-Path $root "version.txt"
    if (Test-Path $versionFile) {
        $Version = (Get-Content $versionFile -Raw).Trim()
    } else {
        $Version = "0.18.0"
    }
}

$Tag = if ($Version.StartsWith("v")) { $Version } else { "v$Version" }
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Gufo Windows - 1-Click GitHub Release Publisher" -ForegroundColor Cyan
Write-Host " Target Repository : $Repo" -ForegroundColor White
Write-Host " Target Version    : $Version (Tag: $Tag)" -ForegroundColor White
Write-Host "==========================================================" -ForegroundColor Cyan

# 2. Verify / Build gufo.exe if needed
$buildExe = Join-Path $root "build\release\gufo.exe"
if (-not (Test-Path $buildExe) -and -not $SkipBuild) {
    Write-Host "`n[1/4] Building Gufo Windows release binary..." -ForegroundColor Yellow
    powershell -ExecutionPolicy Bypass -File (Join-Path $root "tools\windows\build.ps1") -Target gufo
    if ($LASTEXITCODE -ne 0) { throw "Build failed" }
} else {
    Write-Host "`n[1/4] Found built release binary: $buildExe" -ForegroundColor Green
}

# 3. Package release into dist/
Write-Host "`n[2/4] Packaging distribution archive..." -ForegroundColor Yellow
$distDir = Join-Path $root "dist"
if (-not (Test-Path $distDir)) { New-Item -ItemType Directory -Path $distDir | Out-Null }

$packageScript = Join-Path $root "tools\windows\package.ps1"
powershell -ExecutionPolicy Bypass -File $packageScript -Version $Version -Out $distDir
if ($LASTEXITCODE -ne 0) { throw "Packaging failed" }

$zipName = "gufo-windows-$Version.zip"
$zipPath = Join-Path $distDir $zipName
if (-not (Test-Path $zipPath)) { throw "Expected package not found at: $zipPath" }

# 4. Generate comprehensive release notes
Write-Host "`n[3/4] Generating release notes..." -ForegroundColor Yellow
$releaseNotesPath = Join-Path $distDir "RELEASE_NOTES.md"
$notesTemplate = @'
# Gufo for Windows {0} (AMD Strix Halo gfx1151)

High-performance native local LLM inference engine optimized for AMD Ryzen AI MAX+ 395 with Radeon 8060S (`gfx1151`) on Windows 11.

## What's New in this Release

### 🔄 Upstream Merged
- Fully merged upstream `gufo-org/gufo` **v0.20.0** release for AMD Strix Halo gfx1151.

### 📊 Console Logger & Real-Time Monitoring
- **Tabular Request Summaries**: Request completion log displays in a clean tabular box format in console.
- **5-Second Interval Progress Updates**: Real-time console progress logged every 5 seconds during long prompt processing and decoding with instantaneous speed (t/s), average speed, and token counters.
- **OpenAI API Speed Exposure**: `prompt_tokens_per_second`, `completion_tokens_per_second`, and `total_tokens_per_second` included in `usage` and `timings` for `/v1/chat/completions`, `/v1/completions`, and streaming terminal chunks.

### 🚀 Web Portal & UI Enhancements
- **Live Generation Speed**: Real-time `eval: X.X t/s | Y tokens` live tracking in the message footer and status bar.
- **Scroll Intent & Stickiness**: Replaced threshold heuristic with user scroll intent detection (`wheel` / `touchmove`) and programmatic scroll guards to prevent autoscroll jumping during fast streaming.
- **New Chat Attachment Landing View & Popover Menu**: Multi-modal image attachments, document uploads, workspace selector, system prompt personas, and 6 thinking levels.
- **Feature Flags**: Supported `--no-webui` and `--no-tools`.

### ⚡ Strix Halo Engine & Memory Optimizations
- **KV Cache Quantization for Qwen 3.8 Flash-Next (`--kv-cache`)**:
  - `--kv-cache f16`: Standard 16-bit float KV cache (default reference, full fidelity).
  - `--kv-cache q8_0` (or `q8`): 8-bit quantized KV cache on both K and V planes (~47% memory reduction).
  - `--kv-cache q8_0-q4_k` (or `K8V4`): Mixed quantization with Q8_0 keys and Q4_K values (~59% memory reduction, cutting KV footprint down to ~1.81 GB at 262k context).
- **Unsloth UD-IQ4_XS Quantization Support**:
  - Full support for `UD-IQ4_XS` models mixing IQ3_S/IQ4_XS experts and Q6_K heads.
- **Reduced VRAM and RAM Footprint**:
  - Zero-copy unified memory allocations on Windows 11.

## Quickstart (Windows 11)

1. Extract `gufo-windows-{1}.zip`.
2. Double-click `start.cmd` to launch the interactive model manager.
3. Open `http://localhost:8080/` in your web browser.
'@
$notesContent = $notesTemplate -f $Tag, $Version

[System.IO.File]::WriteAllText($releaseNotesPath, $notesContent, [System.Text.Encoding]::UTF8)
Write-Host "Wrote release notes to $releaseNotesPath" -ForegroundColor Green

# 5. Publish to GitHub Release
Write-Host "`n[4/4] Publishing release $Tag to GitHub ($Repo)..." -ForegroundColor Yellow
$uploadScript = Join-Path $root "tools\windows\upload_release.ps1"
$commitSha = (& git rev-parse HEAD).Trim()
$uploadParams = @{
    Tag = $Tag
    Repo = $Repo
    Files = @($zipPath)
    TargetCommitish = $commitSha
}
if ($Force) { $uploadParams["Force"] = $true }
& $uploadScript @uploadParams

if ($LASTEXITCODE -eq 0 -or $? -eq $true) {
    Write-Host "`n==========================================================" -ForegroundColor Green
    Write-Host " Successfully published release $Tag to GitHub!" -ForegroundColor Green
    Write-Host " URL: https://github.com/$Repo/releases/tag/$Tag" -ForegroundColor Cyan
    Write-Host "==========================================================" -ForegroundColor Green
} else {
    Write-Warning "Release upload returned exit code $LASTEXITCODE"
}
