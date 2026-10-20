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
$notesContent = @"
# Gufo for Windows $Tag (AMD Strix Halo gfx1151)

High-performance native local LLM inference engine optimized for AMD Ryzen AI MAX+ 395 with Radeon 8060S (\`gfx1151\`) on Windows 11.

## What's New in this Release

### 🚀 Web Portal & Attachment Dashboard
- **New Chat Attachment Landing View**: On a new chat, the empty main viewport directly presents a sleek, interactive dashboard of all context attachment and session tools.
- **7-Item Context & Attachment Popover Menu**:
  1. 🖼️ **Attach Images**: Multimodal vision inputs directly included in conversation context.
  2. 📄 **Attach Files**: Attach source code, markdown, logs, and docs.
  3. 📁 **Per-Conversation Working Directory**: Dedicated workspace per chat (\`UserData/{convId} - {title}\`).
  4. ⚡ **Agent & MCP Tools**: Master toggle with interactive tool group filters (\`fs\`, \`cli\`, \`eval\`, \`web\`, \`diff\`, \`doc\`).
  5. 🎭 **System Prompt Personas**: Preset personas plus custom system prompt creator.
  6. 🧠 **6 Thinking Levels**: \`auto\`, \`minimum\`, \`low\`, \`medium\`, \`high\`, \`xhigh\`.
  7. 🧹 **Context & Compaction**: Real-time context token usage meter with one-click compact.
- **SQLite Server Database**: Multi-device, cross-browser chat persistence at \`%LOCALAPPDATA%\gufo\chat_portal.db\`.
- **9 Built-in System Tools**: Standard llama.cpp host execution tools + external Model Context Protocol (MCP) servers.

### ⚡ Strix Halo Engine & Memory Optimizations
- **KV Cache Quantization for Qwen 3.8 Flash-Next**:
  - \`f16\` (default reference)
  - \`q8\` (blocked 8-bit symmetric quantization with f16 scales)
  - \`fp8\` (E4M3 format with per-32 f16 scales)
  - \`GUFO_KV_V4=1\` (K8V4: 4-bit values, saves 23.5% KV cache VRAM)
  - \`GUFO_KV_Q4=1\` (K4V8: 4-bit keys)
  - \`K4V4\`: Cuts KV footprint from 6.44 GB (\`f16\`) down to 1.81 GB at 262,144 context!
- **Unsloth UD-IQ4_XS Quantization Support**:
  - Full support for \`UD-IQ4_XS\` models mixing IQ3_S/IQ4_XS experts and Q6_K heads.
  - Reduces weights memory from 22-26 GB to 17-19 GB, fitting comfortable within 32 GB, 64 GB, and 96 GB carve-outs.
- **Reduced VRAM and RAM Footprint**:
  - Zero-copy unified memory allocations on Windows 11.
  - Multi-tier paging and recurrent state optimizations.

## Quickstart (Windows 11)

1. Extract \`gufo-windows-$Version.zip\`.
2. Double-click \`start.cmd\` to launch the interactive model manager.
3. Open \`http://localhost:8080/\` in your web browser.
"@

[System.IO.File]::WriteAllText($releaseNotesPath, $notesContent, [System.Text.Encoding]::UTF8)
Write-Host "Wrote release notes to $releaseNotesPath" -ForegroundColor Green

# 5. Publish to GitHub Release
Write-Host "`n[4/4] Publishing release $Tag to GitHub ($Repo)..." -ForegroundColor Yellow
$uploadScript = Join-Path $root "tools\windows\upload_release.ps1"
powershell -ExecutionPolicy Bypass -File $uploadScript -Tag $Tag -Repo $Repo -Files @($zipPath) -Force:$Force

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n==========================================================" -ForegroundColor Green
    Write-Host " Successfully published release $Tag to GitHub!" -ForegroundColor Green
    Write-Host " URL: https://github.com/$Repo/releases/tag/$Tag" -ForegroundColor Cyan
    Write-Host "==========================================================" -ForegroundColor Green
} else {
    Write-Warning "Release upload returned exit code $LASTEXITCODE"
}
