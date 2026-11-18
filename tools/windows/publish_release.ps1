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
$releaseMdPath = Join-Path $root "RELEASE.md"
if (Test-Path $releaseMdPath) {
    Copy-Item $releaseMdPath $releaseNotesPath -Force
} else {
    $notesTemplate = @"
# Gufo for Windows $Tag (AMD Strix Halo gfx1151)

High-performance native local LLM inference engine optimized for AMD Ryzen AI MAX+ 395 with Radeon 8060S (`gfx1151`) on Windows 11. Built natively with AMD TheRock ROCm 10.1.0.
"@
    [System.IO.File]::WriteAllText($releaseNotesPath, $notesTemplate, [System.Text.Encoding]::UTF8)
}
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
