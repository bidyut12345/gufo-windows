# Serve Qwen3.8-Flash-Next-Uncensored (IQ4_XS) on Windows with MTP, vision, and disk cache.
#
#   powershell -ExecutionPolicy Bypass -File tools\windows\run-qwen3.8-flash-next-uncensored.ps1
#   powershell -ExecutionPolicy Bypass -File tools\windows\run-qwen3.8-flash-next-uncensored.ps1 -DryRun
param(
  [string]$ModelDir = "C:\AI\gufo-llm-model-qwen-3.8-flash-next-uncensored-iq4-xs",
  [string]$Model = "",
  [string]$MtpModel = "",
  [string]$Mmproj = "",
  [string]$HostAddr = "0.0.0.0",
  [int]$Port = 8080,
  [int]$Sessions = 2,
  [int]$MaxRequestBytes = 33554432,
  [string]$ServedName = "qwen3.8-flash-next-unsloth-q4-k-xl",
  [int]$Context = 262144,
  [double]$Temperature = 0.6,
  [double]$TopP = 0.95,
  [int]$TopK = 20,
  [double]$MinP = 0.0,
  [ValidateSet("on", "off", "auto")]
  [string]$Think = "on",
  [ValidateSet("on", "off", "auto")]
  [string]$PreserveThinking = "off",
  [int64]$MaxOutputBytes = 33554432,
  [int64]$CacheRamBytes = 0,
  [string]$CacheDisk = "C:\AI\gufo-data\cache",
  [int64]$CacheDiskBytes = 68719476736,
  [int64]$CacheDiskStagingBytes = 0,
  [string]$LogLevel = "debug",
  [switch]$NoLogProgress,
  [switch]$NoMtp,
  [switch]$NoVision,
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$gufoExe = @(
  (Join-Path $repoRoot "build\windows-release\gufo.exe"),
  (Join-Path $repoRoot "build\release\gufo.exe")
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $gufoExe) {
  $cmd = Get-Command gufo.exe -ErrorAction SilentlyContinue
  if ($cmd) { $gufoExe = $cmd.Source }
}
if (-not $gufoExe) {
  throw "gufo.exe not found in build\windows-release, build\release, or PATH."
}

if (-not $Model) {
  $Model = Join-Path $ModelDir "Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf"
}
if (-not (Test-Path $Model)) {
  throw "Target GGUF model not found: $Model"
}

if (-not $NoMtp -and -not $MtpModel) {
  $MtpModel = Join-Path $ModelDir "Qwen3.8-Flash-Next-Uncensored-MTP-draft.gguf"
}
if (-not $NoMtp -and -not (Test-Path $MtpModel)) {
  throw "MTP draft GGUF not found: $MtpModel (pass -NoMtp to run without it)."
}

if (-not $NoVision -and -not $Mmproj) {
  $Mmproj = Join-Path $ModelDir "mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf"
}
if (-not $NoVision -and -not (Test-Path $Mmproj)) {
  throw "Vision mmproj GGUF not found: $Mmproj (pass -NoVision to run without it)."
}

if ($CacheDisk -and -not $DryRun) {
  New-Item -ItemType Directory -Force -Path $CacheDisk | Out-Null
}

$invariant = [System.Globalization.CultureInfo]::InvariantCulture
$gufoArgs = @(
  "serve",
  "--host", $HostAddr,
  "--port", "$Port",
  "--sessions", "$Sessions",
  "--max-request-bytes", "$MaxRequestBytes",
  "llm",
  "--model", $Model
)
if (-not $NoVision) {
  $gufoArgs += @("--mmproj", $Mmproj)
}
$gufoArgs += @(
  "--served-model-name", $ServedName,
  "--context", "$Context",
  "--temperature", $Temperature.ToString($invariant),
  "--top-p", $TopP.ToString($invariant),
  "--top-k", "$TopK",
  "--min-p", $MinP.ToString($invariant),
  "--think", $Think,
  "--preserve-thinking", $PreserveThinking
)
if (-not $NoMtp) {
  $gufoArgs += @("--speculative", "mtp", "--mtp-model", $MtpModel)
}
$gufoArgs += @(
  "--max-output-bytes", "$MaxOutputBytes",
  "--cache-ram-bytes", "$CacheRamBytes",
  "--cache-disk", $CacheDisk,
  "--cache-disk-bytes", "$CacheDiskBytes",
  "--cache-disk-staging-bytes", "$CacheDiskStagingBytes"
)
if (-not $NoLogProgress) {
  $gufoArgs += "--log-progress"
}
if ($LogLevel) {
  $gufoArgs += @("--log-level", $LogLevel)
}

$shown = ($gufoArgs | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join " "
Write-Host "Starting Qwen3.8-Flash-Next-Uncensored server ($Context context, $Sessions sessions):" -ForegroundColor Cyan
Write-Host "`"$gufoExe`" $shown" -ForegroundColor DarkGray
if ($DryRun) { return }

& $gufoExe @gufoArgs
