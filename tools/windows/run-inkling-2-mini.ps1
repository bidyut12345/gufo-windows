# Serve Inkling-2-Mini on Windows with DSpark speculative decoding and vision (mmproj).
#
#   powershell -ExecutionPolicy Bypass -File tools\windows\run-inkling-2-mini.ps1
#   powershell -ExecutionPolicy Bypass -File tools\windows\run-inkling-2-mini.ps1 -Think off
#   powershell -ExecutionPolicy Bypass -File tools\windows\run-inkling-2-mini.ps1 -HostAddr 0.0.0.0
param(
  [string]$ModelDir = "C:\AI\model-Inkling-2-Mini-Uncensored-AD-Q4_K",
  [string]$Model = "",
  [string]$DsparkModel = "",
  [string]$Mmproj = "",
  [int]$Context = 32768,
  [int]$Sessions = 1,
  [ValidateSet("on", "off")]
  [string]$Think = "on",
  [string]$HostAddr = "127.0.0.1",
  [int]$Port = 8080,
  [string]$ServedName = "inkling-2-mini",
  [double]$Temperature = 1.0,
  [double]$TopP = 0.95,
  [int]$TopK = 40,
  [double]$MinP = 0.05,
  [double]$RepeatPenalty = 1.15,
  [int]$RepeatLastN = 128,
  [double]$FrequencyPenalty = 0.1,
  [double]$PresencePenalty = 0.3,
  [switch]$NoDspark,
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

if (-not (Test-Path $ModelDir)) {
  throw "Model directory not found: $ModelDir"
}

if (-not $Model) {
  $preferred = Join-Path $ModelDir "Inkling-2-Mini-Uncensored-AD-Q4_K.gguf"
  if (Test-Path $preferred) {
    $Model = $preferred
  } else {
    $found = Get-ChildItem -LiteralPath $ModelDir -Filter "*.gguf" -File |
      Where-Object { $_.Name -notmatch "^(dspark|mmproj)-" } |
      Select-Object -First 1
    if ($found) { $Model = $found.FullName }
  }
}
if (-not $Model -or -not (Test-Path $Model)) {
  throw "Target GGUF model not found in '$ModelDir'."
}

if (-not $NoDspark -and -not $DsparkModel) {
  $found = Get-ChildItem -LiteralPath $ModelDir -Filter "dspark-*.gguf" -File | Select-Object -First 1
  if ($found) { $DsparkModel = $found.FullName }
}
if (-not $NoDspark -and (-not $DsparkModel -or -not (Test-Path $DsparkModel))) {
  throw "DSpark draft GGUF not found in '$ModelDir' (pass -NoDspark to run without it)."
}

if (-not $NoVision -and -not $Mmproj) {
  $found = Get-ChildItem -LiteralPath $ModelDir -Filter "mmproj-*.gguf" -File | Select-Object -First 1
  if ($found) { $Mmproj = $found.FullName }
}
if (-not $NoVision -and (-not $Mmproj -or -not (Test-Path $Mmproj))) {
  throw "Vision mmproj GGUF not found in '$ModelDir' (pass -NoVision to run without it)."
}

$invariant = [System.Globalization.CultureInfo]::InvariantCulture
$gufoArgs = @(
  "serve",
  "--host", $HostAddr,
  "--port", "$Port",
  "--sessions", "$Sessions",
  "--max-request-bytes", "33554432",
  "llm",
  "--model", $Model,
  "--served-model-name", $ServedName,
  "--context", "$Context",
  "--max-output-bytes", "8388608",
  "--think", $Think,
  "--temperature", $Temperature.ToString($invariant),
  "--top-p", $TopP.ToString($invariant),
  "--top-k", "$TopK",
  "--min-p", $MinP.ToString($invariant),
  "--repeat-penalty", $RepeatPenalty.ToString($invariant),
  "--repeat-last-n", "$RepeatLastN",
  "--frequency-penalty", $FrequencyPenalty.ToString($invariant),
  "--presence-penalty", $PresencePenalty.ToString($invariant)
)

if (-not $NoDspark) {
  $gufoArgs += @("--speculative", "dspark", "--dspark-model", $DsparkModel)
}
if (-not $NoVision) {
  $gufoArgs += @("--mmproj", $Mmproj)
}

$shown = ($gufoArgs | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join " "
Write-Host "Starting Inkling-2-Mini server ($Context context, $Sessions session):" -ForegroundColor Cyan
Write-Host "`"$gufoExe`" $shown" -ForegroundColor DarkGray
if ($DryRun) { return }

Write-Host "Web Chat:   http://${HostAddr}:${Port}/" -ForegroundColor Cyan
Write-Host "OpenAI API: http://${HostAddr}:${Port}/v1  model `"$ServedName`"" -ForegroundColor Green
& $gufoExe @gufoArgs
