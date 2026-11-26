# Serve Qwen3-TTS 12Hz 1.7B on Windows together with the Speech Studio WebUI.
#
#   powershell -ExecutionPolicy Bypass -File tools\windows\run-qwen3-tts.ps1                      # CustomVoice
#   powershell -ExecutionPolicy Bypass -File tools\windows\run-qwen3-tts.ps1 -Variant VoiceDesign
#   powershell -ExecutionPolicy Bypass -File tools\windows\run-qwen3-tts.ps1 -Variant Base
#   powershell -ExecutionPolicy Bypass -File tools\windows\run-qwen3-tts.ps1 -WebHost 127.0.0.1   # this PC only
#
# Base: WAV files in <ModelLocation>\voices (with optional same-name .txt
# transcripts) are registered as named voices.
param(
  [ValidateSet("CustomVoice", "VoiceDesign", "Base")]
  [string]$Variant = "CustomVoice",
  [string]$ModelLocation = "C:\AI\gufo-tts-models",
  [string]$WebHost = "0.0.0.0",
  [int]$Port = 8070,
  [int]$BackendPort = 8071,
  [int]$Context = 32768,
  [int]$MaxRequestBytes = 33554432,
  [switch]$Npu,
  [switch]$NoWebUi,
  [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "lan.ps1")

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$gufoExe = @(
  (Join-Path $repoRoot "build\windows-release\gufo.exe"),
  (Join-Path $repoRoot "build\release\gufo.exe")
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $gufoExe) {
  $cmd = Get-Command gufo.exe -ErrorAction SilentlyContinue
  if ($cmd) { $gufoExe = $cmd.Source }
}
if (-not $gufoExe) { throw "gufo.exe not found in build\windows-release, build\release, or PATH." }

$dirName = "Qwen3-TTS-12Hz-1.7B-$Variant"
$modelDir = @(
  (Join-Path $ModelLocation "models\$dirName"),
  (Join-Path $ModelLocation $dirName)
) | Where-Object { Test-Path (Join-Path $_ "config.json") } | Select-Object -First 1
if (-not $modelDir) {
  throw "Qwen3-TTS $Variant not found under '$ModelLocation'. Run download-tts-models.cmd first."
}
$modelName = switch ($Variant) {
  "CustomVoice" { "qwen3-tts-12hz-1.7b-customvoice" }
  "VoiceDesign" { "qwen3-tts-12hz-1.7b-voice-design" }
  "Base" { "qwen3-tts-12hz-1.7b-base" }
}

function Get-FreePort {
  param([int]$StartPort, [int[]]$Exclude = @())
  for ($p = $StartPort; $p -lt 65535; $p++) {
    if ($Exclude -contains $p) { continue }
    if (Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue) { continue }
    try {
      $l = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Any, $p)
      $l.Start(); $l.Stop(); return $p
    } catch {}
  }
  throw "No free port found starting from $StartPort"
}

$Port = Get-FreePort -StartPort $Port
# The Gufo backend stays on loopback; the WebUI proxy is what the LAN sees.
$backendHost = if ($NoWebUi) { $WebHost } else { "127.0.0.1" }
$gufoPort = if ($NoWebUi) { $Port } else { Get-FreePort -StartPort $BackendPort -Exclude @($Port) }
$gufoArgs = @(
  "serve", "--host", $backendHost, "--port", "$gufoPort",
  "--max-request-bytes", "$MaxRequestBytes", "--no-webui",
  "tts", "--model", "`"$modelDir`"", "--served-model-name", $modelName,
  "--served-model-name", "qwen3-tts"
)
if ($Context -gt 0) { $gufoArgs += @("--context", "$Context") }
if ($Npu) { $gufoArgs += "--npu" }
if ($Variant -eq "Base") {
  $voiceDir = Join-Path $ModelLocation "voices"
  New-Item -ItemType Directory -Force -Path $voiceDir | Out-Null
  foreach ($wav in Get-ChildItem -Path $voiceDir -Filter *.wav -ErrorAction SilentlyContinue) {
    $gufoArgs += @("--voice", "`"$($wav.BaseName)=$($wav.FullName)`"")
  }
}

$gufoProc = $null
$webProc = $null
try {
  if (-not $NoWebUi) {
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCmd) { throw "python not found in PATH; pass -NoWebUi or install Python 3." }
    $webArgs = @(
      "`"$(Join-Path $PSScriptRoot "tts-webui.py")`"",
      "--host", $WebHost, "--port", "$Port",
      "--backend", "http://127.0.0.1:${gufoPort}",
      "--model", $modelName, "--variant", $Variant,
      "--gufo-exe", "`"$gufoExe`"",
      "--model-location", "`"$ModelLocation`"",
      "--backend-port", "$gufoPort",
      "--context", "$Context",
      "--max-request-bytes", "$MaxRequestBytes",
      "--parent-pid", "$PID"
    )
    if ($Npu) { $webArgs += "--npu" }
    $webProc = Start-Process -FilePath $pythonCmd.Source -ArgumentList $webArgs -NoNewWindow -PassThru
    Show-WebUiUrls -WebHost $WebHost -Port $Port -Label "Speech WebUI"
    if (-not $NoBrowser) { Start-Process "http://127.0.0.1:${Port}/" | Out-Null }
    Wait-Process -Id $webProc.Id
  } else {
    Write-Host "Starting Gufo TTS server ($Variant): $gufoExe $($gufoArgs -join ' ')"
    $gufoProc = Start-Process -FilePath $gufoExe -ArgumentList $gufoArgs -NoNewWindow -PassThru
    if ($WebHost -eq "0.0.0.0") {
      Show-WebUiUrls -WebHost $WebHost -Port $Port -Label "Speech API"
    }
    Wait-Process -Id $gufoProc.Id
  }
} finally {
  if ($webProc -and -not $webProc.HasExited) { Stop-Process -Id $webProc.Id -Force -ErrorAction SilentlyContinue }
  if ($gufoProc -and -not $gufoProc.HasExited) { Stop-Process -Id $gufoProc.Id -Force -ErrorAction SilentlyContinue }
}
