# Serve Qwen-Image-2.1-Turbo (or another Qwen-Image-2.1 checkpoint) on Windows
# together with the Image Studio WebUI for generation and editing.
#
#   powershell -ExecutionPolicy Bypass -File tools\windows\run-qwen-image-turbo.ps1
#   powershell -ExecutionPolicy Bypass -File tools\windows\run-qwen-image-turbo.ps1 -NoWebUi
#   powershell -ExecutionPolicy Bypass -File tools\windows\run-qwen-image-turbo.ps1 -Port 8080 -BackendPort 8081
param(
  [string]$ModelLocation = "C:\AI\gufo-image-models",
  [string]$ModelName = "qwen-image-2.1-turbo",
  [string]$BindHost = "127.0.0.1",
  [string]$WebHost = "0.0.0.0",
  [int]$Port = 8080,
  [int]$BackendPort = 8081,
  [int]$MaxRequestBytes = 33554432,
  [switch]$NoWebUi,
  [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "lan.ps1")

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$candidates = @(
  (Join-Path $repoRoot "build\windows-release\gufo.exe"),
  (Join-Path $repoRoot "build\release\gufo.exe")
)
$gufoExe = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $gufoExe) {
  $cmd = Get-Command gufo.exe -ErrorAction SilentlyContinue
  if ($cmd) { $gufoExe = $cmd.Source }
}
if (-not $gufoExe) {
  throw "gufo.exe not found in build\windows-release, build\release, or PATH."
}

$modelCandidates = @(
  (Join-Path $ModelLocation "models\$ModelName"),
  (Join-Path $ModelLocation $ModelName),
  "D:\gufo-image-models\$ModelName",
  "D:\gufo-image-models\models\$ModelName",
  $ModelName
)
$modelDir = $modelCandidates | Where-Object { Test-Path (Join-Path $_ "model_index.json") } | Select-Object -First 1
if (-not $modelDir) {
  throw "Image model '$ModelName' (with model_index.json) not found under '$ModelLocation'."
}

function Get-FreePort {
  param([int]$StartPort, [int[]]$Exclude = @())
  $p = $StartPort
  while ($p -lt 65535) {
    if ($Exclude -notcontains $p) {
      $inUse = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue
      if (-not $inUse) {
        try {
          $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $p)
          $listener.Start()
          $listener.Stop()
          return $p
        } catch {}
      }
    }
    $p++
  }
  throw "No free port found starting from $StartPort"
}

$Port = Get-FreePort -StartPort $Port
$gufoPort = if ($NoWebUi) { $Port } else { Get-FreePort -StartPort $BackendPort -Exclude @($Port) }
$gufoArgs = @(
  "serve",
  "--host", $BindHost,
  "--port", "$gufoPort",
  "--max-request-bytes", "$MaxRequestBytes",
  "--no-webui",
  "image",
  "--model", $modelDir,
  "--served-model-name", $ModelName
)
$loraDir = Join-Path $ModelLocation "loras"
if (Test-Path $loraDir) {
  $seenLoras = @{}
  foreach ($loraFile in Get-ChildItem -Path $loraDir -Recurse -Filter *.safetensors -ErrorAction SilentlyContinue) {
    $rawName = ($loraFile.BaseName -replace '[^A-Za-z0-9_.-]', '_')
    if ([string]::IsNullOrEmpty($rawName)) { continue }
    $baseName = if ($rawName.Length -gt 64) { $rawName.Substring(0, 64) } else { $rawName }
    $loraName = $baseName
    $suffixIndex = 2
    while ($seenLoras.ContainsKey($loraName)) {
      $suffix = "-$suffixIndex"
      $prefixLen = [Math]::Min($baseName.Length, 64 - $suffix.Length)
      $loraName = $baseName.Substring(0, $prefixLen) + $suffix
      $suffixIndex++
    }
    $seenLoras[$loraName] = $true
    $gufoArgs += @("--lora", "`"${loraName}=$($loraFile.FullName)`"")
  }
}

Write-Host "Starting Gufo image server: $gufoExe $($gufoArgs -join ' ')"
$gufoProc = Start-Process -FilePath $gufoExe -ArgumentList $gufoArgs -NoNewWindow -PassThru

$webProc = $null
try {
  if (-not $NoWebUi) {
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCmd) {
      throw "python not found in PATH; pass -NoWebUi or install Python 3."
    }
    $webuiScript = Join-Path $PSScriptRoot "image-webui.py"
    $webArgs = @(
      $webuiScript,
      "--host", $WebHost,
      "--port", "$Port",
      "--backend", "http://${BindHost}:${gufoPort}",
      "--model", $ModelName,
      "--parent-pid", "$($gufoProc.Id)"
    )
    $webProc = Start-Process -FilePath $pythonCmd.Source -ArgumentList $webArgs -NoNewWindow -PassThru
    $webUrl = "http://127.0.0.1:${Port}/"
    Show-WebUiUrls -WebHost $WebHost -Port $Port -Label "Image WebUI"
    if (-not $NoBrowser) {
      Start-Process $webUrl | Out-Null
    }
  }

  Wait-Process -Id $gufoProc.Id
} finally {
  if ($webProc -and -not $webProc.HasExited) {
    Stop-Process -Id $webProc.Id -Force -ErrorAction SilentlyContinue
  }
  if ($gufoProc -and -not $gufoProc.HasExited) {
    Stop-Process -Id $gufoProc.Id -Force -ErrorAction SilentlyContinue
  }
}
