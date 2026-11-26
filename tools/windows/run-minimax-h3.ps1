# Serve MiniMax H3 text-to-video-and-audio on Windows together with the
# Video Studio WebUI.
#
#   powershell -ExecutionPolicy Bypass -File tools\windows\run-minimax-h3.ps1
#   powershell -ExecutionPolicy Bypass -File tools\windows\run-minimax-h3.ps1 -NoWebUi
param(
  [switch]$Ref2VA,
  [string]$ModelDir = "",
  [string]$JobsRoot = "",
  [string]$ManifestPath = "",
  [string]$ModelName = "minimax-h3",
  [string]$BindHost = "127.0.0.1",
  [string]$WebHost = "0.0.0.0",
  [int]$Port = 8080,
  [int]$BackendPort = 8081,
  [int64]$Ttl = 3153600000,
  [int]$MaxRequestBytes = 268435456,
  [switch]$NoWebUi,
  [switch]$NoBrowser
)

if (-not $ModelDir) {
  $ModelDir = if ($Ref2VA) { "C:\AI\gufo-video-models-ref2va\models\MiniMax-H3" } else { "C:\AI\gufo-video-models\models\MiniMax-H3" }
}
if (-not $JobsRoot) {
  $JobsRoot = if ($Ref2VA) { "C:\AI\gufo-video-models-ref2va\video-jobs" } else { "C:\AI\gufo-video-models\video-jobs" }
}
if (-not $ManifestPath) {
  $ManifestPath = if ($Ref2VA) { "C:\AI\gufo-video-models-ref2va\MINIMAX_H3_REF2VA_BF16.source-manifest.json" } else { "C:\AI\gufo-video-models\MINIMAX_H3_FL2VA_BF16.source-manifest.json" }
}

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
if (-not (Test-Path (Join-Path $ModelDir "model_index.json"))) {
  throw "MiniMax H3 model directory (with model_index.json) not found at '$ModelDir'."
}

New-Item -ItemType Directory -Force -Path $JobsRoot | Out-Null

# Ensure ffmpeg.exe is on PATH for MP4 muxing.
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
  $ffCandidates = @(
    (Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links\ffmpeg.exe")
  ) + @(Get-ChildItem -Path (Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages\Gyan.FFmpeg*") -Recurse -Filter "ffmpeg.exe" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName)
  $ffExe = $ffCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
  if ($ffExe) {
    $env:PATH = "$(Split-Path $ffExe);$env:PATH"
  } else {
    Write-Warning "ffmpeg.exe not found in PATH; MP4 video output will fail unless ffmpeg is installed."
  }
}

# Ensure LF-normalized manifest so SHA-256 matches compiled contract on Windows.
if (-not (Test-Path $ManifestPath)) {
  $manifestName = if ($Ref2VA) { "MINIMAX_H3_REF2VA_BF16.source-manifest.json" } else { "MINIMAX_H3_FL2VA_BF16.source-manifest.json" }
  $srcManifest = Join-Path $repoRoot "src\models\minimax_h3\$manifestName"
  if (Test-Path $srcManifest) {
    $bytes = [System.IO.File]::ReadAllBytes($srcManifest)
    $text = [System.Text.Encoding]::UTF8.GetString($bytes).Replace("`r`n", "`n")
    [System.IO.File]::WriteAllText($ManifestPath, $text, (New-Object System.Text.UTF8Encoding($false)))
  }
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
  "video",
  "--model", $ModelDir,
  "--served-model-name", $ModelName,
  "--manifest", $ManifestPath,
  "--root", $JobsRoot,
  "--ttl", "$Ttl"
)
$loraSearchRoots = @(
  (Join-Path (Split-Path $JobsRoot -Parent) "loras"),
  (Join-Path (Split-Path $ModelDir -Parent) "loras"),
  (Join-Path (Split-Path (Split-Path $ModelDir -Parent) -Parent) "loras")
) | Select-Object -Unique
$seenLoras = @{}
foreach ($loraRoot in $loraSearchRoots) {
  if (-not (Test-Path $loraRoot)) { continue }
  foreach ($loraFile in Get-ChildItem -Path $loraRoot -Recurse -Filter *.safetensors -ErrorAction SilentlyContinue) {
    $loraName = ($loraFile.BaseName -replace '[^A-Za-z0-9_.-]', '_')
    if ($seenLoras.ContainsKey($loraName)) { continue }
    $seenLoras[$loraName] = $true
    $gufoArgs += @("--lora", "`"${loraName}=$($loraFile.FullName)`"")
  }
}

Write-Host "Starting Gufo video server: $gufoExe $($gufoArgs -join ' ')"
$gufoProc = Start-Process -FilePath $gufoExe -ArgumentList $gufoArgs -NoNewWindow -PassThru

$webProc = $null
try {
  if (-not $NoWebUi) {
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCmd) {
      throw "python not found in PATH; pass -NoWebUi or install Python 3."
    }
    $webuiScript = Join-Path $PSScriptRoot "video-webui.py"
    $webArgs = @(
      $webuiScript,
      "--host", $WebHost,
      "--port", "$Port",
      "--backend", "http://${BindHost}:${gufoPort}",
      "--model", $ModelName,
      "--save-dir", (Join-Path $JobsRoot "saved"),
      "--parent-pid", "$($gufoProc.Id)"
    )
    $webProc = Start-Process -FilePath $pythonCmd.Source -ArgumentList $webArgs -NoNewWindow -PassThru
    $webUrl = "http://127.0.0.1:${Port}/"
    Show-WebUiUrls -WebHost $WebHost -Port $Port -Label "Video WebUI"
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
