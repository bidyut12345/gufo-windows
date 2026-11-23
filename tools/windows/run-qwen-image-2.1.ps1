# Serve Qwen-Image-2.1 (base 40-step checkpoint) on Windows together with the
# Image Studio WebUI for generation and editing.
#
#   powershell -ExecutionPolicy Bypass -File tools\windows\run-qwen-image-2.1.ps1
#   powershell -ExecutionPolicy Bypass -File tools\windows\run-qwen-image-2.1.ps1 -NoWebUi
param(
  [string]$ModelLocation = "D:\gufo-image-models",
  [string]$ModelName = "qwen-image-2.1",
  [string]$BindHost = "127.0.0.1",
  [int]$Port = 8080,
  [int]$BackendPort = 8081,
  [int]$MaxRequestBytes = 33554432,
  [switch]$NoWebUi,
  [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

& (Join-Path $PSScriptRoot "run-qwen-image-turbo.ps1") `
  -ModelLocation $ModelLocation `
  -ModelName $ModelName `
  -BindHost $BindHost `
  -Port $Port `
  -BackendPort $BackendPort `
  -MaxRequestBytes $MaxRequestBytes `
  -NoWebUi:$NoWebUi `
  -NoBrowser:$NoBrowser

