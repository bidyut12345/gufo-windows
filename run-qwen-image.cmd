@echo off
rem Start Gufo Qwen-Image-2.1-Turbo server and Image Studio WebUI.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\windows\run-qwen-image-turbo.ps1" %*
