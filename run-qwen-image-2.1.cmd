@echo off
rem Start Gufo Qwen-Image-2.1 (base 40-step) server and Image Studio WebUI.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\windows\run-qwen-image-2.1.ps1" %*

