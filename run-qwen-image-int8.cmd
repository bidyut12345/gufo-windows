@echo off
rem Start Gufo Qwen-Image-2.1 INT8/FP8 server and Image Studio WebUI.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\windows\run-qwen-image-2.1.ps1" -ModelLocation "C:\AI\gufo-image-models-int8" %*
