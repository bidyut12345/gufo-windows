@echo off
rem Start Gufo MiniMax H3 video server and Video Studio WebUI.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\windows\run-minimax-h3.ps1" %*
