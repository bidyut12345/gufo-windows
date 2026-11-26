@echo off
rem Start Gufo LLM server for Qwen3.8-Flash-Next-Uncensored (IQ4_XS) with MTP, vision, 262K context, 2 sessions, and disk cache.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\windows\run-qwen3.8-flash-next-uncensored.ps1" %*
