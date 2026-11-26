@echo off
rem Start Gufo LLM server for Inkling-2-Mini with DSpark and vision (mmproj), 32K context, 1 session.
rem   run-inkling-2-mini.cmd                 32K context, 1 session, DSpark + vision on 127.0.0.1:8080
rem   run-inkling-2-mini.cmd -Think off      disable thinking mode
rem   run-inkling-2-mini.cmd -HostAddr 0.0.0.0   listen on all interfaces
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\windows\run-inkling-2-mini.ps1" -ModelDir "C:\AI\model-Inkling-2-Mini-Uncensored-AD-Q4_K" %*
