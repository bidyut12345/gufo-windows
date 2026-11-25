@echo off
rem Start Gufo MiniMax H3 Ref2VA FP8/INT8 video server and Video Studio WebUI.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\windows\run-minimax-h3.ps1" -Ref2VA -ModelDir "C:\AI\gufo-video-ref2va-models-int8\models\MiniMax-H3" -ManifestPath "C:\AI\gufo-video-ref2va-models-int8\MINIMAX_H3_REF2VA_FP8.source-manifest.json" -JobsRoot "C:\AI\gufo-video-ref2va-models-int8\video-jobs" %*
