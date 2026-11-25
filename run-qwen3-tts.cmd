@echo off
rem Start Gufo Qwen3-TTS server and Speech Studio WebUI (reachable on the LAN).
rem   run-qwen3-tts.cmd                        CustomVoice (built-in speakers)
rem   run-qwen3-tts.cmd -Variant VoiceDesign   describe the voice in words
rem   run-qwen3-tts.cmd -Variant Base          clone a voice from a recording
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\windows\run-qwen3-tts.ps1" -ModelLocation "C:\AI\gufo-tts-models" %*
