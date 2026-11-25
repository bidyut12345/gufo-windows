@echo off
rem Download the three Qwen3-TTS 12Hz 1.7B variants Gufo supports into C:\AI\gufo-tts-models.
setlocal
set "DEST=C:\AI\gufo-tts-models\models"
if not exist "%DEST%" mkdir "%DEST%"
for %%V in (CustomVoice VoiceDesign Base) do (
  echo Downloading Qwen/Qwen3-TTS-12Hz-1.7B-%%V ...
  hf download Qwen/Qwen3-TTS-12Hz-1.7B-%%V --local-dir "%DEST%\Qwen3-TTS-12Hz-1.7B-%%V" || exit /b 1
)
if not exist "C:\AI\gufo-tts-models\voices" mkdir "C:\AI\gufo-tts-models\voices"
echo Done. Put reference WAVs (+ same-name .txt transcripts) in C:\AI\gufo-tts-models\voices for the Base variant.
