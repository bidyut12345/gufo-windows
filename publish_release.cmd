@echo off
setlocal
cd /d "%~dp0"
echo ========================================================
echo  Gufo Windows 1-Click GitHub Release Publisher
echo ========================================================
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\windows\publish_release.ps1" %*
if errorlevel 1 (
  echo.
  echo Release publishing failed with error code %errorlevel%.
  pause
  exit /b %errorlevel%
)
echo.
echo Release published successfully!
pause
