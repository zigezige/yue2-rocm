@echo off
rem ============================================================
rem  YuE2 CLI launcher - AMD RX 9070 XT / native Windows ROCm
rem  NOTE: keep this file ASCII-only. cmd.exe re-tokenises batch
rem  lines under the active code page, so non-ASCII text here gets
rem  split mid-line and executed as bogus commands.
rem ============================================================
setlocal
chcp 65001 >nul
cd /d "%~dp0"

set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
set FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE
set TORCH_BLAS_PREFER_HIPBLASLT=1
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

if "%~1"=="" (
  echo.
  echo  Usage:
  echo     run_yue2.bat ^<output_dir^>
  echo.
  echo  Example:
  echo     run_yue2.bat outputs\mysong
  echo.
  echo  Change style/lyrics by editing YuE\examples\song.json, or run:
  echo     "%~dp0venv\Scripts\python.exe" "%~dp0yue2_run.py" --help
  echo.
  pause
  exit /b 1
)

echo Output dir: %~1
echo This can take several minutes per minute of audio. Ctrl+C to cancel.
echo.

"%~dp0venv\Scripts\python.exe" "%~dp0yue2_run.py" --output "%~1"

echo.
pause
