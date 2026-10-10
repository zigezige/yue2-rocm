@echo off
rem ============================================================
rem  YuE2 Music T8 - local music studio (ROCm / AMD Radeon)
rem
rem  NOTE: keep this file ASCII-only. cmd.exe re-tokenises batch
rem  lines under the active code page, so non-ASCII text here gets
rem  split mid-line and executed as bogus commands, which shows up
rem  as the window closing instantly on a double-click.
rem ============================================================
setlocal
chcp 65001 >nul
cd /d "%~dp0"

set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
set FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE
set TORCH_BLAS_PREFER_HIPBLASLT=1
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONNOUSERSITE=1

set YUE2_HOME=%~dp0
set YUE2_KIT=%~dp0

set "PY=%~dp0runtime\python.exe"
if not exist "%PY%" (
  echo.
  echo  [ERROR] ROCm runtime not found: %PY%
  echo  Build it first:
  echo    powershell -ExecutionPolicy Bypass -File scripts\rocm\setup_rocm_runtime.ps1
  echo.
  pause
  exit /b 1
)

echo ============================================================
echo   YuE2 Music T8  (ROCm / AMD Radeon)
echo   Open in browser:  http://127.0.0.1:8189
echo   The first generation verifies model hashes (1-2 min).
echo   Press Ctrl+C to stop the service.
echo ============================================================
echo.

start "" /b cmd /c "timeout /t 6 >nul & start http://127.0.0.1:8189"
"%PY%" -X utf8 -m app.yue2_app.service --host 127.0.0.1 --port 8189

echo.
echo  [service stopped]
pause
