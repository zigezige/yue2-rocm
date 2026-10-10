# ROCm port of t8's core runtime installer.
#
# Why this file exists
# --------------------
# t8's scripts/setup.ps1 builds three embedded Python interpreters and installs
# PyTorch from https://download.pytorch.org/whl/cu128 -- i.e. it is NVIDIA-only by
# construction. Everything else in t8 is device-agnostic: the only CUDA-shaped code
# on the generation path is
#     if not torch.cuda.is_available(): raise ...
#     if not torch.cuda.is_bf16_supported(): raise ...
# and on ROCm both reach the HIP device through the torch.cuda alias and both
# return True, so those gates pass unchanged.
#
# So the port keeps t8's embedded-interpreter layout exactly as designed -- its app
# expects runtime/core/python.exe at the interpreter root, which `python -m venv`
# does not provide -- but installs ROCm PyTorch from AMD's TheRock nightly index
# instead of the CUDA index. The only source change anywhere in t8 is the attention
# backend fix in vendor/yue2/cuda_graph.py (see patch_cuda_graph_rocm.py).
#
# Installs use uv, not pip. Two reasons, both learned the hard way here:
#   1. AMD's `rocm` package ships as an sdist, so installing it needs setuptools.
#      A python.org embeddable interpreter has no setuptools, and plain pip fails
#      with "Cannot import 'setuptools.build_meta'". uv builds sdists in an
#      isolated env and pulls its own build deps.
#   2. Once the ROCm torch install silently fails, pip's resolver will happily go
#      fetch a CUDA torch wheel from PyPI to satisfy `accelerate`. The assertions
#      at the end of this script exist to make that failure loud instead of silent.
#
# Scope: core runtime only, which is all song generation needs. The transcribe
# (SheetSage2/MERT) and voice (Seed-VC/Demucs) runtimes are optional; without them
# t8 reports those capabilities as unavailable.
param(
    [switch]$Force,
    [string]$PythonVersion = '3.12.10'
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# The embeddable interpreter enables user site-packages via its `import site`, which
# would silently pull random packages from %APPDATA%\Python. Keep it hermetic.
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONUTF8 = '1'

$KitRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Runtime = Join-Path $KitRoot 'runtime'
$Core = Join-Path $Runtime 'core'
$Downloads = Join-Path $KitRoot 'downloads'
New-Item -ItemType Directory -Force $Downloads, $Runtime | Out-Null

function Step($text) { Write-Host "`n=== $text ===" -ForegroundColor Cyan }

# ------------------------------------------------- 1. embedded Python 3.12
Step "[1/6] embeddable Python $PythonVersion"
$CorePython = Join-Path $Core 'python.exe'
if ($Force -or -not (Test-Path -LiteralPath $CorePython)) {
    if (Test-Path -LiteralPath $Core) {
        # Remove junctions with rmdir first so a linked target is never recursively
        # deleted, then clear the real files.
        Get-ChildItem -LiteralPath $Core -Force | Where-Object { $_.LinkType } | ForEach-Object {
            & cmd.exe /c rmdir "$($_.FullName)" | Out-Null
        }
        Remove-Item -LiteralPath $Core -Recurse -Force
    }
    $Archive = Join-Path $Downloads "python-$PythonVersion-embed-amd64.zip"
    if (-not (Test-Path -LiteralPath $Archive)) {
        $Mirrors = @(
            "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip",
            "https://mirrors.huaweicloud.com/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip",
            "https://registry.npmmirror.com/-/binary/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip",
            "https://mirrors.aliyun.com/python-release/windows/python-$PythonVersion-embed-amd64.zip"
        )
        $ok = $false
        foreach ($u in $Mirrors) {
            Write-Host "  trying $($u.Split('/')[2])"
            # Timeouts matter: python.org is unreachable from some networks and an
            # unbounded curl would hang instead of falling through to the next mirror.
            & curl.exe -L --fail --connect-timeout 10 --max-time 120 --retry 1 `
                -o "$Archive.partial" $u 2>$null
            if ($LASTEXITCODE -eq 0 -and (Test-Path "$Archive.partial") -and
                (Get-Item "$Archive.partial").Length -gt 1MB) {
                Move-Item "$Archive.partial" $Archive -Force; $ok = $true; break
            }
            Remove-Item "$Archive.partial" -Force -ErrorAction SilentlyContinue
        }
        if (-not $ok) { throw 'could not download the embeddable Python archive' }
    }
    New-Item -ItemType Directory -Force $Core | Out-Null
    Expand-Archive -LiteralPath $Archive -DestinationPath $Core -Force
    $Short = ($PythonVersion.Split('.')[0..1] -join '')
    $Pth = Join-Path $Core "python$Short._pth"
    $Lines = Get-Content -LiteralPath $Pth | Where-Object {
        $_ -ne '#import site' -and $_ -ne 'import site' -and $_ -ne 'Lib\site-packages' -and $_ -ne '..\..'
    }
    # '..\..' lets the embedded interpreter import the kit itself (t8 relies on it).
    @($Lines + 'Lib\site-packages' + '..\..' + 'import site') | Set-Content -LiteralPath $Pth -Encoding ascii
}
Write-Host "  $(& $CorePython --version 2>&1)"

# --------------------------------------------------------------- 2. uv + pip
Step '[2/6] bootstrapping pip and uv'
if (-not (Test-Path -LiteralPath (Join-Path $Core 'Scripts\pip.exe'))) {
    $GetPip = Join-Path $Downloads 'get-pip.py'
    if (-not (Test-Path -LiteralPath $GetPip)) {
        & curl.exe -L --fail --connect-timeout 10 --max-time 120 --retry 2 -o $GetPip 'https://bootstrap.pypa.io/get-pip.py'
        if ($LASTEXITCODE -ne 0) { throw 'get-pip.py download failed' }
    }
    & $CorePython -X utf8 $GetPip --no-warn-script-location
}
& $CorePython -m pip install --upgrade --quiet uv
& $CorePython -m uv --version

# --------------------------------------------------- 3. ROCm runtime libraries
Step '[3/6] ROCm runtime libraries (gfx1201) -- AMD TheRock'
& $CorePython -m uv pip install --python $CorePython --system --link-mode copy `
    --extra-index-url https://rocm.nightlies.amd.com/v4/whl/ --pre 'rocm[libraries,device-gfx1201]'

# ----------------------------------------- 4. ROCm PyTorch (NOT the CUDA wheel)
Step '[4/6] ROCm PyTorch -- AMD TheRock gfx120X-all index'
& $CorePython -m uv pip install --python $CorePython --system --link-mode copy `
    --index-url https://rocm.nightlies.amd.com/v2/gfx120X-all/ torch

# ------------------------------------------------------------- 5. t8 core deps
Step '[5/6] t8 core requirements'
& $CorePython -m uv pip install --python $CorePython --system --link-mode copy `
    -r (Join-Path $KitRoot 'requirements-core.txt')

$FfmpegDir = Join-Path $Runtime 'ffmpeg'
New-Item -ItemType Directory -Force $FfmpegDir | Out-Null
$Ffmpeg = (& $CorePython -X utf8 -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())' | Out-String).Trim()
if (Test-Path -LiteralPath $Ffmpeg) {
    Copy-Item -LiteralPath $Ffmpeg -Destination (Join-Path $FfmpegDir 'ffmpeg.exe') -Force
    Write-Host "  ffmpeg bundled -> $FfmpegDir\ffmpeg.exe"
}

# --------------------------------------------------------------- 6. verify
Step '[6/6] verifying the ROCm runtime'
& $CorePython -X utf8 -c @"
import torch, transformers, numpy, soundfile
assert torch.version.hip is not None, (
    'this is a CUDA build of PyTorch, not ROCm: torch.version.hip is None. '
    'The ROCm install silently fell back to PyPI -- re-run step 4.')
assert torch.version.cuda is None, 'CUDA build detected; expected a HIP build'
print('torch       ', torch.__version__)
print('hip         ', torch.version.hip)
print('cuda field  ', torch.version.cuda, '(must be None)')
assert torch.cuda.is_available(), 'no HIP device visible'
assert torch.cuda.is_bf16_supported(), 'device does not support BF16'
print('device      ', torch.cuda.get_device_name(0))
print('capability  ', torch.cuda.get_device_capability(0))
print('transformers', transformers.__version__)
print('numpy       ', numpy.__version__)
print('OK: ROCm core runtime is ready')
"@

Write-Host "`nCore runtime ready: $CorePython" -ForegroundColor Green
Write-Host 'Next: run patch_cuda_graph_rocm.py against vendor\yue2\cuda_graph.py, then start_rocm.bat' -ForegroundColor Green
