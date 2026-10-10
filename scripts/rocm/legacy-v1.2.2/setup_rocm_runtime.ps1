# ROCm port of t8's runtime installer, parameterised by role.
#
# t8's scripts/setup.ps1 builds three embedded interpreters and installs PyTorch
# from https://download.pytorch.org/whl/cu128, i.e. NVIDIA-only by construction.
# Everything else in t8 is device-agnostic: on ROCm, torch.cuda is the HIP alias,
# so the only CUDA-shaped code on the generation path --
#     if not torch.cuda.is_available(): raise ...
#     if not torch.cuda.is_bf16_supported(): raise ...
# -- passes unchanged. So this port keeps t8's embedded-interpreter layout exactly
# (its app expects runtime/<role>/python.exe at the interpreter root, which
# `python -m venv` does not provide) and only swaps the wheel index.
#
# Version note (the one deliberate deviation from upstream pins):
#   t8 pins `torch==2.8.0 torchaudio==2.8.0` for the 3.11 runtimes. AMD's TheRock
#   index has no *stable* torchaudio 2.8.0 for cp311 -- only the prerelease
#   2.8.0a0 built against a much older ROCm. torch and torchaudio must be a
#   matched pair, so transcribe/voice install 2.9.0 for both. core takes the
#   newest torch on the index (no torchaudio needed).
param(
    [Parameter(Mandatory = $true)][ValidateSet('core', 'transcribe', 'voice')][string]$Role,
    [string]$PythonVersion = '',
    [string]$TorchVersion = '',
    [switch]$SkipPlaywright,
    [switch]$Force
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# The embeddable interpreter enables user site-packages via its `import site`, which
# would silently pull packages from %APPDATA%\Python. Keep it hermetic.
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONUTF8 = '1'

$Roles = @{
    # All three roles use the SAME torch build as the core runtime.
    # Rationale (both discovered by breaking on them):
    #  * torch and torchaudio MUST be the same full build, or the ABI mismatches.
    #    uv resolves them independently and happily pairs
    #    torch 2.9.0+rocm7.10.0a20251117 with torchaudio 2.9.0+rocm7.13.0a20260416.
    #  * torch 2.9.0 (a 2025-11 build) needs the `hipsparselt` library, which is
    #    absent from both the rocm 7.10 SDK it pairs with and the v4/whl index
    #    entirely ("Unknown rocm library 'hipsparselt'" at import time).
    #  * rocm 7.13.0a20260416 is the build the core runtime was verified on: its
    #    SDK lists hipsparselt, and torch+torchaudio+torchvision 2.11.0 cp311 all
    #    exist for it. So every role uses that one build.
    core       = @{ Python = '3.12.10'; Torch = '2.11.0+rocm7.13.0a20260416'; Audio = $false; Reqs = 'requirements-core.txt' }
    transcribe = @{ Python = '3.11.9';  Torch = '2.11.0+rocm7.13.0a20260416'; Audio = $true;  Reqs = 'requirements-transcribe.txt' }
    voice      = @{ Python = '3.11.9';  Torch = '2.11.0+rocm7.13.0a20260416'; Audio = $true;  Reqs = 'requirements-voice.txt' }
}
$Cfg = $Roles[$Role]
if (-not $PythonVersion) { $PythonVersion = $Cfg.Python }
if (-not $TorchVersion) { $TorchVersion = $Cfg.Torch }

$KitRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Runtime = Join-Path $KitRoot 'runtime'
$Target = Join-Path $Runtime $Role
$Downloads = Join-Path $KitRoot 'downloads'
New-Item -ItemType Directory -Force $Downloads, $Runtime | Out-Null

function Step($t) { Write-Host "`n=== [$Role] $t ===" -ForegroundColor Cyan }

# PowerShell does not throw when a native command exits non-zero, so a failed pip
# or a crashed torch import would otherwise be reported as success by whoever runs
# this script. Fail loudly instead.
function Assert-LastExit([string]$What) {
    if ($LASTEXITCODE -ne 0) { throw "$What failed with exit code $LASTEXITCODE" }
}

# ------------------------------------------------- 1. embedded interpreter
Step "embeddable Python $PythonVersion"
$TargetPython = Join-Path $Target 'python.exe'
if ($Force -or -not (Test-Path -LiteralPath $TargetPython)) {
    if (Test-Path -LiteralPath $Target) {
        # rmdir junctions first so a linked target is never recursively deleted.
        Get-ChildItem -LiteralPath $Target -Force | Where-Object { $_.LinkType } | ForEach-Object {
            & cmd.exe /c rmdir "$($_.FullName)" | Out-Null
        }
        Remove-Item -LiteralPath $Target -Recurse -Force
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
            # Bounded timeouts: python.org is unreachable on some networks and an
            # unbounded curl would hang instead of falling through to a mirror.
            & curl.exe -L --fail --connect-timeout 10 --max-time 120 --retry 1 `
                -o "$Archive.partial" $u 2>$null
            if ($LASTEXITCODE -eq 0 -and (Test-Path "$Archive.partial") -and
                (Get-Item "$Archive.partial").Length -gt 1MB) {
                Move-Item "$Archive.partial" $Archive -Force; $ok = $true; break
            }
            Remove-Item "$Archive.partial" -Force -ErrorAction SilentlyContinue
        }
        if (-not $ok) { throw "could not download the embeddable Python archive for $Role" }
    }
    New-Item -ItemType Directory -Force $Target | Out-Null
    Expand-Archive -LiteralPath $Archive -DestinationPath $Target -Force
    $Short = ($PythonVersion.Split('.')[0..1] -join '')
    $Pth = Join-Path $Target "python$Short._pth"
    $Lines = Get-Content -LiteralPath $Pth | Where-Object {
        $_ -ne '#import site' -and $_ -ne 'import site' -and $_ -ne 'Lib\site-packages' -and $_ -ne '..\..'
    }
    # '..\..' lets the embedded interpreter import the kit itself (t8 relies on it).
    @($Lines + 'Lib\site-packages' + '..\..' + 'import site') | Set-Content -LiteralPath $Pth -Encoding ascii
}
Write-Host "  $(& $TargetPython --version 2>&1)"

# --------------------------------------------------------------- 2. uv
Step 'bootstrapping pip and uv'
if (-not (Test-Path -LiteralPath (Join-Path $Target 'Scripts\pip.exe'))) {
    $GetPip = Join-Path $Downloads 'get-pip.py'
    if (-not (Test-Path -LiteralPath $GetPip)) {
        & curl.exe -L --fail --connect-timeout 10 --max-time 120 --retry 2 -o $GetPip 'https://bootstrap.pypa.io/get-pip.py'
        if ($LASTEXITCODE -ne 0) { throw 'get-pip.py download failed' }
    }
    & $TargetPython -X utf8 $GetPip --no-warn-script-location
    Assert-LastExit 'get-pip bootstrap'
}
# uv, not pip: AMD's `rocm` package ships as an sdist, and plain pip in an
# embeddable interpreter fails with "Cannot import 'setuptools.build_meta'". uv
# builds sdists in an isolated env with its own build deps.
& $TargetPython -m pip install --upgrade --quiet uv
Assert-LastExit 'uv install'
& $TargetPython -m uv --version

# --------------------------------------------------- 3. ROCm runtime libraries
Step 'ROCm runtime libraries (gfx1201) -- AMD TheRock'
& $TargetPython -m uv pip install --python $TargetPython --system --link-mode copy `
    --extra-index-url https://rocm.nightlies.amd.com/v4/whl/ --pre 'rocm[libraries,device-gfx1201]'
Assert-LastExit 'ROCm runtime libraries install'

# ----------------------------------------- 4. ROCm PyTorch (never the CUDA wheel)
if ($Cfg.Audio) {
    Step "ROCm PyTorch $TorchVersion + torchaudio (matched pair)"
    & $TargetPython -m uv pip install --python $TargetPython --system --link-mode copy `
        --index-url https://rocm.nightlies.amd.com/v2/gfx120X-all/ `
        "torch==$TorchVersion" "torchaudio==$TorchVersion"
    Assert-LastExit 'ROCm torch/torchaudio install'
} else {
    Step 'ROCm PyTorch -- AMD TheRock gfx120X-all index'
    & $TargetPython -m uv pip install --python $TargetPython --system --link-mode copy `
        --index-url https://rocm.nightlies.amd.com/v2/gfx120X-all/ torch
    Assert-LastExit 'ROCm torch install'
}

# ------------------------------------------------------------- 5. role deps
Step "t8 $Role requirements"
$Req = Join-Path $KitRoot $Cfg.Reqs
& $TargetPython -m uv pip install --python $TargetPython --system --link-mode copy `
    -r $Req
Assert-LastExit "t8 $Role requirements install"

if ($Role -eq 'transcribe' -and -not $SkipPlaywright) {
    Step 'playwright chromium (offline score renderer)'
    $env:PLAYWRIGHT_BROWSERS_PATH = Join-Path $Runtime 'playwright'
    & $TargetPython -m playwright install --only-shell chromium
    Assert-LastExit 'playwright chromium install'
}

# --------------------------------------------------------------- 6. verify
Step 'verifying the ROCm runtime'
if ($Cfg.Audio) {
    & $TargetPython -X utf8 -c @"
import torch, torchaudio
assert torch.version.hip is not None, (
    'CUDA build detected: torch.version.hip is None. Re-run with the TheRock index.')
assert torch.version.cuda is None, 'CUDA build detected; expected HIP'
print('torch       ', torch.__version__)
print('torchaudio  ', torchaudio.__version__)
print('hip         ', torch.version.hip, '| cuda field', torch.version.cuda)
assert torch.cuda.is_available(), 'no HIP device visible'
print('device      ', torch.cuda.get_device_name(0))
print('OK: $Role runtime is a working ROCm build')
"@
Assert-LastExit "$Role runtime verification (torch import / HIP)"
} else {
    & $TargetPython -X utf8 -c @"
import torch, transformers, numpy, soundfile
assert torch.version.hip is not None, (
    'CUDA build detected: torch.version.hip is None. Re-run with the TheRock index.')
assert torch.version.cuda is None, 'CUDA build detected; expected HIP'
print('torch       ', torch.__version__)
print('hip         ', torch.version.hip, '| cuda field', torch.version.cuda)
assert torch.cuda.is_available(), 'no HIP device visible'
assert torch.cuda.is_bf16_supported(), 'device does not support BF16'
print('device      ', torch.cuda.get_device_name(0))
print('OK: $Role runtime is a working ROCm build')
"@
Assert-LastExit "$Role runtime verification (torch import / HIP)"
}

Write-Host "`n$Role runtime ready: $TargetPython" -ForegroundColor Green
