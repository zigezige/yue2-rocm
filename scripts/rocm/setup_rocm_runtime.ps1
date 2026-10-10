# ROCm counterpart of scripts/setup_unified.ps1.
#
# Upstream builds one embedded CPython 3.12 at runtime\python.exe and installs
# torch==2.10.0+cu128 from https://download.pytorch.org/whl/cu128, which is
# NVIDIA-only by construction. Everything else in the kit is device agnostic: on
# ROCm torch.cuda is the HIP alias, so the only CUDA-shaped checks on the
# generation path --
#     if not torch.cuda.is_available(): raise ...
#     if not torch.cuda.is_bf16_supported(): raise ...
# -- already pass on a HIP build (gfx1201 reports BF16 support natively).
#
# This script reproduces upstream's layout and verification contract and swaps
# only the wheel source:
#
#   * python-3.12.10-embed-amd64.zip          same archive, same sha256
#   * AMD TheRock ROCm runtime libraries      rocm[libraries,device-<Device>]
#   * Windows torch + torchaudio + gfx1030    rocm.nightlies.amd.com/v4/whl/
#   * every other pin from requirements-unified.lock.txt, with the
#     --extra-index-url line and the torch/torchaudio lines stripped
#
# Two things it does that upstream's installer cannot, because both are ROCm
# specific:
#
#   * uv, not pip. AMD's `rocm` package is an sdist and an embedded interpreter
#     has no setuptools, so pip fails with "Cannot import 'setuptools.build_meta'".
#     uv builds sdists in an isolated environment with their own build deps.
#   * a final assert that torch.version.cuda is None. When the ROCm wheel install
#     fails, pip silently falls back to the PyPI CUDA wheel and everything still
#     looks installed; this assertion is what catches that.
#
# ASCII only, deliberately: cmd.exe/PowerShell consoles under a non-UTF-8 code
# page mangle non-ASCII here, and a mangled path is much harder to debug than a
# missing translation.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\rocm\setup_rocm_runtime.ps1
#   ... -Device gfx1030 -TorchVersion 2.9.1+rocm7.15.0a20260715 -Force
param(
    [string]$Device = 'gfx1030',
    [string]$TorchVersion = '2.9.1+rocm7.15.0a20260715',
    [string]$TorchAudioVersion = '2.9.0+rocm7.15.0a20260715',
    [string]$PythonVersion = '3.12.10',
    [string]$RocmIndex = 'https://rocm.nightlies.amd.com',
    [switch]$SkipPlaywright,
    [switch]$SkipRequirements,
    [switch]$Force
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# The embeddable interpreter honours user site-packages via its `import site`,
# which would silently pull packages from %APPDATA%\Python. Keep it hermetic.
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

$SourceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$RocmPackageVersion = ($TorchVersion -split '\+rocm', 2)[1]
$Runtime = Join-Path $SourceRoot 'runtime'
$Downloads = Join-Path $SourceRoot 'downloads'
New-Item -ItemType Directory -Force $Downloads | Out-Null

function Step([string]$Text) { Write-Host "`n=== $Text ===" -ForegroundColor Cyan }

# PowerShell does not throw when a native command exits non-zero, so a failed pip
# or a crashed torch import would otherwise be reported as success by whoever
# runs this script. Fail loudly instead.
function Assert-LastExit([string]$What) {
    if ($LASTEXITCODE -ne 0) { throw "$What failed with exit code $LASTEXITCODE" }
}

# python.org is unreachable on some networks and an unbounded curl would hang
# rather than fall through to the next mirror.
function Get-FirstReachable([string[]]$Urls, [string]$Destination) {
    if (Test-Path -LiteralPath $Destination) { return }
    foreach ($url in $Urls) {
        Write-Host "  trying $(([uri]$url).Host)"
        & curl.exe --silent --show-error -L --fail --connect-timeout 10 --max-time 300 --retry 1 -o "$Destination.partial" $url
        if ($LASTEXITCODE -eq 0 -and (Test-Path "$Destination.partial") -and
            (Get-Item "$Destination.partial").Length -gt 1MB) {
            Move-Item "$Destination.partial" $Destination -Force
            return
        }
        Remove-Item "$Destination.partial" -Force -ErrorAction SilentlyContinue
    }
    throw "could not download $Destination from any mirror"
}

# --------------------------------------------------------- 1. interpreter
Step "embeddable Python $PythonVersion"
$TargetPython = Join-Path $Runtime 'python.exe'
if ($Force -or -not (Test-Path -LiteralPath $TargetPython)) {
    if (Test-Path -LiteralPath $Runtime) {
        # rmdir junctions first so a linked target is never recursively deleted.
        Get-ChildItem -LiteralPath $Runtime -Force | Where-Object { $_.LinkType } | ForEach-Object {
            & cmd.exe /c rmdir "$($_.FullName)" | Out-Null
        }
        Remove-Item -LiteralPath $Runtime -Recurse -Force
    }
    $Archive = Join-Path $Downloads "python-$PythonVersion-embed-amd64.zip"
    Get-FirstReachable @(
        "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip",
        "https://mirrors.huaweicloud.com/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip",
        "https://registry.npmmirror.com/-/binary/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip",
        "https://mirrors.aliyun.com/python-release/windows/python-$PythonVersion-embed-amd64.zip"
    ) $Archive
    New-Item -ItemType Directory -Force $Runtime | Out-Null
    Expand-Archive -LiteralPath $Archive -DestinationPath $Runtime -Force
    $Short = ($PythonVersion.Split('.')[0..1] -join '')
    $Pth = Join-Path $Runtime "python$Short._pth"
    # '..' is the kit root itself; the app imports both `app.*` and `vendor.*`.
    @("python$Short.zip", '.', 'Lib\site-packages', '..', 'import site') |
        Set-Content -LiteralPath $Pth -Encoding ascii
}
Write-Host "  $(& $TargetPython --version 2>&1)"

# ------------------------------------------------------------ 2. pip + uv
Step 'bootstrapping pip and uv'
$GetPip = Join-Path $Downloads 'get-pip.py'
Get-FirstReachable @('https://bootstrap.pypa.io/get-pip.py') $GetPip
& $TargetPython -X utf8 $GetPip --no-warn-script-location
Assert-LastExit 'get-pip bootstrap'
& $TargetPython -m pip install --upgrade --quiet uv
Assert-LastExit 'uv install'
& $TargetPython -m uv --version

function Invoke-Uv([string[]]$UvArgs) {
    & $TargetPython -m uv pip install --python $TargetPython --system --link-mode copy @UvArgs
    Assert-LastExit ("uv pip install " + ($UvArgs -join ' '))
}

# ------------------------------------------------- 3. ROCm runtime libraries
Step "ROCm runtime libraries ($Device) -- AMD TheRock"
Invoke-Uv @('--extra-index-url', "$RocmIndex/v4/whl/", '--pre',
    "rocm[libraries,device-$Device]==$RocmPackageVersion")

# -------------------------------------------------- 4. ROCm torch/torchaudio
# AMD publishes native Windows wheels in the v4 flat index. The package
# versions are intentionally different: the verified pair is torch 2.9.1 and
# torchaudio 2.9.0 from the same ROCm build date. torchaudio declares an
# unversioned torch dependency, so uv resolves it to the explicit pin below.
Step "ROCm PyTorch $TorchVersion + torchaudio $TorchAudioVersion"
if ($Device -notmatch '^gfx1030$') {
    throw "This Windows prebuilt path is verified only for gfx1030 (RX 6800 XT). For another GPU, pass a separately verified wheel set and update this guard."
}
$TorchWheelIndex = "$RocmIndex/v4/whl/"
$TorchWheelDeviceName = "amd_torch_device_$Device"
$TorchWheelLinks = @(
    ("$TorchWheelIndex" + 'torch/'),
    ("$TorchWheelIndex" + 'torchaudio/'),
    ("$TorchWheelIndex" + 'amd-torch-device-' + $Device + '/'),
    ("$TorchWheelIndex" + 'rocm/')
)
$WheelNames = @(
    "torch-$($TorchVersion.Replace('+', '%2B'))-cp312-cp312-win_amd64.whl",
    "torchaudio-$($TorchAudioVersion.Replace('+', '%2B'))-cp312-cp312-win_amd64.whl",
    "$TorchWheelDeviceName-$($TorchVersion.Replace('+', '%2B'))-cp312-cp312-win_amd64.whl"
)
foreach ($wheel in $WheelNames) {
    try {
        $null = Invoke-WebRequest -UseBasicParsing -Method Head -Uri ($TorchWheelIndex + $wheel) -TimeoutSec 20
    } catch {
        throw "Required Windows ROCm wheel is unavailable: $TorchWheelIndex$wheel"
    }
}
Invoke-Uv @('--extra-index-url', $TorchWheelIndex,
    '--find-links', $TorchWheelLinks[0], '--find-links', $TorchWheelLinks[1],
    '--find-links', $TorchWheelLinks[2], '--find-links', $TorchWheelLinks[3],
    '--pre', "torch==$TorchVersion",
    "torchaudio==$TorchAudioVersion", "amd-torch-device-$Device==$TorchVersion",
    "rocm==$RocmPackageVersion", "rocm-sdk-core==$RocmPackageVersion",
    "rocm-sdk-libraries==$RocmPackageVersion", "rocm-sdk-device-$Device==$RocmPackageVersion")

if (-not $SkipRequirements) {
    # ---------------------------------------------------- 5. the rest of the pins
    Step 'remaining pins from requirements-unified.lock.txt'
    $Lock = Join-Path $SourceRoot 'requirements-unified.lock.txt'
    if (-not (Test-Path -LiteralPath $Lock)) { throw "missing $Lock" }
    # The lock pins torch==2.10.0+cu128 and carries the cu128 --extra-index-url.
    # Dropping those lines keeps every other pin byte-identical to upstream's
    # verified set while leaving the ROCm wheels we just installed alone.
    $Filtered = Join-Path $Downloads 'requirements-unified.rocm.txt'
    Get-Content -LiteralPath $Lock | Where-Object {
        $line = $_.Trim()
        $line -ne '' -and $line -notlike '#*' -and
        $line -notlike '--extra-index-url*' -and
        $line -notmatch '^torch(audio|-stoi)?\s*=='
    } | Set-Content -LiteralPath $Filtered -Encoding utf8
    Write-Host "  $(($(Get-Content -LiteralPath $Filtered) | Measure-Object).Count) pins (torch lines removed)"
    Invoke-Uv @('-r', $Filtered)
}

# --------------------------------------------------------- 6. renderer
if (-not $SkipPlaywright) {
    Step 'playwright chromium (offline score renderer)'
    $env:PLAYWRIGHT_BROWSERS_PATH = Join-Path $Runtime 'playwright'
    & $TargetPython -m playwright install --only-shell chromium
    Assert-LastExit 'playwright chromium install'
}

# --------------------------------------------------------- 7. verify
Step 'verifying the ROCm runtime'
& $TargetPython -X utf8 -c @"
import torch
assert torch.version.hip is not None, (
    'CUDA build detected: torch.version.hip is None. The ROCm wheel install '
    'failed and pip fell back to the PyPI CUDA build; re-run with -Force.')
assert torch.version.cuda is None, 'CUDA build detected; expected a HIP build'
print('torch       ', torch.__version__)
print('hip         ', torch.version.hip, '| cuda field:', torch.version.cuda)
assert torch.cuda.is_available(), 'no HIP device visible to torch'
assert torch.cuda.is_bf16_supported(), 'device does not report BF16 support'
print('device      ', torch.cuda.get_device_name(0))
print('capability  ', torch.cuda.get_device_capability(0))
print('OK: runtime is a working ROCm build')
"@
Assert-LastExit 'ROCm runtime verification (torch import / HIP)'

Write-Host "`nROCm runtime ready: $TargetPython" -ForegroundColor Green
Write-Host "Next: python scripts\rocm\apply_rocm_port.py ." -ForegroundColor Green
