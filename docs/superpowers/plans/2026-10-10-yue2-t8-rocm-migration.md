# YuE2 T8 ROCm Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the standalone YuE2 v0.1.6 checkout with the `YuE2-T8-ROCm` integrated application and adapt its Windows ROCm installer for the user's AMD Radeon RX 6800 XT.

**Architecture:** The T8 repository becomes the project source tree and supplies the WebUI/ComfyUI integration, vendored YuE2 runtime, Seed-VC, transcription, model manifests, and worker process layout. The ROCm installer remains parameterized, but defaults target the user's `gfx1030` family through AMD's `gfx103X-all` wheel index; verification explicitly rejects CUDA/CPU Torch builds.

**Tech Stack:** Python 3.12 embedded runtime, PyTorch ROCm/TheRock, PowerShell, ComfyUI-style local service, YuE2, Seed-VC, Hugging Face model files.

**Spec:** User-approved migration from standalone YuE2 v0.1.6 to `Newaiguy/YuE2-T8-ROCm`, including T8 WebUI and Seed-VC.

## Global Constraints

- Preserve the current repository's `.git` metadata while replacing tracked source contents.
- Target Windows AMD Radeon RX 6800 XT (`gfx1030`; package family `gfx103X-all`).
- Do not claim GPU generation success from this CPU-only host; provide the exact runtime verification command.
- Keep ROCm and model downloads parameterized so users can override device family and Torch build.
- Keep upstream T8 application files intact except for the AMD architecture/default-index adaptation.

### Task 1: Replace the source tree with T8 ROCm

**Files:**
- Replace: repository working tree contents with the checked-out `YuE2-T8-ROCm` tree, excluding `.git`.
- Preserve: current repository `.git` directory.

- [ ] Copy the complete T8 tree into the current checkout and remove stale standalone YuE2 files.
- [ ] Verify T8 application, vendor, scripts, model manifest, and startup files are present.

### Task 2: Adapt the ROCm installer for RX 6800 XT

**Files:**
- Modify: `scripts/rocm/setup_rocm_runtime.ps1`
- Modify: `start_rocm.bat`
- Create: `scripts/rocm/verify_rocm.py`

- [ ] Change the installer default device family from `gfx1201`/`gfx120X-all` to `gfx1030`/`gfx103X-all`.
- [ ] Keep `-Device`, `-TorchVersion`, and `-RocmIndex` overrides intact.
- [ ] Add a verification script that checks `torch.version.hip`, `torch.version.cuda is None`, device availability, BF16, and reports the detected GPU.
- [ ] Ensure the startup environment enables the ROCm-safe attention settings and points `YUE2_HOME`/`YUE2_KIT` at the checkout.

### Task 3: Validate the migrated package

**Files:**
- Inspect: `README.md`, `docs/ROCM_PORT.md`, `scripts/rocm/verify_capabilities.py`, `models/README.md`.
- Test: repository import/compile checks and available unit tests.

- [ ] Run Python compile checks over the migrated application and vendor code.
- [ ] Run the package's test suite if dependencies permit; report environment failures separately.
- [ ] Run installer/index preflight checks for `gfx103X-all` without downloading the full runtime.
- [ ] Report the exact commands needed on the AMD machine to install runtime, download models, verify manifests, and start the WebUI.
