"""Apply every ROCm port change to a YuE2-Music-T8 checkout, idempotently.

Seven changes across six files. Patches 1-5 are required on AMD hardware; 3-5 also
reproduce on NVIDIA with torch >= 2.9/2.11 (see the per-patch notes). Patches 6-7
are kit-level hygiene for a hand-maintained ROCm install.

PATCH 1 -- vendor/yue2/cuda_graph.py : decode-time attention backend
    GraphAR picks its backend from a schema-only probe. On ROCm/HIP
    device.type is still "cuda" (torch.cuda is the HIP alias) and the ATen schema
    is shared across backends, so it selects "flash" -- but ROCm rejects the
    argument that makes varlen FlashAttention correct:
        RuntimeError: [ROCm] mha_varlen_fwd: seqused_k must be nullopt
    Passing seqused_k=None is not a workaround: varlen FA would then attend over
    unused future cache slots and return wrong logits. Force the masked-SDPA
    branch on HIP (verified numerically exact: 0.0 max abs diff vs eager inside a
    captured graph on gfx1201).

PATCH 2 -- app/yue2_app/core_worker.py : VAE tile size
    The pipeline derives vae_core_frames from memory_budget_gib alone
    (512 if <=12 GiB else 1024), which assumes a 24 GiB card. On a 16 GiB Radeon
    the 1024-frame tiles select far slower MIOpen convolution solvers: measured
    313 s of VAE decode versus 156 s with 512-frame tiles for the identical 175 s
    song on an RX 9070 XT. Make the default device-aware below 20 GiB, and let a
    request override it with "vae_core_frames".

PATCH 3 -- voice runtime, audiotools/ml/decorators.py : lazy torch.distributed
    torch >= 2.9 hides dist.ReduceOp until a process group exists, but
    descript-audiotools evaluates it in a class body at import time, so Seed-VC
    cannot start ("AttributeError: module 'torch.distributed' has no attribute
    'ReduceOp'"). Inject a sentinel; Seed-VC is single-process.

PATCH 4 -- vendor/seed-vc/inference.py : CAMPPlus and RMVPE precision
    Both were the only fp32 models in load_models(); fp32 batchnorm makes MIOpen
    JIT-compile a kernel via HIPRTC, which fails on a wheel with no libc++ headers
    ("'type_traits' file not found" -> miopenStatusUnknownError). Run them in fp16
    like every other model there, and cast the fbank features at the call site.

PATCH 5 -- app/yue2_app/voice_worker.py : MIOpen off + torchaudio.save fallback
    Same JIT problem for other fp32 kernels (spatial batchnorm, the GRU in RMVPE),
    so disable the MIOpen backend for this worker (in ROCm PyTorch,
    torch.backends.cudnn IS MIOpen) and let conv/BN/RNN use native kernels.
    Also, torchaudio 2.11 routes .save() through torchcodec, which needs FFmpeg
    shared libraries absent on Windows; substitute soundfile (WAV only).

PATCH 6 -- app/yue2_app/updater.py : never self-update
    The in-app updater re-extracts a release archive over the kit, which would
    replace the ROCm PyTorch runtimes and wipe every patch above. Report the kit
    as up to date unconditionally.

PATCH 7 -- device wording (index.html + three worker messages)
    The UI tells the user inference runs on an "NVIDIA GPU" and the self-check
    errors say "no NVIDIA CUDA detected". On a HIP build those messages are
    misleading, so they say AMD / CUDA-HIP instead. UI text only; no behaviour.

Usage:  python apply_rocm_port.py <path-to-t8-kit> [<path-to-t8-kit> ...]
Re-running is safe: each patch reports SKIP when already applied.
"""
import sys
from pathlib import Path

# --------------------------------------------------------------------- patch 1
CG_OLD = (
    '        if attention_backend == "auto":\n'
    '            attention_backend = "flash" if flash else "cudnn" if fused and torch.backends.cudnn.is_available() else "sdpa"\n'
)
CG_NEW = (
    '        if attention_backend == "auto":\n'
    '            # ROCm/HIP: torch.cuda is the HIP alias, so device.type == "cuda" and\n'
    '            # the ATen schema advertises seqused_k, but ROCm\'s mha_varlen_fwd\n'
    '            # rejects it ("[ROCm] mha_varlen_fwd: seqused_k must be nullopt").\n'
    '            # The schema-only probe above would therefore select "flash" and\n'
    '            # crash on the first decode step. seqused_k=None is not a fix\n'
    '            # either: varlen FA would attend over unused future cache slots.\n'
    '            # Masked SDPA is numerically exact here (0.0 max abs diff vs eager\n'
    '            # inside a captured graph on gfx1201) and captures cleanly, so\n'
    '            # force it on HIP.\n'
    '            if getattr(torch.version, "hip", None) is not None:\n'
    '                attention_backend = "sdpa"\n'
    '            else:\n'
    '                attention_backend = "flash" if flash else "cudnn" if fused and torch.backends.cudnn.is_available() else "sdpa"\n'
)

# --------------------------------------------------------------------- patch 2
CW_OLD = (
    "def create_pipe(root: Path, request: dict):\n"
    "    from yue2 import YuE2Pipeline\n"
    "\n"
    "    paths = model_paths(root)\n"
    '    backend = request.get("backend", "torch-eager")\n'
    '    budget = float(request.get("memory_budget_gib", 23.5))\n'
    "    return YuE2Pipeline.from_pretrained(\n"
    '        str(paths["model"]), vae=str(paths["vae"]), device="cuda",\n'
    "        memory_budget_gib=budget, backend=backend, quantization=\"none\",\n"
    '        offload_ar=bool(request.get("offload_ar", True)), local_files_only=True,\n'
    '        nar_attention=request.get("nar_attention", "sdpa"),\n'
    '        nar_query_chunk_size=int(request.get("nar_query_chunk_size", 256)),\n'
    '        verify_hashes=bool(request.get("verify_hashes", False)), progress=False,\n'
    "    )\n"
)
CW_NEW = (
    "def vae_core_frames_for(request: dict) -> int:\n"
    '    """ROCm/consumer-GPU fix: do not derive the VAE tile size from budget alone.\n'
    "\n"
    "    Upstream lets the pipeline pick it from memory_budget_gib (512 if <=12 GiB\n"
    "    else 1024), which assumes the 24 GiB card this project was validated on. On a\n"
    "    16 GiB Radeon the 1024-frame tiles select far slower MIOpen convolution\n"
    "    solvers: 313 s of VAE decode versus 156 s with 512-frame tiles for the\n"
    "    identical 175 s song on an RX 9070 XT (gfx1201). Default to the smaller tile\n"
    "    below 20 GiB and still honour an explicit request field.\n"
    '    """\n'
    '    explicit = request.get("vae_core_frames")\n'
    "    if explicit:\n"
    "        return int(explicit)\n"
    "    try:\n"
    "        import torch\n"
    "        total_gib = torch.cuda.get_device_properties(0).total_memory / 2 ** 30\n"
    "    except Exception:\n"
    "        return 512\n"
    "    return 1024 if total_gib >= 20 else 512\n"
    "\n"
    "\n"
    "def create_pipe(root: Path, request: dict):\n"
    "    from yue2 import YuE2Pipeline\n"
    "\n"
    "    paths = model_paths(root)\n"
    '    backend = request.get("backend", "torch-eager")\n'
    '    budget = float(request.get("memory_budget_gib", 23.5))\n'
    "    return YuE2Pipeline.from_pretrained(\n"
    '        str(paths["model"]), vae=str(paths["vae"]), device="cuda",\n'
    "        memory_budget_gib=budget, backend=backend, quantization=\"none\",\n"
    '        offload_ar=bool(request.get("offload_ar", True)), local_files_only=True,\n'
    '        nar_attention=request.get("nar_attention", "sdpa"),\n'
    '        nar_query_chunk_size=int(request.get("nar_query_chunk_size", 256)),\n'
    '        verify_hashes=bool(request.get("verify_hashes", False)), progress=False,\n'
    "        vae_core_frames=vae_core_frames_for(request),\n"
    "    )\n"
)

# --------------------------------------------------------------------- patch 3
# descript-audiotools (pulled in by demucs -> dac) evaluates `dist.ReduceOp`
# inside a class body at import time. torch >= 2.9 makes torch.distributed a lazy
# module that does not expose ReduceOp until a process group exists, so the import
# itself raises AttributeError and Seed-VC never starts:
#   AttributeError: module 'torch.distributed' has no attribute 'ReduceOp'
# This is a descript-audiotools vs modern-torch incompatibility, not an AMD issue --
# the same crash happens on NVIDIA with torch 2.11. t8 pinned torch 2.8 partly for
# this. Seed-VC never initialises distributed training, so a harmless sentinel in
# the voice runtime's audiotools is enough to keep the annotation importable.
AT_REL = Path("runtime/voice/Lib/site-packages/audiotools/ml/decorators.py")
AT_OLD = "import torch.distributed as dist\n"
AT_NEW = (
    "import torch.distributed as dist\n"
    "\n"
    "# ROCm-port patch: torch >= 2.9 makes torch.distributed lazy and hides\n"
    "# ReduceOp until a process group is initialised, but descript-audiotools\n"
    "# evaluates `dist.ReduceOp` in the Tracker class body at import time, which\n"
    "# crashes every Seed-VC launch with:\n"
    "#   AttributeError: module 'torch.distributed' has no attribute 'ReduceOp'\n"
    "# Seed-VC never initialises distributed training, so a minimal sentinel is\n"
    "# enough to keep the annotation importable.\n"
    "if not hasattr(dist, 'ReduceOp'):\n"
    "    class _ReduceOpStub:\n"
    "        AVG = 'avg'\n"
    "        SUM = 'sum'\n"
    "    dist.ReduceOp = _ReduceOpStub\n"
)

# --------------------------------------------------------------------- patch 4
# The CAMPPlus speaker encoder is the only model in Seed-VC's load_models() that
# stays fp32, and fp32 batch_norm on gfx1201 selects MIOpen's
# BatchNormFwdInferSpatial kernel, which HIPRTC JIT-compiles at first use. The
# TheRock wheel carries no libc++ headers for comgr, so that compile fails:
#   fatal error: 'type_traits' file not found  ->  miopenStatusUnknownError
# Measured on this box: fp32 conv1d+BN fails, fp16 succeeds -- fp16 selects a
# different MIOpen batchnorm kernel that needs no JIT. Every other model in
# load_models() already runs .half() when args.fp16, so keep CAMPPlus consistent
# instead of special-casing the precision of one submodule.
CP_REL = "vendor/seed-vc/inference.py"
CP_OLD_A = (
    "    campplus_model.eval()\n"
    "    campplus_model.to(device)\n"
)
CP_NEW_A = (
    "    campplus_model.eval()\n"
    "    campplus_model.to(device)\n"
    "    # ROCm-port patch: keep CAMPPlus in fp16 like every other model here.\n"
    "    # fp32 batch_norm on gfx1201 JIT-compiles MIOpen's spatial-batchnorm\n"
    "    # kernel via HIPRTC, and the TheRock wheel has no libc++ headers for\n"
    "    # comgr ('type_traits' file not found) -> miopenStatusUnknownError on the\n"
    "    # encoder's first conv. fp16 uses a non-JIT kernel and runs fine.\n"
    "    if fp16:\n"
    "        campplus_model.half()\n"
)
# The caller feeds the encoder fp32 fbank features; with the weights now half the
# conv would raise "Input type (torch.cuda.FloatTensor) and weight type
# (torch.cuda.HalfTensor) should be the same". Cast at the call site.
CP_OLD_B = (
    "    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)\n"
    "    style2 = campplus_model(feat2.unsqueeze(0))\n"
)
CP_NEW_B = (
    "    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)\n"
    "    # ROCm-port patch: cast the fbank features to the encoder's dtype\n"
    "    # (fp16 when args.fp16 -- see the CAMPPlus note in load_models).\n"
    "    style2 = campplus_model(feat2.unsqueeze(0).half())\n"
)
# RMVPE (the F0 extractor) is the last fp32 model in load_models(); its fp32
# batchnorm triggers the same MIOpen JIT failure. It natively supports half.
CP_OLD_C = "        f0_extractor = RMVPE(model_path, is_half=False, device=device)\n"
CP_NEW_C = (
    "        # ROCm-port patch: RMVPE runs fp32 batchnorm, which makes MIOpen\n"
    "        # JIT-compile its spatial-batchnorm kernel; the TheRock wheel carries no\n"
    "        # libc++ headers for comgr, so that compile fails with\n"
    "        # 'type_traits' file not found -> miopenStatusUnknownError. RMVPE natively\n"
    "        # supports half, and args.fp16 is the same switch every other model here\n"
    "        # uses.\n"
    "        f0_extractor = RMVPE(model_path, is_half=bool(fp16), device=device)\n"
)

# --------------------------------------------------------------------- patch 5
# Root-cause mitigation for the MIOpen JIT failures above: rather than hunting
# every fp32 kernel one by one (spatial batchnorm here, the GRU in RMVPE there),
# run the voice worker with MIOpen disabled entirely -- in ROCm PyTorch,
# torch.backends.cudnn IS the MIOpen backend, so disabling it makes conv/BN/RNN
# use PyTorch's native implementations, which need no HIPRTC compilation. The
# cost is some conv throughput inside Seed-VC/Demucs only; generation
# (core_worker) is unaffected and keeps MIOpen for its heavy GEMMs.
VW_OLD = (
    "        import torch\n"
    '        if not torch.cuda.is_available():\n'
    '            raise RuntimeError("参考音色运行时未检测到 NVIDIA CUDA")\n'
)
VW_NEW = (
    "        import torch\n"
    "        # ROCm-port patch: MIOpen JIT-compiles several fp32 kernels (spatial\n"
    "        # batchnorm, RNN) via HIPRTC at first use, and the TheRock wheel has no\n"
    "        # libc++ headers for comgr, so those compiles fail with\n"
    "        # 'type_traits' file not found -> miopenStatusUnknownError. Disable the\n"
    "        # MIOpen backend for this worker: conv/BN/RNN then use PyTorch's native\n"
    "        # implementations, which need no JIT. Generation is unaffected.\n"
    "        torch.backends.cudnn.enabled = False\n"
    '        if not torch.cuda.is_available():\n'
    '            raise RuntimeError("参考音色运行时未检测到可用的 GPU（CUDA/HIP）")\n'
    "        # ROCm-port patch: torchaudio 2.11 routes .save() through torchcodec,\n"
    "        # which needs FFmpeg shared libraries that are not shipped on Windows.\n"
    "        # Seed-VC only ever writes plain WAV files, so substitute soundfile.\n"
    "        import torchaudio\n"
    "        import soundfile as _sf\n"
    "\n"
    "        def _sf_save(filepath, src, sample_rate, *_, **__):\n"
    "            import numpy as _np\n"
    "            data = src.detach().cpu().numpy()\n"
    "            if data.ndim == 2 and data.shape[0] < data.shape[-1]:\n"
    "                data = data.T\n"
    "            _sf.write(str(filepath), _np.squeeze(data), sample_rate)\n"
    "\n"
    "        torchaudio.save = _sf_save\n"
)

# --------------------------------------------------------------------- patch 6
# The ROCm kit must never self-update. The in-app updater pulls a release archive
# from GitHub and re-extracts the whole kit, which would wipe every ROCm patch and
# replace the ROCm PyTorch runtimes with the upstream cu128 ones. t8 1.3.0 also
# ships a newer core runtime than what this port is tested against. So report the
# kit as up to date, unconditionally; users can still update by re-running the
# installer scripts after a fresh t8 release.
UP_REL = "app/yue2_app/updater.py"
UP_OLD = (
    'def public_update(manifest: dict, current_version: str) -> dict:\n'
    '    available = version_tuple(manifest["version"]) > version_tuple(current_version)\n'
)
UP_NEW = (
    'def public_update(manifest: dict, current_version: str) -> dict:\n'
    "    # ROCm-port patch: in-app self-update would replace the ROCm PyTorch\n"
    "    # runtimes and wipe every port patch (attention backend, VAE tile size,\n"
    "    # Seed-VC fixes), so this kit reports itself as up to date. To update:\n"
    "    # re-apply apply_rocm_port.py after installing the new release manually.\n"
    "    available = False\n"
)

# --------------------------------------------------------------------- patch 7
# Device wording. The UI and the worker self-checks name NVIDIA specifically; on a
# HIP build that is simply wrong and reads as "unsupported hardware". Text only --
# no behaviour change, and the checks themselves already work through the HIP alias.
IDX_REL = "app/web/index.html"
IDX_OLD = "所有推理都在你的 NVIDIA GPU 上完成。"
IDX_NEW = "所有推理都在你的 AMD GPU 上完成。"

VID_OLD = "未检测到 NVIDIA CUDA"
VID_NEW = "未检测到可用的 GPU（CUDA/HIP）"
# transcribe_worker.py uses a differently-worded prefix than core/voice workers.
VIDTR_OLD = "转谱运行时未检测到 NVIDIA CUDA"
VIDTR_NEW = "转谱运行时未检测到可用的 GPU（CUDA/HIP）"

# Each entry: (relative path, idempotence marker, [(old, new), ...]).
# Anchors are applied in order; every one must match exactly once.
PATCHES = [
    ("vendor/yue2/cuda_graph.py", "torch.version, \"hip\"",
     [(CG_OLD, CG_NEW)]),
    ("app/yue2_app/core_worker.py", "vae_core_frames_for",
     [(CW_OLD, CW_NEW)]),
    # core_worker's self-check message lives far from create_pipe(), so it gets its
    # own entry with its own marker rather than folding into the pair above.
    ("app/yue2_app/core_worker.py", VID_NEW,
     [("自检失败：" + VID_OLD, "自检失败：" + VID_NEW)]),
    ("runtime/voice/Lib/site-packages/audiotools/ml/decorators.py", "ROCm-port patch",
     [(AT_OLD, AT_NEW)]),
    (CP_REL, "is_half=bool(fp16)",
     [(CP_OLD_A, CP_NEW_A), (CP_OLD_B, CP_NEW_B), (CP_OLD_C, CP_NEW_C)]),
    ("app/yue2_app/voice_worker.py", "torchaudio.save = _sf_save",
     [(VW_OLD, VW_NEW)]),
    (UP_REL, "ROCm-port patch: in-app self-update",
     [(UP_OLD, UP_NEW)]),
    ("app/yue2_app/transcribe_worker.py", VIDTR_NEW,
     [(VIDTR_OLD, VIDTR_NEW)]),
    (IDX_REL, "AMD GPU 上完成",
     [(IDX_OLD, IDX_NEW)]),
]


def main(kits):
    if not kits:
        print(__doc__)
        return 2
    rc = 0
    for kit in kits:
        root = Path(kit).resolve()
        print("=== %s ===" % root)
        for rel, marker, pairs in PATCHES:
            path = root / rel
            if not path.is_file():
                if rel.startswith("runtime/"):
                    # Runtime payloads are created by setup_rocm_runtime.ps1. Applying
                    # the port before installing a role is a normal order of events,
                    # so this is a hint, not a failure -- just re-run after install.
                    print("  SKIP  %-34s (runtime not installed yet)" % rel)
                else:
                    print("  MISS  %-34s (not found in this checkout)" % rel)
                    rc = 1
                continue
            # Read/write with newline="" so the file's existing line endings survive
            # untouched. Text-mode round-tripping would rewrite every LF as CRLF on
            # Windows and turn a 2-line fix into a whole-file diff.
            with open(path, "r", encoding="utf-8", newline="") as handle:
                text = handle.read()
            if marker in text:
                print("  SKIP  %-34s (already patched)" % rel)
                continue
            ok = True
            for old, new in pairs:
                count = text.count(old)
                if count != 1:
                    print("  FAIL  %-34s (anchor matched %d times, expected 1)"
                          % (rel, count))
                    ok = False
                    rc = 1
                    break
                text = text.replace(old, new)
            if ok:
                with open(path, "w", encoding="utf-8", newline="") as handle:
                    handle.write(text)
                print("  PATCH %-34s applied (%d change%s)"
                      % (rel, len(pairs), "s" if len(pairs) > 1 else ""))
        print()
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
