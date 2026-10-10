"""Apply the ROCm CUDA-graph fix to any YuE2 cuda_graph.py (upstream or vendored).

The bug
-------
GraphAR picks its decode-time attention backend with a *schema-only* probe:

    fused = self.device.type == "cuda" and self.dtype in {bf16, fp16} and head_dim % 8 == 0
    flash = fused and head_dim <= 256 and hasattr(torch.ops.aten, "_flash_attention_forward") \\
            and "seqused_k" in str(torch.ops.aten._flash_attention_forward.default._schema)
    if attention_backend == "auto":
        attention_backend = "flash" if flash else "cudnn" if ... else "sdpa"

On ROCm/HIP, ``device.type`` is still the string "cuda" (torch.cuda is the HIP
alias) and the ATen schema is shared across backends, so ``flash`` comes out True -
but ROCm's implementation rejects the argument:

    RuntimeError: [ROCm] mha_varlen_fwd: seqused_k must be nullopt

Passing seqused_k=None is NOT a valid workaround: variable-length FlashAttention
would then attend over unused future cache slots and return wrong logits. The
masked-SDPA branch is the numerically exact path here (measured 0.0 max abs diff
vs eager inside a captured graph on gfx1201), so force it when torch.version.hip
is set.

Usage:  python patch_cuda_graph_rocm.py <path-to-cuda_graph.py> [...more paths]
Idempotent: a file already carrying the guard is reported and skipped.
"""
import sys
from pathlib import Path

OLD = (
    '        if attention_backend == "auto":\n'
    '            attention_backend = "flash" if flash else "cudnn" if fused and torch.backends.cudnn.is_available() else "sdpa"\n'
)

NEW = (
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


def main(paths):
    if not paths:
        print(__doc__)
        return 2
    rc = 0
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            print("SKIP  %s (not a file)" % path)
            rc = 1
            continue
        text = path.read_text(encoding="utf-8")
        if "torch.version, \"hip\"" in text or "torch.version.hip" in text:
            print("SKIP  %s (already patched)" % path)
            continue
        if text.count(OLD) != 1:
            print("FAIL  %s (anchor matched %d times, expected 1)" % (path, text.count(OLD)))
            rc = 1
            continue
        path.write_text(text.replace(OLD, NEW), encoding="utf-8")
        print("PATCH %s" % path)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
