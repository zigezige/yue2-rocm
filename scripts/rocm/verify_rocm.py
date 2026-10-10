"""Verify the embedded Windows runtime is an AMD ROCm PyTorch install."""
from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit a machine-readable report")
    args = parser.parse_args()
    try:
        import torch
    except Exception as exc:
        print(f"PyTorch import failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    report = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()) if torch.cuda.is_available() else False,
        "devices": [],
    }
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            report["devices"].append({
                "index": index,
                "name": props.name,
                "memory_gib": round(props.total_memory / 2**30, 2),
                "capability": list(torch.cuda.get_device_capability(index)),
            })

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))

    failures = []
    if report["hip"] is None:
        failures.append("torch.version.hip is None: a ROCm wheel is not installed")
    if report["cuda"] is not None:
        failures.append("torch.version.cuda is set: a CUDA wheel was installed")
    if not report["cuda_available"]:
        failures.append("torch.cuda.is_available() is false: no usable AMD GPU is visible")
    if not report["bf16_supported"]:
        failures.append("the selected GPU does not report BF16 support")
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1
    print("OK: ROCm runtime and BF16 device are ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
