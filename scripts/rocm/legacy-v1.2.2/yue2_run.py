"""Convenience runner for YuE2 on this RX 9070 XT / Windows ROCm box.

Defaults chosen for a 16 GB card:
  memory_budget_gib=16   the pipeline then clamps the real budget to
                         min(16-2, 15.92-2) = 13.92 GiB automatically
  vae_core_frames=512    smaller VAE tiles; the 1024 default is sized for 24 GB
  backend='torch-eager'  measured faster than the CUDA-graph path on this card
                         AND exits with code 0. The graph path builds capacity =
                         prefix + max_tokens and, because ROCm rejects the
                         seqused_k argument that upstream FlashAttention uses to
                         mask unused slots, its masked SDPA attends over the
                         whole capacity every step: 24.7 vs 14.3 tok/s at
                         max_tokens=9000. It also crashes during teardown
                         (exit 1) after artifacts are safely written.
  ode_steps=32           production default
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "YuE2-3B"
VAE = ROOT / "models" / "YuE2-Vae"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--request", default=str(ROOT / "YuE" / "examples" / "song.json"),
                        help="request JSON with style/lyrics/cot/seed")
    parser.add_argument("--output", required=True, help="fresh output directory")
    parser.add_argument("--style", help="override the request's style prompt")
    parser.add_argument("--lyrics-file", help="override the request's lyrics from a .txt file")
    parser.add_argument("--cot", choices=("full", "melody", "off"),
                        help="full = melody+chords, melody = melody only, off = no symbolic plan")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ode-steps", type=int, default=32)
    parser.add_argument("--vae-core-frames", type=int, default=512)
    parser.add_argument("--memory-budget-gib", type=float, default=16)
    parser.add_argument("--backend", choices=("torch", "torch-eager"), default="torch-eager",
                        help="torch-eager (default here) = faster for long songs AND exits cleanly; "
                             "torch = CUDA-graph decode, only wins when prefix+max_tokens is small")
    parser.add_argument("--offload-ar", action="store_true",
                        help="offload AR weights to CPU during NAR if VRAM is tight")
    args = parser.parse_args()

    import torch
    if not torch.cuda.is_available():
        sys.exit("No ROCm/HIP device visible to PyTorch.")
    print("device     :", torch.cuda.get_device_name(0))
    print("total VRAM : %.2f GiB" % (torch.cuda.get_device_properties(0).total_memory / 2**30))
    print("bf16       :", torch.cuda.is_bf16_supported())

    from yue2 import YuE2Pipeline
    from yue2.protocol import GenerationConfig

    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    if args.style:
        request["style"] = args.style
    if args.lyrics_file:
        request["lyrics"] = Path(args.lyrics_file).read_text(encoding="utf-8")
    if args.cot:
        request["cot"] = args.cot
    if args.seed is not None:
        request["seed"] = args.seed

    print("style      :", request["style"][:90])
    print("cot/seed   :", request.get("cot", "full"), request.get("seed", 831001))
    print("backend    :", args.backend, "| ode_steps:", args.ode_steps,
          "| vae_core_frames:", args.vae_core_frames)

    start = time.time()
    with YuE2Pipeline.from_pretrained(
            str(MODEL), vae=str(VAE), device="cuda", backend=args.backend,
            memory_budget_gib=args.memory_budget_gib,
            vae_core_frames=args.vae_core_frames, offload_ar=args.offload_ar,
            generation_config=GenerationConfig(ode_steps=args.ode_steps)) as pipe:
        song = pipe(**request)
        output = Path(args.output)
        result = song.save_artifacts(output)
        print(json.dumps({k: result[k] for k in
                          ("status", "truncated", "sample_rate", "audio_seconds", "timing")},
                         indent=2, ensure_ascii=False))
    print("audio      :", output / "audio.flac")
    print("total      : %.1f s" % (time.time() - start))
    return 0


if __name__ == "__main__":
    sys.exit(main())
