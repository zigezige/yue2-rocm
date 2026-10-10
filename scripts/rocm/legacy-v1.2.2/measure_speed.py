"""Measure generation speed after Patch 2 (device-aware VAE tile size).

Background
----------
Two numbers were on record and never reconciled:

  T8, before Patch 2 : 613.7 s for 174.91866666666667 s of audio
                       (config.json showed vae_core_frames=1024, the upstream
                        default derived from memory_budget_gib=23.5)
  official CLI       : 423.4 s for 174.91866666666667 s of audio
                       (yue2_run.py, vae_core_frames=512, offload_ar=False)

Hypothesis: the 190 s gap is VAE tiling. Patch 2 makes the tile size device-aware
(512 below 20 GiB) instead of budget-derived, so the same request through T8 should
land near the official-CLI number, with the remainder explained by T8's
offload_ar=True default.

This runs the identical request (same lyrics, seed 20260917) twice through the t8
service -- once with t8's own default (offload_ar=True) and once with
offload_ar=False -- and prints stage-by-stage timings from each run's artifacts.
"""
import json
import os
import sys
import time
from pathlib import Path

KIT = Path(r"D:\YuE2\T8")
os.environ["YUE2_HOME"] = str(KIT)
os.environ["YUE2_SERVICE"] = "http://127.0.0.1:8189"
sys.path.insert(0, str(KIT))

import client  # noqa: E402

REQUEST = json.loads(Path(r"D:\DSHWEB\_yue_probe\zh_song.json").read_text(encoding="utf-8"))


def run(label, extra):
    payload = {
        "style": REQUEST["style"],
        "lyrics": REQUEST["lyrics"],
        "cot": REQUEST.get("cot", "full"),
        "seed": int(REQUEST.get("seed", 831001)),
        "candidates": 1,
    }
    payload.update(extra)
    print("=== %s ===" % label, flush=True)
    print("    request overrides: %s" % (extra or "(t8 defaults)"), flush=True)
    job = client.submit("generate", payload)
    jid = job["id"]
    start = time.time()
    last = None
    while True:
        st = client.request("/api/jobs/%s" % jid, timeout=30)
        state = st.get("status")
        line = "status=%s stage=%s tokens=%s" % (state, st.get("stage"), st.get("tokens"))
        if line != last:
            print("    %6.1fs  %s" % (time.time() - start, line), flush=True)
            last = line
        if state == "complete":
            return st, time.time() - start
        if state in ("failed", "cancelled"):
            print("    FAILED:", st.get("error"), flush=True)
            log = st.get("log")
            if log and Path(log).is_file():
                for ln in Path(log).read_text(encoding="utf-8", errors="replace").splitlines()[-15:]:
                    print("    | " + ln, flush=True)
            raise SystemExit(1)
        time.sleep(4)


results = {}
for label, extra in (("A: t8 defaults (offload_ar=True)", {}),
                     ("B: offload_ar=False", {"offload_ar": False})):
    status, wall = run(label, extra)
    song_dir = Path(status["result"]["artifact_dir"])
    cfg = json.loads((song_dir / "config.json").read_text(encoding="utf-8"))
    res = json.loads((song_dir / "result.json").read_text(encoding="utf-8"))
    timing = res["timing"]
    results[label] = {
        "wall_seconds": round(wall, 1),
        "audio_seconds": round(res["audio_seconds"], 3),
        "vae_core_frames": cfg["vae_core_frames"],
        "memory_budget_gib": cfg["memory_budget_gib"],
        "offload_ar": cfg["offload_ar"],
        "abc_seconds": round(timing.get("abc", {}).get("seconds", 0), 1),
        "semantic_seconds": round(timing.get("semantic", {}).get("seconds", 0), 1),
        "semantic_tps": round(timing.get("semantic", {}).get("output_tps", 0), 1),
        "nar_seconds": round(timing.get("nar_seconds", 0), 1),
        "vae_seconds": round(timing.get("vae_seconds", 0), 1),
        "e2e_seconds": round(timing.get("e2e_seconds", 0), 1),
        "truncated": res["truncated"],
        "job_id": status["id"],
    }
    print("    -> wall=%.1fs  vae_frames=%s  offload_ar=%s" % (
        wall, cfg["vae_core_frames"], cfg["offload_ar"]), flush=True)
    print(flush=True)

print("=== SUMMARY ===")
print(json.dumps(results, indent=2, ensure_ascii=False))
Path(r"D:\DSHWEB\_yue_probe\speed_compare.json").write_text(
    json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print("\nwritten to speed_compare.json")
