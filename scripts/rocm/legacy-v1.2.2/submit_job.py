"""Submit a generation job to the running t8 service and follow it to completion.

This exercises the part of the port that has NOT been proven yet:
    service -> workflow/queue -> core_worker (runtime/core/python.exe)
            -> vendored yue2 (vendor/yue2) -> ROCm
We have already proven the yue2 pipeline itself on this GPU and that the service
starts; this closes the loop through t8's own job machinery.

Uses t8's own client.py so the contract (version check, endpoint, payload shape)
is the project's, not a hand-rolled guess.
"""
import json
import os
import sys
import time
from pathlib import Path

KIT = r"D:\YuE2\T8"
os.environ["YUE2_HOME"] = KIT
os.environ["YUE2_SERVICE"] = "http://127.0.0.1:8189"
sys.path.insert(0, KIT)

import client  # noqa: E402

REQUEST = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"D:\DSHWEB\_yue_probe\zh_song.json")
song = json.loads(REQUEST.read_text(encoding="utf-8"))

payload = {
    "style": song["style"],
    "lyrics": song["lyrics"],
    "cot": song.get("cot", "full"),
    "seed": int(song.get("seed", 831001)),
    "candidates": 1,
    # Leave backend / offload_ar / nar_attention at t8's own defaults: it already
    # ships torch-eager + offload_ar=True, which is what this 16 GB card wants.
}
print("request file :", REQUEST)
print("style        :", payload["style"][:80])
print("lyrics chars :", len(payload["lyrics"]))
print("seed         :", payload["seed"])
print(flush=True)

health = client.ensure_service()
print("service      :", health["version"], "root", health["root"])
caps = health["ready"]["capabilities"]
print("capabilities :", {k: caps[k] for k in ("generation", "transcription",
                                              "score_renderer", "voice_conversion")})
print(flush=True)

job = client.submit("generate", payload)
job_id = job["id"]
print("submitted job:", job_id, "status", job.get("status"))
print(flush=True)

start = time.time()
last_line = None
deadline = start + 60 * 60
while time.time() < deadline:
    try:
        status = client.request("/api/jobs/%s" % job_id, timeout=30)
    except Exception as exc:
        print("%6.1fs  poll error: %s" % (time.time() - start, type(exc).__name__), flush=True)
        time.sleep(5)
        continue
    state = status.get("status")
    line = "status=%s stage=%s tokens=%s progress=%s seed=%s candidate=%s" % (
        state, status.get("stage"), status.get("tokens"), status.get("progress"),
        status.get("seed"), status.get("candidate"))
    if line != last_line:
        print("%6.1fs  %s" % (time.time() - start, line), flush=True)
        last_line = line
    if state == "complete":
        print(flush=True)
        print("=== COMPLETE in %.1f s ===" % (time.time() - start))
        print(json.dumps(status, indent=2, ensure_ascii=True)[:4000])
        break
    if state in ("failed", "cancelled"):
        print(flush=True)
        print("=== %s in %.1f s ===" % (state.upper(), time.time() - start))
        print("error:", status.get("error"))
        print(json.dumps(status, indent=2, ensure_ascii=True)[:3000])
        raise SystemExit(1)
    time.sleep(4)
else:
    print("TIMED OUT after 60 min")
    raise SystemExit(1)
