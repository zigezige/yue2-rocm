"""End-to-end functional verification of the t8 port's optional capabilities.

Three jobs, all submitted through the service API so they run exactly as the WebUI
would run them:

  1. transcribe      -> SheetSage2 + MERT on the transcribe runtime, with
                        render_score=True which additionally exercises the
                        playwright/abcjs score renderer
  2. voice_convert   -> Seed-VC (voice cloning) + Demucs (vocal separation) on the
                        voice runtime
  3. doctor          -> the kit's own GPU/runtime/model self-check

The same 24 s generated song doubles as transcription source, cover source and
voice reference, which keeps each run fast while still exercising every stage.
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

KIT = Path(r"D:\YuE2\T8")
os.environ["YUE2_HOME"] = str(KIT)
os.environ["YUE2_SERVICE"] = "http://127.0.0.1:8189"
sys.path.insert(0, str(KIT))

import client  # noqa: E402

SOURCE_AUDIO = Path(r"D:\YuE2\outputs\smoke01\audio.flac")
UPLOAD_NAME = "verify_vocal.flac"

upload = KIT / "uploads" / UPLOAD_NAME
if not upload.is_file():
    upload.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE_AUDIO, upload)
print("upload       :", upload, "(%d KB)" % (upload.stat().st_size // 1024))

health = client.ensure_service()
caps = health["ready"]["capabilities"]
print("capabilities :", {k: caps[k] for k in
                         ("generation", "transcription", "score_renderer", "voice_conversion")})
print(flush=True)


def follow(kind, job_id, timeout):
    start = time.time()
    last = None
    while time.time() - start < timeout:
        try:
            status = client.request("/api/jobs/%s" % job_id, timeout=30)
        except Exception as exc:
            print("    poll error: %s" % type(exc).__name__, flush=True)
            time.sleep(5)
            continue
        state = status.get("status")
        line = "status=%s stage=%s tokens=%s progress=%s" % (
            state, status.get("stage"), status.get("tokens"), status.get("progress"))
        if line != last:
            print("    %6.1fs  %s" % (time.time() - start, line), flush=True)
            last = line
        if state == "complete":
            return status
        if state in ("failed", "cancelled"):
            print("    %s: %s" % (state, status.get("error")), flush=True)
            log = status.get("log")
            if log and Path(log).is_file():
                tail = Path(log).read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
                print("    --- log tail ---", flush=True)
                for line_text in tail:
                    print("    " + line_text, flush=True)
            raise SystemExit(1)
        time.sleep(3)
    raise SystemExit("%s timed out after %ds" % (kind, timeout))


# ------------------------------------------------------------------ 1. doctor
print("=== [1/3] doctor ===", flush=True)
job = client.submit("doctor", {})
jid = job["id"]
print("  job", jid, flush=True)
doctor = follow("doctor", jid, 600)
result = doctor.get("result", {})
print("  torch/version:", result.get("versions", {}).get("torch"))
print("  gpu          :", result.get("gpu"), "| cuda_available:", result.get("cuda_available"))
print("  model sha    :", result.get("model", {}).get("sha256"))
print(flush=True)

# --------------------------------------------------------------- 2. transcribe
print("=== [2/3] transcribe + score render ===", flush=True)
# The worker resolves source_path relative to the KIT ROOT, not the uploads dir,
# so the uploads/ prefix is required for the sandbox check to pass.
job = client.submit("transcribe", {
    "source_path": "uploads/" + UPLOAD_NAME,
    "melody_only": True,
    "dtype": "bf16",
    "preset": "default",
    "render_score": True,
    "render_audio": True,
})
jid = job["id"]
print("  job", jid, flush=True)
transcribe = follow("transcribe", jid, 1800)
result = transcribe.get("result", {})
print("  keys:", sorted(result), flush=True)
for key in ("abc", "abc_error", "rendered", "render_error", "duration_seconds"):
    if key in result:
        value = result[key]
        shown = str(value)[:220] if key != "abc" else "(%d chars)" % len(str(value or ""))
        print("  %-16s %s" % (key, shown), flush=True)
abc_text = result.get("abc") or ""
if abc_text:
    print("  --- ABC head ---", flush=True)
    for line_text in abc_text.splitlines()[:10]:
        print("    " + line_text, flush=True)
print(flush=True)

# -------------------------------------------------------------- 3. voice cover
print("=== [3/3] voice_convert (Seed-VC + Demucs) ===", flush=True)
job = client.submit("voice_convert", {
    "source_path": "uploads/" + UPLOAD_NAME,
    "reference_path": "uploads/" + UPLOAD_NAME,
    "diffusion_steps": 8,
    "cfg_rate": 0.7,
    "semi_tone_shift": 0,
})
jid = job["id"]
print("  job", jid, flush=True)
voice = follow("voice_convert", jid, 2400)
result = voice.get("result", {})
print("  keys:", sorted(result), flush=True)
for key in ("audio", "vocal", "accompaniment", "artifact_dir", "settings"):
    if key in result:
        print("  %-16s %s" % (key, str(result[key])[:200]), flush=True)
print(flush=True)

print("=== ALL THREE CAPABILITIES EXERCISED ===")
