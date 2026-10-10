"""Fetch the reference-voice weights (Seed-VC + Demucs) from the t8 bundle repo.

Downloads only what t8's voice runtime needs:
  Seed-VC/*  DiT checkpoint, whisper-small encoder, bigvgan vocoder, rmvpe, campplus
  Demucs/*   htdemucs weights, config, json
plus the upstream VOICE_MODEL_MANIFEST.json, which is the authority the kit's own
scripts/verify_voice_models.py checks against (path + size + sha256).

Model files come over the mirror with ranged, resumable, parallel HTTP -- the hub
client cannot reach huggingface.co in bulk from this network and is incompatible
with the mirror's API (see DEPLOY_NOTES.md).
"""
import json
import os
import socket
import ssl
import sys
import threading
import time
import urllib.request
from pathlib import Path

REPO = "t8star/YuE2-Comfy"
MIRROR = "https://hf-mirror.com"
API_HOSTS = ["https://hf-mirror.com", "https://huggingface.co"]
MODELS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"D:\YuE2\models")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) t8-rocm-port"
PARTS = 6
LARGE = 40 * 1024 * 1024

socket.setdefaulttimeout(90)
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
LOCK = threading.Lock()


def log(m):
    with LOCK:
        print(m, flush=True)


def listing(host):
    req = urllib.request.Request("%s/api/models/%s?blobs=true" % (host, REPO),
                                 headers={"User-Agent": UA})
    payload = json.loads(urllib.request.urlopen(req, timeout=60, context=CTX).read())
    return {s["rfilename"]: (s.get("size") or 0) for s in payload.get("siblings", [])}


def open_range(url, start):
    headers = {"User-Agent": UA}
    if start:
        headers["Range"] = "bytes=%d-" % start
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90, context=CTX)


def fetch(url, dest, lo, hi, label, stall_limit=8):
    want = hi - lo + 1
    stalls = 0
    while stalls < stall_limit:
        have = os.path.getsize(dest) if os.path.exists(dest) else 0
        if have >= want:
            return True
        try:
            resp = open_range(url, lo + have)
            if have and resp.status != 206:
                resp.close()
                os.remove(dest)
                continue
            with open(dest, "ab") as handle:
                got = have
                while got < want:
                    block = resp.read(min(1 << 21, want - got))
                    if not block:
                        break
                    handle.write(block)
                    got += len(block)
            resp.close()
            if os.path.getsize(dest) >= want:
                return True
            stalls += 1
        except Exception:
            stalls += 1
            time.sleep(2)
    return os.path.exists(dest) and os.path.getsize(dest) >= want


def grab_large(url, dest, size, label):
    part = size // PARTS
    bounds = [(i * part, size - 1 if i == PARTS - 1 else (i + 1) * part - 1) for i in range(PARTS)]
    ok = [False] * PARTS
    start = time.time()

    def run(i, lo, hi):
        ok[i] = fetch(url, "%s.p%02d" % (dest, i), lo, hi, label)

    threads = [threading.Thread(target=run, args=(i, lo, hi), daemon=True)
               for i, (lo, hi) in enumerate(bounds)]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        time.sleep(12)
        got = sum(os.path.getsize("%s.p%02d" % (dest, i)) for i in range(PARTS)
                  if os.path.exists("%s.p%02d" % (dest, i)))
        el = time.time() - start
        log("   %s %6.1f/%6.1f MiB  %5.2f MiB/s" % (label, got / 2**20, size / 2**20,
                                                    got / 2**20 / el if el else 0))
    for t in threads:
        t.join()
    if not all(ok):
        log("   %s MISSING PARTS %s" % (label, [i for i, v in enumerate(ok) if not v]))
        return False
    with open(dest, "wb") as out:
        for i in range(PARTS):
            piece = "%s.p%02d" % (dest, i)
            with open(piece, "rb") as src:
                while True:
                    block = src.read(1 << 22)
                    if not block:
                        break
                    out.write(block)
            os.remove(piece)
    return os.path.getsize(dest) == size


def grab_small(url, dest, size):
    for attempt in range(6):
        try:
            data = open_range(url, 0).read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            if size and len(data) != size:
                time.sleep(2)
                continue
            return True
        except Exception:
            time.sleep(3)
    return False


def main():
    table = None
    for host in API_HOSTS:
        try:
            table = listing(host)
            log("file table from %s: %d files" % (host.split("//")[1], len(table)))
            break
        except Exception as exc:
            log("  %s failed: %s" % (host, type(exc).__name__))
    if table is None:
        raise SystemExit("cannot list %s" % REPO)

    wanted = sorted(n for n in table
                    if n.startswith("Seed-VC/") or n.startswith("Demucs/")
                    or n == "VOICE_MODEL_MANIFEST.json")
    total = sum(table[n] for n in wanted)
    log("voice payload: %d files, %.2f GB" % (len(wanted), total / 1024**3))

    failed = []
    for name in wanted:
        size = table[name]
        dest = MODELS_DIR / name.replace("/", os.sep)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_file() and size and dest.stat().st_size == size:
            log("  skip %-52s (present)" % name)
            continue
        url = "%s/%s/resolve/main/%s" % (MIRROR, REPO, name)
        if size >= LARGE:
            ok = grab_large(url, str(dest), size, name)
        else:
            ok = grab_small(url, dest, size)
        if not ok:
            log("  FAILED %s" % name)
            failed.append(name)

    if failed:
        log("FAILED FILES: %s" % failed)
        raise SystemExit(1)

    got = sum(os.path.getsize(os.path.join(r, f)) for r, _d, fs in os.walk(MODELS_DIR) for f in fs)
    log("")
    log("models dir now: %.2f GB" % (got / 1024**3))

    manifest = MODELS_DIR / "VOICE_MODEL_MANIFEST.json"
    if manifest.is_file():
        log("verifying with the kit's own scripts/verify_voice_models.py ...")
        log("  (run it separately: python scripts/verify_voice_models.py --root <kit>)")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
