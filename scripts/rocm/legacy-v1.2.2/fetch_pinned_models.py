"""Fetch the two remaining T8-pinned models and emit a byte-exact MODEL_MANIFEST.json.

Why the two extra models: t8's core_worker calls verify_bundle(), which insists the
manifest declare EXACTLY the four REQUIRED_FILES entries and then SHA-verifies each
one. We could patch that check away, but a portable ROCm port of t8 should keep
t8's own integrity gate intact - and SheetSage2 + MERT-v2-FullSong also unlock the
audio->score transcription feature. They cost ~2.58 GB.

The manifest is built by AST-parsing t8's own PINNED_MODELS, so the source /
revision / file / size / sha256 fields are guaranteed identical to what t8 expects.

Model downloads go through hf-mirror.com over plain ranged HTTP, because
huggingface.co resets bulk transfers here and the hub client is incompatible with
the mirror (see DEPLOY_NOTES.md).
"""
import ast
import hashlib
import json
import os
import socket
import ssl
import sys
import threading
import time
import urllib.request
from pathlib import Path

# ---- parse t8's own pins ------------------------------------------------
VERIFY_SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"D:\DSHWEB\_yue_probe\t8\app\yue2_app\model_verify.py")
MODELS_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(r"D:\YuE2\models")

tree = ast.parse(VERIFY_SRC.read_text(encoding="utf-8"))
PINNED = None
REQUIRED = None
for node in tree.body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
        if node.targets[0].id == "PINNED_MODELS":
            PINNED = ast.literal_eval(node.value)
        elif node.targets[0].id == "REQUIRED_FILES":
            REQUIRED = ast.literal_eval(node.value)
if PINNED is None or REQUIRED is None:
    raise SystemExit("could not parse PINNED_MODELS / REQUIRED_FILES from %s" % VERIFY_SRC)

print("pinned models :", ", ".join(PINNED))
print("required files:", {k: len(v) for k, v in REQUIRED.items()})
print("models dir    :", MODELS_DIR)
print(flush=True)

MIRROR = "https://hf-mirror.com"
API_HOSTS = ["https://huggingface.co", "https://hf-mirror.com"]
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) t8-rocm-port"
PARTS = 6
LARGE = 60 * 1024 * 1024

socket.setdefaulttimeout(90)
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
LOCK = threading.Lock()


def log(msg):
    with LOCK:
        print(msg, flush=True)


def listing(repo):
    for host in API_HOSTS:
        try:
            req = urllib.request.Request("%s/api/models/%s?blobs=true" % (host, repo),
                                         headers={"User-Agent": UA})
            payload = json.loads(urllib.request.urlopen(req, timeout=60, context=CTX).read())
            return {s["rfilename"]: (s.get("size") or 0) for s in payload.get("siblings", [])}
        except Exception as exc:
            log("  api %s: %s" % (host, type(exc).__name__))
    return {}


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
    for _ in range(8):
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


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ---- download only the two missing models -------------------------------
WANT = ["SheetSage2", "MERT-v2-FullSong"]
for name in WANT:
    pin = PINNED[name]
    repo = pin["source"]
    dest_dir = MODELS_DIR / name
    dest_dir.mkdir(parents=True, exist_ok=True)
    weight = MODELS_DIR / pin["file"]
    if weight.is_file() and weight.stat().st_size == pin["size"]:
        log("%s already present (%d bytes), skipping" % (name, pin["size"]))
        continue
    log("=== %s from %s ===" % (name, repo))
    table = listing(repo)
    if not table:
        log("SKIP %s (no file table)" % name)
        continue
    needed = set(REQUIRED.get(name, ())) | {Path(pin["file"]).name}
    for rel in sorted(table, key=lambda k: -table[k]):
        base = Path(rel).name
        if not (rel in needed or base in needed or rel.startswith("render_assets/")):
            continue
        size = table[rel]
        target = MODELS_DIR / name / rel
        if target.is_file() and size and target.stat().st_size == size:
            continue
        url = "%s/%s/resolve/main/%s" % (MIRROR, repo, rel)
        if size >= LARGE:
            grab_large(url, str(target), size, rel)
        else:
            grab_small(url, target, size)
    log("%s done" % name)

# ---- emit the manifest t8 expects --------------------------------------
manifest = {"bundle": "t8star/YuE2-Comfy", "models": PINNED}
manifest_path = MODELS_DIR / "MODEL_MANIFEST.json"
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
log("wrote %s" % manifest_path)

# ---- verify using t8's own rules ---------------------------------------
log("")
log("=== verifying against t8's pinned size + sha256 ===")
failures = []
for name, pin in PINNED.items():
    weight = MODELS_DIR / pin["file"]
    if not weight.is_file():
        log("  %-18s MISSING" % name)
        failures.append(name)
        continue
    size_ok = weight.stat().st_size == pin["size"]
    digest = sha256(weight) if size_ok else ""
    sha_ok = digest.lower() == pin["sha256"].lower()
    missing = [f for f in REQUIRED.get(name, ()) if not (weight.parent / f).is_file()]
    status = "OK" if (size_ok and sha_ok and not missing) else "FAIL"
    log("  %-18s %s  size=%s sha=%s missing=%s" % (name, status, size_ok, sha_ok, missing or "-"))
    if status == "FAIL":
        failures.append(name)

log("")
log("ALL VERIFIED" if not failures else "FAILURES: %s" % failures)
raise SystemExit(1 if failures else 0)
