"""Complete the SheetSage2 and MERT-v2-FullSong model directories from the bundle.

SheetSage2 loads through trust_remote_code, so it needs its modelling .py files as
well as the weights; my earlier fetch only pulled the three files t8's
REQUIRED_FILES check enforces, which is enough to pass verification but not enough
for AutoModel.from_pretrained to actually build the model.

This tops up every file under SheetSage2/ and MERT-v2-FullSong/ that is missing
(small code/config assets only -- the weights are already on disk and skipped).
"""
import json
import os
import socket
import ssl
import sys
import urllib.request
from pathlib import Path

REPO = "t8star/YuE2-Comfy"
MIRROR = "https://hf-mirror.com"
MODELS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"D:\YuE2\models")
PREFIXES = ("SheetSage2/", "MERT-v2-FullSong/")

socket.setdefaulttimeout(90)
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=90, context=CTX).read()


table = None
for host in (MIRROR, "https://huggingface.co"):
    try:
        table = {s["rfilename"]: (s.get("size") or 0)
                 for s in json.loads(get("%s/api/models/%s?blobs=true" % (host, REPO))
                                      ).get("siblings", [])}
        print("file table from", host.split("//")[1])
        break
    except Exception as exc:
        print("  %s failed: %s" % (host.split("//")[1], type(exc).__name__))
if table is None:
    raise SystemExit("cannot list the bundle")

wanted = sorted(n for n in table if n.startswith(PREFIXES))
print("in scope: %d files, %.2f GB" % (len(wanted), sum(table[n] for n in wanted) / 1024**3))

missing = []
for name in wanted:
    dest = MODELS_DIR / name.replace("/", os.sep)
    size = table[name]
    if dest.is_file() and size and dest.stat().st_size == size:
        continue
    missing.append((name, size, dest))

print("to download: %d files" % len(missing))
failed = []
for name, size, dest in missing:
    url = "%s/%s/resolve/main/%s" % (MIRROR, REPO, name)
    for attempt in range(5):
        try:
            data = get(url)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            if size and len(data) != size:
                continue
            print("  ok  %-58s %8.1f KB" % (name, len(data) / 1024))
            break
        except Exception as exc:
            if attempt == 4:
                print("  FAIL %s: %s" % (name, type(exc).__name__))
                failed.append(name)
            import time
            time.sleep(2)

if failed:
    print("FAILED:", failed)
    raise SystemExit(1)
print("SheetSage2 / MERT directories complete")
