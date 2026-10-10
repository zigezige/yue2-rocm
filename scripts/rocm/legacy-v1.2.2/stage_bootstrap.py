"""Pre-stage the embeddable-Python archive and get-pip.py into the kit's downloads dir.

python.org is unreachable from this network (SSL EOF) and a bare curl against it
hangs, so fetch the two bootstrap files from a mirror here instead. setup_rocm_core.ps1
skips any download whose destination already exists, so pre-staging makes the
installer deterministic on a censored network without changing its fallback list.
"""
import os
import socket
import ssl
import sys
import urllib.request
from pathlib import Path

DOWNLOADS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"D:\YuE2\T8\downloads")
VERSION = sys.argv[2] if len(sys.argv) > 2 else "3.12.10"

socket.setdefaulttimeout(60)
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

TARGETS = [
    ("python-%s-embed-amd64.zip" % VERSION, [
        "https://mirrors.huaweicloud.com/python/%s/python-%s-embed-amd64.zip" % (VERSION, VERSION),
        "https://registry.npmmirror.com/-/binary/python/%s/python-%s-embed-amd64.zip" % (VERSION, VERSION),
        "https://mirrors.aliyun.com/python-release/windows/python-%s-embed-amd64.zip" % VERSION,
        "https://www.python.org/ftp/python/%s/python-%s-embed-amd64.zip" % (VERSION, VERSION),
    ]),
    ("get-pip.py", [
        "https://bootstrap.pypa.io/get-pip.py",
    ]),
]

DOWNLOADS.mkdir(parents=True, exist_ok=True)
failed = []

for name, urls in TARGETS:
    dest = DOWNLOADS / name
    if dest.is_file() and dest.stat().st_size > 100_000:
        print("already staged: %s (%.2f MB)" % (dest, dest.stat().st_size / 1024**2))
        continue
    got = False
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            data = urllib.request.urlopen(req, timeout=60, context=CTX).read()
            if len(data) < 100_000:
                print("  too small from %s (%d bytes)" % (url, len(data)))
                continue
            dest.write_bytes(data)
            print("staged %s from %s (%.2f MB)" % (dest, url.split("/")[2], len(data) / 1024**2))
            got = True
            break
        except Exception as exc:
            print("  fail %s: %s" % (url.split("/")[2], type(exc).__name__))
    if not got:
        failed.append(name)

print()
print("FAILED: %s" % failed if failed else "both bootstrap files staged")
raise SystemExit(1 if failed else 0)
