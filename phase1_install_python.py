"""Phase 1: create build root and install a pinned CPython 3.11.6 into it.

Runs OUTSIDE the DSH workspace (D:\\aitext_kernel_build), so it needs the
one-shot full-access escalation. No pipes are used for child output (the
sandbox forbids piped stdio for some hosts); everything is inherited or
written to files.
"""
import os
import subprocess
import sys
import time
import urllib.request

BUILD = r"D:\aitext_kernel_build"
PYDIR = os.path.join(BUILD, "py311")
EXE = os.path.join(BUILD, "python-3.11.6-amd64.exe")
URL = "https://www.python.org/ftp/python/3.11.6/python-3.11.6-amd64.exe"


def log(msg):
    print("[phase1] %s" % msg, flush=True)


def download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 20_000_000:
        log("installer already present: %d bytes" % os.path.getsize(dest))
        return
    log("downloading %s" % url)
    t0 = time.time()
    last = 0
    req = urllib.request.Request(url, headers={"User-Agent": "python-urllib/3.12"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if got - last >= (10 << 20):
                last = got
                pct = ("%.1f%%" % (100.0 * got / total)) if total else "?"
                log("  %d bytes (%s) %.1f s" % (got, pct, time.time() - t0))
    log("downloaded %d bytes in %.1f s" % (os.path.getsize(dest), time.time() - t0))


def main():
    os.makedirs(BUILD, exist_ok=True)
    download(URL, EXE)
    if os.path.exists(os.path.join(PYDIR, "python.exe")):
        log("python already installed at %s" % PYDIR)
    else:
        args = [
            EXE, "/quiet", "InstallAllUsers=0", "TargetDir=%s" % PYDIR,
            "Include_pip=1", "Include_test=0", "Include_doc=0", "Include_tcltk=1",
            "Include_tools=0", "Include_launcher=0", "PrependPath=0",
            "Shortcuts=0", "AssociateFiles=0", "CompileAll=0", "SimpleInstall=1",
        ]
        log("running installer (quiet, per-user, custom TargetDir)")
        p = subprocess.run(args)
        log("installer exit code = %s" % p.returncode)

    py = os.path.join(PYDIR, "python.exe")
    if not os.path.exists(py):
        log("FAIL: %s not found" % py)
        return 2
    for cmd in ([py, "--version"], [py, "-m", "pip", "--version"]):
        log("run: %s" % " ".join(cmd))
        p = subprocess.run(cmd)
        log("  exit = %s" % p.returncode)
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
