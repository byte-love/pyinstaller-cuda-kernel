"""Phase 2: install the exact kernel dependency set into py311.

torch/torchvision/torchaudio come from the CUDA 12.8 index (sm_120 / RTX 50
series); everything else is pinned to the versions recorded from the shipped
cu118 bundle so the rebuilt bundle is byte-for-byte equivalent apart from the
CUDA runtime.

Child processes inherit stdio (no pipes) because the sandbox forbids piped
stdio for some hosts. TEMP/TMP are redirected into the build root.
"""
import os
import subprocess
import sys
import time

BUILD = r"D:\aitext_kernel_build"
PY = os.path.join(BUILD, "py311", "python.exe")
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
PYPI = "https://pypi.tuna.tsinghua.edu.cn/simple"
PYPI_EXTRA = "https://pypi.org/simple"

# torch trio first (they pull numpy/filelock/networkx/sympy/jinja2/typing-extensions)
TORCH_PKGS = ["torch==2.7.1+cu128", "torchvision==0.22.1+cu128", "torchaudio==2.7.1+cu128"]

# the other 58 packages exactly as recorded from the shipped bundle
OTHER_PKGS = [
    "MarkupSafe==2.1.5", "aiohappyeyeballs==2.6.1", "aiohttp==3.13.2", "aiosignal==1.4.0",
    "altgraph==0.17.5", "anyio==4.12.0", "attrs==25.4.0", "certifi==2025.11.12",
    "charset_normalizer==3.4.4", "colorama==0.4.6", "datasets==4.4.2", "dill==0.4.0",
    "filelock==3.20.1", "frozenlist==1.8.0", "fsspec==2025.10.0", "h11==0.16.0",
    "httpcore==1.0.9", "httpx==0.28.1", "huggingface_hub==0.36.0", "idna==3.11",
    "jinja2==3.1.6", "joblib==1.5.3", "mpmath==1.3.0", "multidict==6.7.0",
    "multiprocess==0.70.18", "networkx==3.6.1", "numpy==2.3.5", "packaging==25.0",
    "pandas==2.3.3", "pefile==2024.8.26", "pillow==12.0.0", "propcache==0.4.1",
    "psutil==7.2.0", "pyarrow==22.0.0", "pyinstaller==6.17.0",
    "pyinstaller_hooks_contrib==2025.10", "python_dateutil==2.9.0.post0", "pytz==2025.2",
    "pywin32_ctypes==0.2.3", "pyyaml==6.0.3", "regex==2025.11.3", "requests==2.32.5",
    "safetensors==0.7.0", "scikit_learn==1.8.0", "scipy==1.16.3", "setuptools==65.5.0",
    "six==1.17.0", "sympy==1.14.0", "threadpoolctl==3.6.0", "tokenizers==0.22.1",
    "tqdm==4.67.1", "transformers==4.57.3", "typing_extensions==4.15.0", "tzdata==2025.3",
    "urllib3==2.6.2", "xxhash==3.6.0", "yarl==1.22.0",
]


def log(msg):
    print("[phase2] %s" % msg, flush=True)


def run(cmd, timeout=None):
    log("run: %s" % " ".join(cmd))
    t0 = time.time()
    p = subprocess.run(cmd, timeout=timeout)
    log("  exit=%s  (%.1f s)" % (p.returncode, time.time() - t0))
    return p.returncode


def main():
    os.makedirs(os.path.join(BUILD, "tmp"), exist_ok=True)
    os.makedirs(os.path.join(BUILD, "pipcache"), exist_ok=True)
    os.environ["TEMP"] = os.path.join(BUILD, "tmp")
    os.environ["TMP"] = os.path.join(BUILD, "tmp")
    os.environ["PIP_CACHE_DIR"] = os.path.join(BUILD, "pipcache")
    os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(k, None)

    if not os.path.exists(PY):
        log("FAIL: %s missing" % PY)
        return 2

    rc = run([PY, "-m", "pip", "install", "--upgrade",
              "pip==23.2.1", "setuptools==65.5.0", "wheel", "-i", PYPI])
    if rc != 0:
        log("WARN: bootstrap upgrade returned %s (continuing)" % rc)

    log("--- CUDA 12.8 torch trio (~4 GB download) ---")
    rc = run([PY, "-m", "pip", "install", "--no-cache-dir", *TORCH_PKGS, "-i", TORCH_INDEX])
    if rc != 0:
        log("FAIL: torch cu128 install returned %s" % rc)
        return 3

    log("--- remaining %d packages ---" % len(OTHER_PKGS))
    rc = run([PY, "-m", "pip", "install", *OTHER_PKGS,
              "-i", PYPI, "--extra-index-url", PYPI_EXTRA])
    if rc != 0:
        log("bulk install failed (%s) -> retrying one by one to find the bad pin" % rc)
        bad = []
        for pkg in OTHER_PKGS:
            if run([PY, "-m", "pip", "install", pkg, "-i", PYPI,
                    "--extra-index-url", PYPI_EXTRA]) != 0:
                bad.append(pkg)
        log("packages that failed: %s" % (bad or "none"))

    log("--- versions actually installed ---")
    run([PY, "-m", "pip", "list"])
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
