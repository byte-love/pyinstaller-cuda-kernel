"""Phase 6: build the kernel with PyInstaller (onedir + _internal) and then run
the produced exe exactly the way the client does, on the same smoke SQLite split.

Usage:
    phase6_build_kernel.py            # build, then test
    phase6_build_kernel.py --no-build # only test the existing dist

Everything lives under D:\\aitext_kernel_build so the SVN working copy stays clean.
"""
import os
import shutil
import subprocess
import sys
import time

BUILD = r"D:\aitext_kernel_build"
PY = os.path.join(BUILD, "py311", "python.exe")
SRC = r".\DemoText客户端svn版本\大模型\LargeModelDemoText.py"
WORK = os.path.join(BUILD, "build")
DIST = os.path.join(BUILD, "dist")
PRETRAIN = r"C:\DemoText客户端\Resources\BigModel\PreTrainingModel\FinBert-base-zh"

COLLECT_ALL = [
    "torch", "torchvision", "torchaudio", "transformers", "tokenizers",
    "sklearn", "scipy", "numpy", "pandas", "joblib", "psutil",
    "safetensors", "huggingface_hub", "regex", "tqdm", "datasets", "pyarrow",
]


def log(msg):
    print("[phase6] %s" % msg, flush=True)


def build():
    os.makedirs(WORK, exist_ok=True)
    entry = os.path.join(WORK, "LargeModelDemoText.py")
    shutil.copy2(SRC, entry)
    log("entry copied: %s (%d bytes)" % (entry, os.path.getsize(entry)))

    args = [PY, "-m", "PyInstaller", "--noconfirm", "--clean",
            "--name", "LargeModelDemoText",
            "--workpath", os.path.join(WORK, "work"),
            "--specpath", WORK,
            "--distpath", DIST]
    for pkg in COLLECT_ALL:
        args += ["--collect-all", pkg]
    args.append(entry)
    log("pyinstaller args: %s" % " ".join(args[3:]))
    t0 = time.time()
    p = subprocess.run(args, cwd=WORK, stdout=sys.stdout, stderr=subprocess.STDOUT)
    log("pyinstaller exit=%s in %.1f s" % (p.returncode, time.time() - t0))
    return p.returncode


def tree_size(path):
    total = 0
    files = 0
    for root, _dirs, names in os.walk(path):
        for n in names:
            try:
                total += os.path.getsize(os.path.join(root, n))
                files += 1
            except OSError:
                pass
    return files, total


def test():
    app = os.path.join(DIST, "LargeModelDemoText", "LargeModelDemoText.exe")
    if not os.path.exists(app):
        log("FAIL: %s not found" % app)
        return 2
    files, size = tree_size(os.path.join(DIST, "LargeModelDemoText"))
    log("dist: %d files / %.2f GB" % (files, size / 1024 ** 3))
    log("exe : %d bytes" % os.path.getsize(app))

    work = os.path.join(BUILD, "smoke_built")
    os.makedirs(work, exist_ok=True)
    db = os.path.join(work, "SqliteData_Temp.sqlite")
    shutil.copy2(r"C:\DemoText客户端\Resources\BigModel\SqliteData_Temp.sqlite", db)
    report = os.path.join(work, "TestResultTxt.txt")
    save = os.path.join(work, "final_model")
    os.makedirs(save, exist_ok=True)
    for f in (report,):
        if os.path.exists(f):
            os.remove(f)

    args = [app,
            "--device", "GPU",
            "--sqliteFilePath", db,
            "--TestResultTxt", report,
            "--model_path", PRETRAIN,
            "--modelSavePath", save,
            "--isUploadModel", "0",
            "--ModelName", "FinBert-base-zh",
            "--isSetSeed", "seed",
            "--seed", "42",
            "--EPOCHS", "1",
            "--MAX_LENGTH", "64",
            "--BATCH_SIZE", "2",
            "--learning_rate", "2e-5",
            "--weight_decay", "0.01",
            "--warmup_steps", "50",
            "--gradient_clip_norm", "0",
            "--dropout_prob", "0.1",
            "--total_count", "7"]
    logfile = os.path.join(work, "kernel_built.log")
    log("running built exe (log -> %s)" % logfile)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    t0 = time.time()
    with open(logfile, "w", encoding="utf-8", errors="replace") as fh:
        p = subprocess.run(args, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=work)
    log("built exe exit=%s in %.1f s" % (p.returncode, time.time() - t0))

    text = open(logfile, encoding="utf-8", errors="replace").read()
    for line in text.splitlines():
        low = line.lower()
        if ("cuda" in low or "error" in low or "traceback" in low or "出错" in line
                or "device" in low or "开始" in line):
            log("  | " + line.strip()[:200])

    log("--- report file ---")
    if os.path.exists(report):
        body = open(report, encoding="utf-8", errors="replace").read()
        log("report exists, %d bytes" % len(body))
        for line in body.splitlines():
            log("  > " + line)
    else:
        log("report file MISSING (kernel never wrote it)")
    return 0


def main():
    if "--no-build" not in sys.argv:
        rc = build()
        if rc != 0:
            log("build failed, skipping test")
            return rc
    return test()


if __name__ == "__main__":
    sys.exit(main())
