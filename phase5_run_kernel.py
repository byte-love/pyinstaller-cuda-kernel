"""Phase 5: run the training kernel (from source, in py311) against a real
SQLite split, exactly the way the client launches it, and report what happened.

The first run is expected to reproduce the empty-metrics bug (testing_set holds
a single class while the training set holds two) - that is the 问题4/5 repro.
After the labels= patch the same run must produce a filled report.

Usage: phase5_run_kernel.py <tag>
"""
import os
import shutil
import sqlite3
import subprocess
import sys
import time

BUILD = r"D:\aitext_kernel_build"
PY = os.path.join(BUILD, "py311", "python.exe")
KERNEL = r".\DemoText客户端svn版本\大模型\LargeModelDemoText.py"
SRC_DB = r"C:\DemoText客户端\Resources\BigModel\SqliteData_Temp.sqlite"
PRETRAIN = r"C:\DemoText客户端\Resources\BigModel\PreTrainingModel\FinBert-base-zh"


def log(msg):
    print("[phase5] %s" % msg, flush=True)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "run1"
    work = os.path.join(BUILD, "smoke_" + tag)
    os.makedirs(work, exist_ok=True)
    db = os.path.join(work, "SqliteData_Temp.sqlite")
    shutil.copy2(SRC_DB, db)

    con = sqlite3.connect(db)
    counts = {}
    for t in ("training_set", "testing_set", "dev_set"):
        try:
            counts[t] = con.execute("select count(*) from %s" % t).fetchone()[0]
        except Exception as e:
            counts[t] = "n/a (%s)" % e
    try:
        rows = con.execute("select TextLabelName, count(*) from training_set group by TextLabelName").fetchall()
        train_labels = rows
    except Exception as e:
        train_labels = e
    try:
        rows = con.execute("select TextLabelName, count(*) from testing_set group by TextLabelName").fetchall()
        test_labels = rows
    except Exception as e:
        test_labels = e
    con.close()
    log("db counts: %s" % counts)
    log("training_set labels: %s" % (train_labels,))
    log("testing_set labels : %s" % (test_labels,))

    total = counts.get("training_set")
    if not isinstance(total, int):
        log("FAIL: cannot read training_set count")
        return 2

    report = os.path.join(work, "TestResultTxt.txt")
    save = os.path.join(work, "final_model")
    os.makedirs(save, exist_ok=True)
    if os.path.exists(report):
        os.remove(report)

    args = [
        PY, KERNEL,
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
        "--total_count", str(total),
    ]
    logfile = os.path.join(work, "kernel_%s.log" % tag)
    log("launching kernel (log -> %s)" % logfile)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    t0 = time.time()
    with open(logfile, "w", encoding="utf-8", errors="replace") as fh:
        p = subprocess.run(args, stdout=fh, stderr=subprocess.STDOUT, env=env)
    log("kernel exit=%s in %.1f s" % (p.returncode, time.time() - t0))

    text = open(logfile, encoding="utf-8", errors="replace").read()
    log("--- kernel log: CUDA / device evidence ---")
    for line in text.splitlines():
        low = line.lower()
        if ("cuda" in low or "device" in low or "gpu" in low or "error" in low
                or "traceback" in low or "出错" in line or "开始" in line):
            log("  | " + line.strip()[:180])

    log("--- report file ---")
    if os.path.exists(report):
        body = open(report, encoding="utf-8", errors="replace").read()
        log("report exists, %d bytes" % len(body))
        for line in body.splitlines():
            log("  > " + line)
    else:
        log("report file MISSING (kernel never wrote it)")

    log("--- saved model dir ---")
    if os.path.isdir(save):
        log("  %s" % sorted(os.listdir(save)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
