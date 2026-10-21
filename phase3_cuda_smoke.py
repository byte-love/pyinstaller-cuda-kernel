"""Phase 3: prove the freshly installed torch can actually run kernels on this
GPU (RTX 5070 Laptop, sm_120). torch.cuda.is_available() alone is not enough:
a cu118 build reports True on a 50-series card but fails later with
"no kernel image is available for execution on the device".
"""
import sys
import time

import torch

FAIL = []


def step(name, fn):
    t0 = time.time()
    try:
        out = fn()
        print("[phase3] OK   %-28s %-40s (%.2f s)" % (name, out, time.time() - t0), flush=True)
    except Exception as e:
        FAIL.append((name, repr(e)))
        print("[phase3] FAIL %-28s %s" % (name, e), flush=True)


def info():
    return "torch=%s cuda=%s avail=%s dev=%s cap=%s" % (
        torch.__version__, torch.version.cuda, torch.cuda.is_available(),
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-",
        torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "-",
    )


def matmul():
    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    c = a @ b
    return "sum=%.4f" % float(c.sum())


def conv():
    x = torch.randn(2, 3, 32, 32, device="cuda")
    m = torch.nn.Conv2d(3, 8, 3).cuda()
    return "out=%s" % (tuple(m(x).shape),)


def linear_bwd():
    x = torch.randn(64, 128, device="cuda")
    layer = torch.nn.Linear(128, 3).cuda()
    loss = layer(x).pow(2).mean()
    loss.backward()
    g = layer.weight.grad
    return "grad_norm=%.6f" % float(g.norm())


def softmax_ce():
    logits = torch.randn(16, 3, device="cuda", requires_grad=True)
    y = torch.randint(0, 3, (16,), device="cuda")
    return "loss=%.4f" % float(torch.nn.functional.cross_entropy(logits, y))


def mem():
    free, total = torch.cuda.mem_get_info()
    return "free=%.2fGB total=%.2fGB" % (free / 1024 ** 3, total / 1024 ** 3)


def main():
    step("torch/cuda info", info)
    step("cublas matmul 512x512", matmul)
    step("cudnn conv2d", conv)
    step("linear fwd+bwd", linear_bwd)
    step("cross_entropy", softmax_ce)
    step("gpu memory", mem)
    if FAIL:
        print("[phase3] RESULT: %d step(s) failed" % len(FAIL), flush=True)
        for n, e in FAIL:
            print("   - %s: %s" % (n, e), flush=True)
        return 1
    print("[phase3] RESULT: all CUDA kernel steps passed -> sm_120 usable", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
