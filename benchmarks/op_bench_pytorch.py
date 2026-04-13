"""
Op-level benchmark: PyTorch reference.
Measures forward and backward for each individual op at Z-Image Base shapes.

Usage:
    python benchmarks/op_bench_pytorch.py
    python benchmarks/op_bench_pytorch.py --csv   # machine-readable output
"""
from __future__ import annotations

import argparse
import statistics
import sys

import torch


# ---------------------------------------------------------------------------
# Shapes matching Z-Image Base (hidden=1280, heads=20, head_dim=64,
# ffn=5120, seq=1024, batch=1).  All BF16 on CUDA.
# ---------------------------------------------------------------------------
DEVICE = "cuda"
DTYPE = torch.bfloat16

WARMUP = 100
ITERS = 200


def _sync():
    torch.cuda.synchronize()


def _median_us(times: list[float]) -> float:
    return statistics.median(times) * 1e6  # seconds → μs


# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------

def bench_forward(fn, warmup=WARMUP, iters=ITERS) -> float:
    """Return median forward time in μs."""
    for _ in range(warmup):
        fn()
    _sync()

    times = []
    for _ in range(iters):
        _sync()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        _sync()
        times.append(start.elapsed_time(end))  # ms
    return statistics.median(times) * 1000  # ms → μs


def bench_backward(setup_fn, warmup=WARMUP, iters=ITERS) -> float:
    """Return median backward time in μs.

    setup_fn() must return (output, grad_output) where output.requires_grad.
    The grad_output tensor should be pre-allocated.
    """
    # warmup
    for _ in range(warmup):
        out, go = setup_fn()
        out.backward(go, retain_graph=False)

    _sync()
    times = []
    for _ in range(iters):
        out, go = setup_fn()
        _sync()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out.backward(go, retain_graph=False)
        end.record()
        _sync()
        times.append(start.elapsed_time(end) * 1000)  # ms → μs
    return statistics.median(times)


# ---------------------------------------------------------------------------
# Op benchmarks
# ---------------------------------------------------------------------------

def _randn(*shape):
    return torch.randn(*shape, device=DEVICE, dtype=DTYPE)


def _randn_grad(*shape):
    return torch.randn(*shape, device=DEVICE, dtype=DTYPE, requires_grad=True)


results: list[tuple[str, float, float]] = []


def register(name: str, fwd_us: float, bwd_us: float):
    results.append((name, fwd_us, bwd_us))


# ---- Level 1: Elementwise ----

def bench_cast_bf16_to_fp32():
    x = _randn(1, 1024, 1280)

    fwd = bench_forward(lambda: x.float())

    def setup():
        inp = _randn_grad(1, 1024, 1280)
        out = inp.float()
        go = torch.randn_like(out)
        return out, go

    bwd = bench_backward(setup)
    register("Cast BF16→FP32", fwd, bwd)


def bench_cast_fp32_to_bf16():
    x = torch.randn(1, 1024, 1280, device=DEVICE, dtype=torch.float32)

    fwd = bench_forward(lambda: x.bfloat16())

    def setup():
        inp = torch.randn(1, 1024, 1280, device=DEVICE, dtype=torch.float32, requires_grad=True)
        out = inp.bfloat16().float()  # bfloat16 cast doesn't track grad alone; round-trip
        go = torch.randn_like(out)
        return out, go

    bwd = bench_backward(setup)
    register("Cast FP32→BF16", fwd, bwd)


def bench_abs():
    x = _randn(1, 1024, 1280)
    fwd = bench_forward(lambda: x.abs())

    def setup():
        inp = _randn_grad(1, 1024, 1280)
        out = inp.abs()
        go = torch.randn_like(out)
        return out, go

    bwd = bench_backward(setup)
    register("Abs", fwd, bwd)


def bench_add():
    a = _randn(1, 1024, 1280)
    b = _randn(1, 1024, 1280)
    fwd = bench_forward(lambda: a + b)

    def setup():
        a2 = _randn_grad(1, 1024, 1280)
        b2 = _randn_grad(1, 1024, 1280)
        out = a2 + b2
        go = _randn(1, 1024, 1280)
        return out, go

    bwd = bench_backward(setup)
    register("Add (residual)", fwd, bwd)


def bench_mul_scalar():
    x = _randn(1, 1024, 1280)
    fwd = bench_forward(lambda: x * 0.7071)

    def setup():
        inp = _randn_grad(1, 1024, 1280)
        out = inp * 0.7071
        go = _randn(1, 1024, 1280)
        return out, go

    bwd = bench_backward(setup)
    register("Mul (scalar)", fwd, bwd)


def bench_mul_elementwise():
    a = _randn(1, 1024, 1280)
    b = _randn(1, 1024, 1280)
    fwd = bench_forward(lambda: a * b)

    def setup():
        a2 = _randn_grad(1, 1024, 1280)
        b2 = _randn_grad(1, 1024, 1280)
        out = a2 * b2
        go = _randn(1, 1024, 1280)
        return out, go

    bwd = bench_backward(setup)
    register("Mul (elementwise)", fwd, bwd)


# ---- Level 2: Reshape / Memory ----

def bench_reshape():
    x = _randn(1, 1024, 1280)
    fwd = bench_forward(lambda: x.reshape(1, 1024, 20, 64))

    def setup():
        inp = _randn_grad(1, 1024, 1280)
        out = inp.reshape(1, 1024, 20, 64)
        go = _randn(1, 1024, 20, 64)
        return out, go

    bwd = bench_backward(setup)
    register("Reshape", fwd, bwd)


def bench_permute():
    x = _randn(1, 1024, 20, 64)
    fwd = bench_forward(lambda: x.permute(0, 2, 1, 3).contiguous())

    def setup():
        inp = _randn_grad(1, 1024, 20, 64)
        out = inp.permute(0, 2, 1, 3).contiguous()
        go = _randn(1, 20, 1024, 64)
        return out, go

    bwd = bench_backward(setup)
    register("Permute (0,2,1,3)", fwd, bwd)


# ---- Level 3: Activations ----

def bench_silu():
    x = _randn(1, 1024, 5120)
    fwd = bench_forward(lambda: torch.nn.functional.silu(x))

    def setup():
        inp = _randn_grad(1, 1024, 5120)
        out = torch.nn.functional.silu(inp)
        go = _randn(1, 1024, 5120)
        return out, go

    bwd = bench_backward(setup)
    register("SiLU", fwd, bwd)


def bench_gelu():
    x = _randn(1, 1024, 5120)
    fwd = bench_forward(lambda: torch.nn.functional.gelu(x))

    def setup():
        inp = _randn_grad(1, 1024, 5120)
        out = torch.nn.functional.gelu(inp)
        go = _randn(1, 1024, 5120)
        return out, go

    bwd = bench_backward(setup)
    register("GELU", fwd, bwd)


def bench_softmax():
    x = _randn(20, 1024, 1024)
    fwd = bench_forward(lambda: torch.nn.functional.softmax(x, dim=-1))

    def setup():
        inp = _randn_grad(20, 1024, 1024)
        out = torch.nn.functional.softmax(inp, dim=-1)
        go = _randn(20, 1024, 1024)
        return out, go

    bwd = bench_backward(setup)
    register("Softmax", fwd, bwd)


# ---- Level 4: Normalization ----

def bench_layer_norm():
    ln = torch.nn.LayerNorm(1280, device=DEVICE, dtype=DTYPE)
    x = _randn(1, 1024, 1280)
    fwd = bench_forward(lambda: ln(x))

    def setup():
        inp = _randn_grad(1, 1024, 1280)
        out = ln(inp)
        go = _randn(1, 1024, 1280)
        return out, go

    bwd = bench_backward(setup)
    register("LayerNorm", fwd, bwd)


# ---- Level 5: MatMul ----

def bench_matmul_proj():
    x = _randn(1, 1024, 1280)
    w = _randn(1280, 1280)
    fwd = bench_forward(lambda: x @ w)

    def setup():
        inp = _randn_grad(1, 1024, 1280)
        out = inp @ w
        go = _randn(1, 1024, 1280)
        return out, go

    bwd = bench_backward(setup)
    register("MatMul (proj)", fwd, bwd)


def bench_matmul_ffn():
    x = _randn(1, 1024, 1280)
    w = _randn(1280, 5120)
    fwd = bench_forward(lambda: x @ w)

    def setup():
        inp = _randn_grad(1, 1024, 1280)
        out = inp @ w
        go = _randn(1, 1024, 5120)
        return out, go

    bwd = bench_backward(setup)
    register("MatMul (FFN)", fwd, bwd)


def bench_bmm_qk():
    q = _randn(20, 1024, 64)
    k = _randn(20, 64, 1024)
    fwd = bench_forward(lambda: q @ k)

    def setup():
        q2 = _randn_grad(20, 1024, 64)
        k2 = _randn_grad(20, 64, 1024)
        out = q2 @ k2
        go = _randn(20, 1024, 1024)
        return out, go

    bwd = bench_backward(setup)
    register("BMM (QK^T)", fwd, bwd)


def bench_bmm_av():
    a = _randn(20, 1024, 1024)
    v = _randn(20, 1024, 64)
    fwd = bench_forward(lambda: a @ v)

    def setup():
        a2 = _randn_grad(20, 1024, 1024)
        v2 = _randn_grad(20, 1024, 64)
        out = a2 @ v2
        go = _randn(20, 1024, 64)
        return out, go

    bwd = bench_backward(setup)
    register("BMM (@V)", fwd, bwd)


# ---- Level 6: LoRA ----

def bench_lora_merge():
    lora_a = _randn(1280, 64)
    lora_b = _randn(64, 1280)
    base = _randn(1280, 1280)
    scale = 1.0

    def merge():
        return base + (lora_a @ lora_b) * scale

    fwd = bench_forward(merge)
    register("LoRA merge", fwd, 0.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", action="store_true")
    args = parser.parse_args()

    # GPU warmup — run matmuls to stabilize clocks
    print("Warming up GPU...", file=sys.stderr)
    a = _randn(2048, 2048)
    b = _randn(2048, 2048)
    for _ in range(50):
        _ = a @ b
    _sync()
    del a, b
    torch.cuda.empty_cache()

    print(f"Device: {torch.cuda.get_device_name()}", file=sys.stderr)
    print(f"PyTorch: {torch.__version__}", file=sys.stderr)
    print(f"Warmup: {WARMUP}, Iters: {ITERS}\n", file=sys.stderr)

    # Run all benchmarks
    bench_cast_bf16_to_fp32()
    bench_cast_fp32_to_bf16()
    bench_abs()
    bench_add()
    bench_mul_scalar()
    bench_mul_elementwise()
    bench_reshape()
    bench_permute()
    bench_silu()
    bench_gelu()
    bench_softmax()
    bench_layer_norm()
    bench_matmul_proj()
    bench_matmul_ffn()
    bench_bmm_qk()
    bench_bmm_av()
    bench_lora_merge()

    # Print results
    if args.csv:
        print("op,fwd_us,bwd_us,total_us")
        for name, fwd, bwd in results:
            print(f"{name},{fwd:.1f},{bwd:.1f},{fwd+bwd:.1f}")
    else:
        hdr = f"{'Op':<25} {'Fwd (μs)':>10} {'Bwd (μs)':>10} {'Total (μs)':>12}"
        print(hdr)
        print("-" * len(hdr))
        for name, fwd, bwd in results:
            bwd_str = f"{bwd:.1f}" if bwd > 0 else "—"
            print(f"{name:<25} {fwd:>10.1f} {bwd_str:>10} {fwd+bwd:>12.1f}")


if __name__ == "__main__":
    main()
