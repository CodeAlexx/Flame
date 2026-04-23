#!/usr/bin/env python3
"""
generate_pytorch_fixtures.py — PyTorch parity + performance oracle for flame-core

Generates:
  1. Per-op fixtures: input/output tensors as .safetensors + timing as JSON
  2. Combined-pattern fixtures: real DiT block sequences timed end-to-end
  3. Strided-input fixtures: permuted/narrowed inputs to verify stride handling

Usage:
    python generate_pytorch_fixtures.py --output-dir flame-core/tests/pytorch_fixtures/
    python generate_pytorch_fixtures.py --output-dir flame-core/tests/pytorch_fixtures/ --ops-only
    python generate_pytorch_fixtures.py --output-dir flame-core/tests/pytorch_fixtures/ --patterns-only
    python generate_pytorch_fixtures.py --output-dir flame-core/tests/pytorch_fixtures/ --op silu,gelu,add

Requires: torch, safetensors
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any, Tuple

import torch
import torch.nn.functional as F
from safetensors.torch import save_file, load_file


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEVICE = "cuda"
DTYPE = torch.bfloat16
WARMUP_ITERS = 20
BENCH_ITERS = 100
SEED = 42

# Shapes are testability-tuned proxies of real diffusion shapes —
# shapes are named after their production counterparts so test output
# stays mnemonic, but every dim is downsized to keep total fixture
# size around 50-200 MB (not committed to git; see README). Originals
# in git history (ef1da68 parent) if anyone needs production-scale.
#
# The SDPA head_dim stays ≤ 128 because PyTorch's F.scaled_dot_product_attention
# refuses to run with D > 128 ("No available kernel").
SHAPES = {
    "klein_hidden":       (1, 256, 1024),        # proxy for (4, 1024, 4608)
    "klein_attn_qkv":     (1, 4, 256, 128),      # proxy for (4, 16, 1024, 288) — D clamped to 128
    "klein_mlp":          (1, 256, 2048),        # proxy for (4, 1024, 18432)
    "sdxl_hidden":        (1, 512, 512),         # proxy for (2, 4096, 2560)
    "sdxl_attn":          (1, 8, 512, 64),       # proxy for (2, 40, 4096, 64)
    "chroma_hidden":      (1, 256, 768),         # proxy for (2, 1024, 3072)
    "sd3_hidden":         (1, 256, 384),         # proxy for (2, 1024, 1536)
    "qwen_hidden":        (1, 512, 768),         # proxy for (1, 4096, 3072)
    "small_square":       (1, 512, 512),         # microbench shape
    "tiny":               (1, 64, 64),           # sanity check
}

# SDPA-specific shapes: (batch, heads, seq_len, head_dim)
# head_dim ≤ 128 is a PyTorch constraint (same as flame-core's cuDNN
# support range {64, 96, 128}), not a test preference.
SDPA_SHAPES = {
    "klein_sdpa":         (1, 4, 256, 128),
    "sdxl_sdpa":          (1, 8, 512, 64),
    "chroma_sdpa":        (1, 6, 256, 128),
    "sd3_sdpa":           (1, 6, 256, 64),
    "qwen_sdpa":          (1, 6, 512, 128),
}

# Matmul shapes. Input is `[..., in_features]`; weight is PyTorch's
# `F.linear` layout `[out_features, in_features]`. Bias is `[out_features]`.
# Output is `[..., out_features]`. Matches the `torch.nn.Linear` weight
# convention (N, K).
MATMUL_SHAPES = {
    "klein_linear":       ((1, 256, 1024), (1024, 1024)),
    "klein_mlp_up":       ((1, 256, 1024), (2048, 1024)),
    "klein_mlp_down":     ((1, 256, 2048), (1024, 2048)),
    "sdxl_linear":        ((1, 512, 512), (512, 512)),
    "chroma_linear":      ((1, 256, 768), (768, 768)),
    # Production-scale rows to check whether the small-shape gap narrows
    # at Klein's real dimensions. At ~170 MB weight the per-call overhead
    # (transpose-materialize + heuristic) drops from ~75% of compute to
    # ~10%, so the ratio should converge toward 1.0×.
    "klein_prod_linear":  ((1, 1024, 4608), (4608, 4608)),
    "klein_prod_mlp_up":  ((1, 1024, 4608), (18432, 4608)),
}

# Conv2d shapes: (N, C_in, H, W), (C_out, C_in, kH, kW)
CONV2D_SHAPES = {
    "sdxl_conv_entry":    ((1, 4, 64, 64), (64, 4, 3, 3)),
    "sdxl_conv_mid":      ((1, 256, 16, 16), (256, 256, 3, 3)),
    "vae_conv_decode":    ((1, 64, 128, 128), (64, 64, 3, 3)),
    "vae_conv_out":       ((1, 32, 256, 256), (3, 32, 3, 3)),
}


# ---------------------------------------------------------------------------
# Timing utility
# ---------------------------------------------------------------------------

@dataclass
class TimingResult:
    op: str
    shape_name: str
    shape: list
    dtype: str
    median_us: float
    mean_us: float
    min_us: float
    max_us: float
    std_us: float
    warmup_iters: int
    bench_iters: int
    device: str
    notes: str = ""


def bench_op(fn: Callable, warmup: int = WARMUP_ITERS, iters: int = BENCH_ITERS) -> List[float]:
    """Returns list of per-iteration times in microseconds."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)  # ns → µs
    return times


def make_timing_result(op: str, shape_name: str, shape: tuple, times: List[float], notes: str = "") -> TimingResult:
    import statistics
    return TimingResult(
        op=op,
        shape_name=shape_name,
        shape=list(shape),
        dtype=str(DTYPE),
        median_us=statistics.median(times),
        mean_us=statistics.mean(times),
        min_us=min(times),
        max_us=max(times),
        std_us=statistics.stdev(times) if len(times) > 1 else 0.0,
        warmup_iters=WARMUP_ITERS,
        bench_iters=BENCH_ITERS,
        device=torch.cuda.get_device_name(0),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Tensor generation
# ---------------------------------------------------------------------------

def make_input(shape: tuple, seed: int = SEED) -> torch.Tensor:
    """Deterministic BF16 input on CUDA."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=g, dtype=torch.float32).to(dtype=DTYPE, device=DEVICE)


def make_input_pair(shape: tuple, seed: int = SEED) -> Tuple[torch.Tensor, torch.Tensor]:
    a = make_input(shape, seed)
    b = make_input(shape, seed + 1)
    return a, b


# ---------------------------------------------------------------------------
# Section 1: Per-op fixtures — unary
# ---------------------------------------------------------------------------

UNARY_OPS = {
    "silu":     lambda x: F.silu(x),
    "gelu":     lambda x: F.gelu(x, approximate="tanh"),
    "gelu_exact": lambda x: F.gelu(x, approximate="none"),
    "relu":     lambda x: F.relu(x),
    "sigmoid":  lambda x: torch.sigmoid(x),
    "tanh":     lambda x: torch.tanh(x),
    "abs":      lambda x: torch.abs(x),
    "neg":      lambda x: torch.neg(x),
    "square":   lambda x: torch.square(x),
    # Transcendentals assume the input was pre-sanitized by SANITIZE_UNARY_INPUT
    # below. The `input` saved to the fixture is the POST-sanitize tensor so
    # flame-core sees the same values PyTorch applied the op to — otherwise
    # flame-core's log(negative) etc. returns NaN while PyTorch's fixture
    # computed log(|x|+eps) on the sanitized version.
    "exp":      lambda x: torch.exp(x),
    "log":      lambda x: torch.log(x),
    "sqrt":     lambda x: torch.sqrt(x),
    "rsqrt":    lambda x: torch.rsqrt(x),
    "recip":    lambda x: torch.reciprocal(x),
}

# Per-op input sanitization. The fixture writer applies these BEFORE both
# the PyTorch op and the save, so Rust-side load gets the same sanitized
# values and its native op produces matching output.
SANITIZE_UNARY_INPUT = {
    "exp":   lambda x: x.clamp(-10, 10),        # avoid BF16 inf
    "log":   lambda x: x.abs() + 1e-6,           # strictly positive for log
    "sqrt":  lambda x: x.abs() + 1e-6,           # non-negative for sqrt
    "rsqrt": lambda x: x.abs() + 1e-6,           # strictly positive
    "recip": lambda x: x.clamp(min=0.001),       # away from zero
}

BINARY_OPS = {
    "add":      lambda a, b: a + b,
    "sub":      lambda a, b: a - b,
    "mul":      lambda a, b: a * b,
    # div: b is pre-sanitized by SANITIZE_BINARY_INPUT_B (same reasoning as
    # unary — save the sanitized tensor so Rust sees what PyTorch saw).
    "div":      lambda a, b: a / b,
    "maximum":  lambda a, b: torch.maximum(a, b),
    "minimum":  lambda a, b: torch.minimum(a, b),
}

SANITIZE_BINARY_INPUT_B = {
    "div": lambda b: b.abs() + 1e-3,
}

SCALAR_OPS = {
    "add_scalar":  lambda x: x + 0.5,
    "mul_scalar":  lambda x: x * 2.0,
}

# Comparison ops return BF16 0.0/1.0 sentinels (not bool/uint8) — flame-core
# uses BF16 as the output dtype for comparisons (CONVENTIONS.md) and the
# safetensors loader doesn't accept `bool`/`uint8`. Saving the PyTorch
# result as BF16 makes the cross-framework comparison a simple BF16
# equality check.
COMPARISON_OPS = {
    "gt": lambda a, b: (a > b).to(torch.bfloat16),
    "ge": lambda a, b: (a >= b).to(torch.bfloat16),
    "lt": lambda a, b: (a < b).to(torch.bfloat16),
    "le": lambda a, b: (a <= b).to(torch.bfloat16),
    "eq": lambda a, b: (a == b).to(torch.bfloat16),
    "ne": lambda a, b: (a != b).to(torch.bfloat16),
}


def generate_unary_fixtures(out_dir: Path, op_filter: Optional[List[str]] = None):
    """Generate fixtures for all unary ops across all shapes."""
    results = []
    ops = {k: v for k, v in UNARY_OPS.items() if not op_filter or k in op_filter}

    for op_name, op_fn in ops.items():
        for shape_name, shape in SHAPES.items():
            print(f"  unary/{op_name}/{shape_name}...", end=" ", flush=True)
            x = make_input(shape)
            if op_name in SANITIZE_UNARY_INPUT:
                x = SANITIZE_UNARY_INPUT[op_name](x)
            y = op_fn(x)

            # Save fixture
            fixture_dir = out_dir / "unary" / op_name
            fixture_dir.mkdir(parents=True, exist_ok=True)
            save_file(
                {"input": x.contiguous().cpu(), "output": y.contiguous().cpu()},
                str(fixture_dir / f"{shape_name}.safetensors")
            )

            # Bench
            times = bench_op(lambda: op_fn(x))
            result = make_timing_result(f"unary/{op_name}", shape_name, shape, times)
            results.append(result)
            print(f"{result.median_us:.1f} µs")

            # Strided variant: permuted input
            if len(shape) >= 3:
                dims = list(range(len(shape)))
                dims[-1], dims[-2] = dims[-2], dims[-1]
                x_perm = x.permute(*dims)
                y_perm = op_fn(x_perm)
                save_file(
                    {"input": x_perm.contiguous().cpu(), "output": y_perm.contiguous().cpu()},
                    str(fixture_dir / f"{shape_name}_permuted.safetensors")
                )

                times_perm = bench_op(lambda: op_fn(x_perm))
                result_perm = make_timing_result(
                    f"unary/{op_name}", f"{shape_name}_permuted", shape, times_perm,
                    notes="input is permuted (non-contiguous)"
                )
                results.append(result_perm)
                print(f"    permuted: {result_perm.median_us:.1f} µs")

    return results


def generate_binary_fixtures(out_dir: Path, op_filter: Optional[List[str]] = None):
    """Generate fixtures for binary ops."""
    results = []
    ops = {k: v for k, v in BINARY_OPS.items() if not op_filter or k in op_filter}

    for op_name, op_fn in ops.items():
        for shape_name, shape in SHAPES.items():
            print(f"  binary/{op_name}/{shape_name}...", end=" ", flush=True)
            a, b = make_input_pair(shape)
            if op_name in SANITIZE_BINARY_INPUT_B:
                b = SANITIZE_BINARY_INPUT_B[op_name](b)
            y = op_fn(a, b)

            fixture_dir = out_dir / "binary" / op_name
            fixture_dir.mkdir(parents=True, exist_ok=True)
            save_file(
                {"input_a": a.cpu(), "input_b": b.cpu(), "output": y.contiguous().cpu()},
                str(fixture_dir / f"{shape_name}.safetensors")
            )

            times = bench_op(lambda: op_fn(a, b))
            result = make_timing_result(f"binary/{op_name}", shape_name, shape, times)
            results.append(result)
            print(f"{result.median_us:.1f} µs")

            # Broadcast: (shape) + (1, 1, shape[-1])
            if len(shape) >= 3:
                b_bc = make_input((1, 1, shape[-1]), seed=SEED + 1)
                y_bc = op_fn(a, b_bc)
                save_file(
                    {"input_a": a.cpu(), "input_b": b_bc.cpu(), "output": y_bc.contiguous().cpu()},
                    str(fixture_dir / f"{shape_name}_broadcast.safetensors")
                )
                times_bc = bench_op(lambda: op_fn(a, b_bc))
                result_bc = make_timing_result(
                    f"binary/{op_name}", f"{shape_name}_broadcast", shape, times_bc,
                    notes="b is broadcast"
                )
                results.append(result_bc)
                print(f"    broadcast: {result_bc.median_us:.1f} µs")

    return results


def generate_scalar_fixtures(out_dir: Path, op_filter: Optional[List[str]] = None):
    """Generate fixtures for scalar ops."""
    results = []
    ops = {k: v for k, v in SCALAR_OPS.items() if not op_filter or k in op_filter}

    for op_name, op_fn in ops.items():
        for shape_name, shape in SHAPES.items():
            print(f"  scalar/{op_name}/{shape_name}...", end=" ", flush=True)
            x = make_input(shape)
            y = op_fn(x)

            fixture_dir = out_dir / "scalar" / op_name
            fixture_dir.mkdir(parents=True, exist_ok=True)
            save_file(
                {"input": x.cpu(), "output": y.contiguous().cpu()},
                str(fixture_dir / f"{shape_name}.safetensors")
            )

            times = bench_op(lambda: op_fn(x))
            result = make_timing_result(f"scalar/{op_name}", shape_name, shape, times)
            results.append(result)
            print(f"{result.median_us:.1f} µs")

    return results


def generate_comparison_fixtures(out_dir: Path, op_filter: Optional[List[str]] = None):
    """Generate fixtures for comparison ops (output is u8)."""
    results = []
    ops = {k: v for k, v in COMPARISON_OPS.items() if not op_filter or k in op_filter}

    for op_name, op_fn in ops.items():
        for shape_name, shape in SHAPES.items():
            print(f"  comparison/{op_name}/{shape_name}...", end=" ", flush=True)
            a, b = make_input_pair(shape)
            y = op_fn(a, b)

            fixture_dir = out_dir / "comparison" / op_name
            fixture_dir.mkdir(parents=True, exist_ok=True)
            save_file(
                {"input_a": a.cpu(), "input_b": b.cpu(), "output": y.cpu()},
                str(fixture_dir / f"{shape_name}.safetensors")
            )

            times = bench_op(lambda: op_fn(a, b))
            result = make_timing_result(f"comparison/{op_name}", shape_name, shape, times)
            results.append(result)
            print(f"{result.median_us:.1f} µs")

    return results


# ---------------------------------------------------------------------------
# Section 2: SDPA
# ---------------------------------------------------------------------------

def generate_sdpa_fixtures(out_dir: Path):
    """Generate SDPA fixtures — the attention hot path."""
    results = []

    for shape_name, (B, H, N, D) in SDPA_SHAPES.items():
        print(f"  sdpa/{shape_name} (B={B}, H={H}, N={N}, D={D})...", end=" ", flush=True)

        q = make_input((B, H, N, D), seed=SEED)
        k = make_input((B, H, N, D), seed=SEED + 1)
        v = make_input((B, H, N, D), seed=SEED + 2)

        # Standard contiguous SDPA
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION):
            y = F.scaled_dot_product_attention(q, k, v)

        fixture_dir = out_dir / "sdpa"
        fixture_dir.mkdir(parents=True, exist_ok=True)
        save_file(
            {"q": q.cpu(), "k": k.cpu(), "v": v.cpu(), "output": y.contiguous().cpu()},
            str(fixture_dir / f"{shape_name}_contig.safetensors")
        )

        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION):
            times = bench_op(lambda: F.scaled_dot_product_attention(q, k, v))
        result = make_timing_result("sdpa", f"{shape_name}_contig", (B, H, N, D), times)
        results.append(result)
        print(f"{result.median_us:.1f} µs")

        # Strided SDPA: Q/K/V come from a permute (the real DiT pattern)
        # Simulate: [B, N, H, D] → permute(0,2,1,3) → [B, H, N, D] (strided)
        q_bhnd = make_input((B, N, H, D), seed=SEED + 10).permute(0, 2, 1, 3)
        k_bhnd = make_input((B, N, H, D), seed=SEED + 11).permute(0, 2, 1, 3)
        v_bhnd = make_input((B, N, H, D), seed=SEED + 12).permute(0, 2, 1, 3)

        assert not q_bhnd.is_contiguous(), "Should be strided after permute"

        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION):
            y_strided = F.scaled_dot_product_attention(q_bhnd, k_bhnd, v_bhnd)

        save_file(
            {
                "q": q_bhnd.contiguous().cpu(),
                "k": k_bhnd.contiguous().cpu(),
                "v": v_bhnd.contiguous().cpu(),
                "output": y_strided.contiguous().cpu(),
                # Save strides as metadata-only tensors (1D int64)
                "q_strides": torch.tensor(q_bhnd.stride(), dtype=torch.int64),
                "k_strides": torch.tensor(k_bhnd.stride(), dtype=torch.int64),
                "v_strides": torch.tensor(v_bhnd.stride(), dtype=torch.int64),
                "q_shape": torch.tensor(q_bhnd.shape, dtype=torch.int64),
            },
            str(fixture_dir / f"{shape_name}_strided.safetensors")
        )

        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION):
            times_s = bench_op(lambda: F.scaled_dot_product_attention(q_bhnd, k_bhnd, v_bhnd))
        result_s = make_timing_result(
            "sdpa", f"{shape_name}_strided", (B, H, N, D), times_s,
            notes="Q/K/V are permuted (non-contiguous) — simulates real DiT pattern"
        )
        results.append(result_s)
        print(f"    strided: {result_s.median_us:.1f} µs")

    return results


# ---------------------------------------------------------------------------
# Section 3: Matmul / Linear
# ---------------------------------------------------------------------------

def generate_matmul_fixtures(out_dir: Path):
    """Generate matmul fixtures with real model shapes."""
    results = []

    for shape_name, (input_shape, weight_shape) in MATMUL_SHAPES.items():
        # weight_shape is (N, K) per PyTorch Linear; N = out, K = in.
        N, K = weight_shape
        print(f"  matmul/{shape_name} {input_shape} x {weight_shape}...", end=" ", flush=True)

        x = make_input(input_shape, seed=SEED)
        w = make_input(weight_shape, seed=SEED + 1)
        bias = make_input((N,), seed=SEED + 2)

        # Linear with bias (the common case)
        y = F.linear(x, w, bias)

        fixture_dir = out_dir / "matmul"
        fixture_dir.mkdir(parents=True, exist_ok=True)
        save_file(
            {"input": x.cpu(), "weight": w.cpu(), "bias": bias.cpu(), "output": y.contiguous().cpu()},
            str(fixture_dir / f"{shape_name}.safetensors")
        )

        times = bench_op(lambda: F.linear(x, w, bias))
        result = make_timing_result("matmul/linear", shape_name, input_shape, times)
        results.append(result)
        print(f"{result.median_us:.1f} µs")

        # Matmul without bias: x @ w^T computes (*, K) @ (K, N) → (*, N).
        y_nobias = x @ w.T
        # bench below re-uses the same expression
        save_file(
            {"input": x.cpu(), "weight": w.cpu(), "output": y_nobias.contiguous().cpu()},
            str(fixture_dir / f"{shape_name}_nobias.safetensors")
        )

        times_nb = bench_op(lambda: x @ w.T)
        result_nb = make_timing_result("matmul/mm", f"{shape_name}_nobias", input_shape, times_nb)
        results.append(result_nb)
        print(f"    nobias: {result_nb.median_us:.1f} µs")

    return results


# ---------------------------------------------------------------------------
# Section 4: Conv2d
# ---------------------------------------------------------------------------

def generate_conv2d_fixtures(out_dir: Path):
    """Generate conv2d fixtures with real model shapes."""
    results = []

    for shape_name, (input_shape, weight_shape) in CONV2D_SHAPES.items():
        C_out = weight_shape[0]
        print(f"  conv2d/{shape_name} {input_shape} x {weight_shape}...", end=" ", flush=True)

        x = make_input(input_shape, seed=SEED)
        w = make_input(weight_shape, seed=SEED + 1)
        bias = make_input((C_out,), seed=SEED + 2)

        y = F.conv2d(x, w, bias, padding=1)

        fixture_dir = out_dir / "conv2d"
        fixture_dir.mkdir(parents=True, exist_ok=True)
        save_file(
            {"input": x.cpu(), "weight": w.cpu(), "bias": bias.cpu(), "output": y.contiguous().cpu()},
            str(fixture_dir / f"{shape_name}.safetensors")
        )

        times = bench_op(lambda: F.conv2d(x, w, bias, padding=1))
        result = make_timing_result("conv2d", shape_name, input_shape, times)
        results.append(result)
        print(f"{result.median_us:.1f} µs")

    return results


# ---------------------------------------------------------------------------
# Section 5: Combined patterns — real DiT block sequences
# ---------------------------------------------------------------------------

def generate_pattern_fixtures(out_dir: Path):
    """
    Time real DiT block patterns end-to-end.
    These are the sequences that matter for step time — not isolated ops.
    """
    results = []
    fixture_dir = out_dir / "patterns"
    fixture_dir.mkdir(parents=True, exist_ok=True)

    # --- Pattern 1: Klein double-block attention path ---
    # QKV projection → split → permute → RoPE → SDPA → permute back → output projection
    print("  pattern/klein_attention_path...", end=" ", flush=True)

    B, N, D = 4, 1024, 4608
    H, HD = 16, 288  # 16 heads, 288 head_dim (4608 / 16)

    x = make_input((B, N, D), seed=SEED)
    w_qkv = make_input((D * 3, D), seed=SEED + 1)
    w_out = make_input((D, D), seed=SEED + 2)
    b_qkv = make_input((D * 3,), seed=SEED + 3)
    b_out = make_input((D,), seed=SEED + 4)

    # Precompute RoPE freqs (simplified — real RoPE is more complex)
    freqs = make_input((1, 1, N, HD), seed=SEED + 5)

    def klein_attn_pattern():
        qkv = F.linear(x, w_qkv, b_qkv)                     # [B, N, 3*D]
        q, k, v = qkv.chunk(3, dim=-1)                        # 3x [B, N, D]
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3)          # [B, H, N, HD]
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3)
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3)
        # Simplified RoPE (real is complex rotation, this is a stand-in)
        q = q * freqs
        k = k * freqs
        # SDPA
        attn = F.scaled_dot_product_attention(q, k, v)        # [B, H, N, HD]
        attn = attn.permute(0, 2, 1, 3).reshape(B, N, D)     # [B, N, D]
        out = F.linear(attn, w_out, b_out)                     # [B, N, D]
        return out

    y_attn = klein_attn_pattern()
    save_file(
        {"input": x.cpu(), "output": y_attn.contiguous().cpu()},
        str(fixture_dir / "klein_attention_path.safetensors")
    )

    times = bench_op(klein_attn_pattern)
    result = make_timing_result("pattern/klein_attention", "full", (B, N, D), times)
    results.append(result)
    print(f"{result.median_us:.1f} µs")

    # --- Pattern 2: Klein MLP path ---
    # Linear → SwiGLU (split + silu + mul) → Linear
    print("  pattern/klein_mlp_path...", end=" ", flush=True)

    MLP_DIM = D * 4  # 18432
    w_up = make_input((MLP_DIM * 2, D), seed=SEED + 10)
    b_up = make_input((MLP_DIM * 2,), seed=SEED + 11)
    w_down = make_input((D, MLP_DIM), seed=SEED + 12)
    b_down = make_input((D,), seed=SEED + 13)

    def klein_mlp_pattern():
        gate_up = F.linear(x, w_up, b_up)                    # [B, N, 2*MLP]
        gate, up = gate_up.chunk(2, dim=-1)                    # 2x [B, N, MLP]
        hidden = F.silu(gate) * up                             # SwiGLU
        out = F.linear(hidden, w_down, b_down)                 # [B, N, D]
        return out

    y_mlp = klein_mlp_pattern()
    save_file(
        {"input": x.cpu(), "output": y_mlp.contiguous().cpu()},
        str(fixture_dir / "klein_mlp_path.safetensors")
    )

    times = bench_op(klein_mlp_pattern)
    result = make_timing_result("pattern/klein_mlp", "full", (B, N, D), times)
    results.append(result)
    print(f"{result.median_us:.1f} µs")

    # --- Pattern 3: DiT modulate + residual ---
    # AdaLN: modulate(norm(x), shift, scale) + compute + gate * result + residual
    print("  pattern/dit_modulate_residual...", end=" ", flush=True)

    shift = make_input((B, 1, D), seed=SEED + 20)
    scale = make_input((B, 1, D), seed=SEED + 21)
    gate = make_input((B, 1, D), seed=SEED + 22)
    residual = make_input((B, N, D), seed=SEED + 23)

    def dit_modulate_residual_pattern():
        # RMSNorm (simplified)
        norm = x / (x.float().pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-5).to(DTYPE)
        # Modulate
        modulated = norm * (1.0 + scale) + shift
        # Gate + residual
        out = residual + gate * modulated
        return out

    y_mod = dit_modulate_residual_pattern()
    save_file(
        {"input": x.cpu(), "residual": residual.cpu(), "output": y_mod.contiguous().cpu()},
        str(fixture_dir / "dit_modulate_residual.safetensors")
    )

    times = bench_op(dit_modulate_residual_pattern)
    result = make_timing_result("pattern/dit_modulate_residual", "full", (B, N, D), times)
    results.append(result)
    print(f"{result.median_us:.1f} µs")

    # --- Pattern 4: Full Klein double-block (attention + MLP + modulate + residual) ---
    print("  pattern/klein_double_block...", end=" ", flush=True)

    def klein_double_block():
        # Attention path
        attn_out = klein_attn_pattern()
        # Residual + gate
        h = x + gate * attn_out
        # MLP path on the residual-added result
        mlp_in = h
        gate_up = F.linear(mlp_in, w_up, b_up)
        g, u = gate_up.chunk(2, dim=-1)
        mlp_out = F.linear(F.silu(g) * u, w_down, b_down)
        out = h + gate * mlp_out
        return out

    y_block = klein_double_block()
    save_file(
        {"input": x.cpu(), "output": y_block.contiguous().cpu()},
        str(fixture_dir / "klein_double_block.safetensors")
    )

    times = bench_op(klein_double_block)
    result = make_timing_result("pattern/klein_double_block", "full", (B, N, D), times)
    results.append(result)
    print(f"{result.median_us:.1f} µs")

    # --- Pattern 5: Permute → narrow → linear (the view chain that matters) ---
    print("  pattern/permute_narrow_linear...", end=" ", flush=True)

    x_pnl = make_input((B, N, D), seed=SEED + 30)
    # F.linear wants weight as (out, in). After narrow, input is [B, N, D//2]
    # (in=D//2); output should be [B, N, D] (out=D) → weight shape (D, D//2).
    w_pnl = make_input((D, D // 2), seed=SEED + 31)

    def permute_narrow_linear():
        # Permute
        h = x_pnl.permute(0, 2, 1)          # [B, D, N]
        # Narrow (take first half of channels)
        h = h.narrow(1, 0, D // 2)           # [B, D//2, N]
        # Permute back for linear
        h = h.permute(0, 2, 1)               # [B, N, D//2]
        # Linear
        out = F.linear(h, w_pnl)             # [B, N, D]
        return out

    y_pnl = permute_narrow_linear()
    save_file(
        {"input": x_pnl.cpu(), "output": y_pnl.contiguous().cpu()},
        str(fixture_dir / "permute_narrow_linear.safetensors")
    )

    times = bench_op(permute_narrow_linear)
    result = make_timing_result("pattern/permute_narrow_linear", "full", (B, N, D), times)
    results.append(result)
    print(f"{result.median_us:.1f} µs")

    # --- Pattern 6: SDPA backward (training path) ---
    print("  pattern/sdpa_backward...", end=" ", flush=True)

    B_t, H_t, N_t, D_t = 2, 16, 512, 64  # Smaller for backward (memory)
    q_t = make_input((B_t, H_t, N_t, D_t), seed=SEED + 40).requires_grad_(True)
    k_t = make_input((B_t, H_t, N_t, D_t), seed=SEED + 41).requires_grad_(True)
    v_t = make_input((B_t, H_t, N_t, D_t), seed=SEED + 42).requires_grad_(True)

    def sdpa_fwd_bwd():
        q_t.grad = None
        k_t.grad = None
        v_t.grad = None
        o = F.scaled_dot_product_attention(q_t, k_t, v_t)
        loss = o.sum()
        loss.backward()
        return q_t.grad, k_t.grad, v_t.grad

    dq, dk, dv = sdpa_fwd_bwd()
    save_file(
        {
            "q": q_t.detach().cpu(), "k": k_t.detach().cpu(), "v": v_t.detach().cpu(),
            "dq": dq.cpu(), "dk": dk.cpu(), "dv": dv.cpu(),
        },
        str(fixture_dir / "sdpa_backward.safetensors")
    )

    times = bench_op(sdpa_fwd_bwd)
    result = make_timing_result("pattern/sdpa_backward", "full", (B_t, H_t, N_t, D_t), times,
                                 notes="forward + backward, sum() loss")
    results.append(result)
    print(f"{result.median_us:.1f} µs")

    return results


# ---------------------------------------------------------------------------
# Section 6: RMSNorm (fused op comparison)
# ---------------------------------------------------------------------------

def generate_rmsnorm_fixtures(out_dir: Path):
    """Compare PyTorch's unfused RMSNorm vs what flame-core fuses."""
    results = []
    fixture_dir = out_dir / "fused"
    fixture_dir.mkdir(parents=True, exist_ok=True)

    for shape_name in ["klein_hidden", "chroma_hidden", "sdxl_hidden"]:
        shape = SHAPES[shape_name]
        print(f"  fused/rmsnorm/{shape_name}...", end=" ", flush=True)

        x = make_input(shape, seed=SEED)
        weight = make_input((shape[-1],), seed=SEED + 1)
        eps = 1e-5

        def rmsnorm_unfused():
            variance = x.float().pow(2).mean(dim=-1, keepdim=True)
            normed = x * torch.rsqrt(variance + eps).to(DTYPE)
            return normed * weight

        y = rmsnorm_unfused()
        save_file(
            {"input": x.cpu(), "weight": weight.cpu(), "output": y.contiguous().cpu()},
            str(fixture_dir / f"rmsnorm_{shape_name}.safetensors")
        )

        times = bench_op(rmsnorm_unfused)
        result = make_timing_result("fused/rmsnorm", shape_name, shape, times,
                                     notes="PyTorch unfused (variance → rsqrt → mul → mul). flame-core has fused_rms_norm_bf16.")
        results.append(result)
        print(f"{result.median_us:.1f} µs")

    return results


# ---------------------------------------------------------------------------
# Section 7: SwiGLU (fused op comparison)
# ---------------------------------------------------------------------------

def generate_swiglu_fixtures(out_dir: Path):
    """Compare PyTorch's unfused SwiGLU vs flame-core's fused version."""
    results = []
    fixture_dir = out_dir / "fused"
    fixture_dir.mkdir(parents=True, exist_ok=True)

    for shape_name in ["klein_hidden", "chroma_hidden"]:
        shape = SHAPES[shape_name]
        D = shape[-1]
        gate_up_shape = (*shape[:-1], D * 2)
        print(f"  fused/swiglu/{shape_name}...", end=" ", flush=True)

        gate_up = make_input(gate_up_shape, seed=SEED)

        def swiglu_unfused():
            gate, up = gate_up.chunk(2, dim=-1)
            return F.silu(gate) * up

        y = swiglu_unfused()
        save_file(
            {"input": gate_up.cpu(), "output": y.contiguous().cpu()},
            str(fixture_dir / f"swiglu_{shape_name}.safetensors")
        )

        times = bench_op(swiglu_unfused)
        result = make_timing_result("fused/swiglu", shape_name, gate_up_shape, times,
                                     notes="PyTorch unfused (chunk → silu → mul). flame-core has swiglu_fused_bf16.")
        results.append(result)
        print(f"{result.median_us:.1f} µs")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate PyTorch parity + perf fixtures for flame-core")
    parser.add_argument("--output-dir", type=str, default="flame-core/tests/pytorch_fixtures/",
                        help="Output directory for fixtures")
    parser.add_argument("--ops-only", action="store_true", help="Only generate per-op fixtures")
    parser.add_argument("--patterns-only", action="store_true", help="Only generate pattern fixtures")
    parser.add_argument("--op", type=str, default=None,
                        help="Comma-separated list of specific ops to generate (e.g. silu,gelu,add)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    op_filter = args.op.split(",") if args.op else None

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"cuDNN: {torch.backends.cudnn.version()}")
    print(f"Output: {out_dir}")
    print(f"Dtype: {DTYPE}")
    print(f"Seed: {SEED}")
    print(f"Warmup: {WARMUP_ITERS}, Bench: {BENCH_ITERS}")
    print()

    all_results = []

    # Save environment info
    env_info = {
        "device": torch.cuda.get_device_name(0),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "dtype": str(DTYPE),
        "seed": SEED,
        "warmup_iters": WARMUP_ITERS,
        "bench_iters": BENCH_ITERS,
    }
    with open(out_dir / "environment.json", "w") as f:
        json.dump(env_info, f, indent=2)

    if not args.patterns_only:
        print("=" * 60)
        print("UNARY OPS")
        print("=" * 60)
        all_results.extend(generate_unary_fixtures(out_dir, op_filter))

        print()
        print("=" * 60)
        print("BINARY OPS")
        print("=" * 60)
        all_results.extend(generate_binary_fixtures(out_dir, op_filter))

        print()
        print("=" * 60)
        print("SCALAR OPS")
        print("=" * 60)
        all_results.extend(generate_scalar_fixtures(out_dir, op_filter))

        print()
        print("=" * 60)
        print("COMPARISON OPS")
        print("=" * 60)
        all_results.extend(generate_comparison_fixtures(out_dir, op_filter))

        print()
        print("=" * 60)
        print("SDPA")
        print("=" * 60)
        all_results.extend(generate_sdpa_fixtures(out_dir))

        print()
        print("=" * 60)
        print("MATMUL / LINEAR")
        print("=" * 60)
        all_results.extend(generate_matmul_fixtures(out_dir))

        print()
        print("=" * 60)
        print("CONV2D")
        print("=" * 60)
        all_results.extend(generate_conv2d_fixtures(out_dir))

    if not args.ops_only:
        print()
        print("=" * 60)
        print("COMBINED PATTERNS")
        print("=" * 60)
        all_results.extend(generate_pattern_fixtures(out_dir))

        print()
        print("=" * 60)
        print("FUSED OPS (flame-core advantage)")
        print("=" * 60)
        all_results.extend(generate_rmsnorm_fixtures(out_dir))
        all_results.extend(generate_swiglu_fixtures(out_dir))

    # Save all timing results
    timing_file = out_dir / "timing_results.json"
    with open(timing_file, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)

    # Print summary table
    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"{'op':<40} {'shape':<25} {'median µs':>12} {'notes'}")
    print("-" * 100)
    for r in all_results:
        notes = f"  {r.notes}" if r.notes else ""
        print(f"{r.op:<40} {r.shape_name:<25} {r.median_us:>12.1f}{notes}")

    print()
    print(f"Total fixtures: {len(all_results)}")
    print(f"Timing saved to: {timing_file}")
    print(f"Fixtures saved to: {out_dir}")
    print()
    print("Run flame-core parity tests with:")
    print(f"  PYTORCH_FIXTURES={out_dir} cargo test --release --test pytorch_parity")


if __name__ == "__main__":
    main()
