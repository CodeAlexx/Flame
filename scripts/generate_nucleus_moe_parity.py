"""Generate Phase-5 PyTorch parity fixture for the Nucleus-Image MoE
expert-FFN forward.

Saves a single safetensors with everything the Rust test needs:
inputs, weights, and the PyTorch expected output. The Rust test in
`src/ops/nucleus_moe.rs` (test_nucleus_moe_parity_vs_pytorch) loads
this and asserts agreement within BF16 tolerance.

Output:
    flame-core/tests/pytorch_fixtures/moe/nucleus_moe_parity.safetensors

Run from the flame-core directory:
    python3 scripts/generate_nucleus_moe_parity.py

Reference: SwiGLUExperts._run_experts_grouped_mm + the routing block of
NucleusMoELayer.forward in
diffusers/models/transformers/transformer_nucleusmoe_image.py.
"""

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file


def main():
    # ------------------------------------------------------------------
    # Toy shape — same as the in-Rust scalar parity test in
    # src/ops/nucleus_moe.rs. Small enough to be fast, large enough that
    # BF16 accumulation stays sane on a real GPU.
    # ------------------------------------------------------------------
    B, E, S, D, INTER, CAPACITY = 1, 4, 8, 64, 64, 4
    ROUTE_SCALE = 2.5
    SEED = 12345
    DEVICE = "cuda"
    DTYPE = torch.bfloat16

    torch.manual_seed(SEED)
    g = torch.Generator(device=DEVICE).manual_seed(SEED)

    # ------------------------------------------------------------------
    # Inputs + weights. Drawn from a uniform distribution and rounded to
    # BF16 to match what the Rust test feeds to the kernels. Affinity is
    # the *post-softmax-then-transpose* tensor; we generate it directly
    # so the parity gate isn't muddied by softmax differences. (The
    # router matmul + softmax is a normal Tensor op in flame-core.)
    # ------------------------------------------------------------------
    x_flat = (
        (torch.rand(B * S, D, generator=g, device=DEVICE) - 0.5) * 0.5
    ).to(DTYPE)
    affinity = (torch.rand(B, E, S, generator=g, device=DEVICE) - 0.5) * 2.0
    gate_up_w = (
        (torch.rand(E, D, 2 * INTER, generator=g, device=DEVICE) - 0.5) * 0.1
    ).to(DTYPE)
    down_w = (
        (torch.rand(E, INTER, D, generator=g, device=DEVICE) - 0.5) * 0.1
    ).to(DTYPE)

    # ------------------------------------------------------------------
    # Run the reference SwiGLU MoE expert forward in PyTorch. Mirrors
    # the diffusers source step-for-step.
    # ------------------------------------------------------------------
    capacity = int(CAPACITY)

    # Top-C per (B, E) along the S axis.
    top_w, top_idx = torch.topk(affinity, k=capacity, dim=-1)  # (B, E, C)

    # Flatten to expert-major (E, B, C).
    batch_offsets = torch.arange(B, device=DEVICE, dtype=torch.long).view(B, 1, 1) * S
    global_token_indices = (
        (batch_offsets + top_idx).transpose(0, 1).reshape(E, -1).reshape(-1)
    )
    gating_flat = top_w.transpose(0, 1).reshape(E, -1).reshape(-1).to(torch.float32)

    # Per-token renormalisation.
    total_tokens = B * S
    token_score_sums = torch.zeros(total_tokens, device=DEVICE, dtype=torch.float32)
    token_score_sums.scatter_add_(0, global_token_indices, gating_flat)
    gating_flat = gating_flat / (token_score_sums[global_token_indices] + 1e-12)
    gating_flat = gating_flat * ROUTE_SCALE

    # Permute tokens to expert-major.
    routed_input = x_flat[global_token_indices]  # (E*B*C, D) BF16

    # Use F.grouped_mm for the expert dispatch so the parity hits exactly
    # the same kernel semantics our Rust wrapper mirrors.
    tokens_per_expert = B * capacity
    offsets = torch.tensor(
        [tokens_per_expert * (k + 1) for k in range(E)],
        dtype=torch.int32,
        device=DEVICE,
    )

    gate_up = F.grouped_mm(routed_input, gate_up_w, offs=offsets)  # (E*B*C, 2*INTER) BF16
    gate, up = gate_up.chunk(2, dim=-1)
    act = F.silu(gate) * up                                        # (E*B*C, INTER) BF16
    routed_out = F.grouped_mm(act, down_w, offs=offsets)            # (E*B*C, D) BF16

    # Weighted scatter-add into an F32 accumulator, then cast to BF16.
    accum = torch.zeros(total_tokens, D, device=DEVICE, dtype=torch.float32)
    weighted = (routed_out.float() * gating_flat.unsqueeze(-1))
    scatter_idx = global_token_indices.view(-1, 1).expand(-1, D)
    accum.scatter_add_(0, scatter_idx, weighted)
    expected_y = accum.to(DTYPE)  # (B*S, D) BF16

    # ------------------------------------------------------------------
    # Save fixture. Metadata travels via 0-D tensors so the safetensors
    # loader (which only handles tensors) doesn't need a separate JSON.
    # ------------------------------------------------------------------
    out_dir = Path("tests/pytorch_fixtures/moe")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "nucleus_moe_parity.safetensors"

    fixture = {
        "x_flat": x_flat.contiguous().cpu(),
        "affinity": affinity.contiguous().cpu(),
        "gate_up_w": gate_up_w.contiguous().cpu(),
        "down_w": down_w.contiguous().cpu(),
        "expected_y": expected_y.contiguous().cpu(),
        # 1-D metadata tensors as F32 so flame-core's safetensors loader
        # (which filters non-{F32,BF16,F16,F8} dtypes) accepts them.
        # Saved as length-1 to dodge any 0-D shape edge cases on the load
        # path. Small positive integers round-trip exactly through f32.
        "meta_B": torch.tensor([B], dtype=torch.float32),
        "meta_E": torch.tensor([E], dtype=torch.float32),
        "meta_S": torch.tensor([S], dtype=torch.float32),
        "meta_D": torch.tensor([D], dtype=torch.float32),
        "meta_inter": torch.tensor([INTER], dtype=torch.float32),
        "meta_capacity": torch.tensor([capacity], dtype=torch.float32),
        "meta_route_scale": torch.tensor([ROUTE_SCALE], dtype=torch.float32),
    }
    save_file(fixture, str(out))

    # ------------------------------------------------------------------
    # Print sanity stats so failures are easier to diagnose.
    # ------------------------------------------------------------------
    print(f"saved {out}")
    print(f"  shapes: x_flat {tuple(x_flat.shape)} {x_flat.dtype}")
    print(f"          affinity {tuple(affinity.shape)} {affinity.dtype}")
    print(f"          gate_up_w {tuple(gate_up_w.shape)} {gate_up_w.dtype}")
    print(f"          down_w    {tuple(down_w.shape)} {down_w.dtype}")
    print(f"          expected_y {tuple(expected_y.shape)} {expected_y.dtype}")
    ey32 = expected_y.float()
    print(f"  expected_y stats: mean={ey32.mean().item():.5f} "
          f"std={ey32.std().item():.5f} "
          f"min={ey32.min().item():.5f} max={ey32.max().item():.5f}")


if __name__ == "__main__":
    main()
