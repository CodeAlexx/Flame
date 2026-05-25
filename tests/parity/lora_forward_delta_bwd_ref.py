#!/usr/bin/env python3
"""Parity reference for LoRALinear::forward_delta backward.

Replicates the EXACT forward_delta implementation in
EriDiffusion-v2/crates/eridiffusion-core/src/lora.rs:

    pub fn forward_delta(&self, input: &Tensor) -> Result<Tensor> {
        // 1. Flatten leading dims  [B, N, Cin] -> [B*N, Cin]
        // 2. Cast lora_a (F32) -> BF16, then transpose  -> [Cin, rank]
        // 3. Cast lora_b (F32) -> BF16, then transpose  -> [rank, Cout]
        // 4. intermediate = input_2d @ a_t    [B*N, rank]
        // 5. out_2d = intermediate @ b_t      [B*N, Cout]
        // 6. scaled = out_2d * (alpha / rank)
        // 7. reshape back                     [B, N, Cout]

This is NOT a simplified matmul test — it follows the actual param dtypes and
cast order. lora_a and lora_b are F32 leaves; they get cast to BF16 inside the
forward before the matmul. The backward must flow through:
    Op::Cast (F32->BF16) for both A and B
    Op::Transpose
    Op::MatMul (BF16)

This is the code path the handoff (§2, HANDOFF_2026-05-24_SHARED_BACKWARD_BUG_LOCALIZED.md)
identified as "Remaining candidate #2": the actual LoRALinear::forward_delta uses
F32 -> BF16 cast followed by two matmuls, which was NOT tested by the isolated
checkpoint/matmul tests in Experiments A-C.

Cases cover:
  - The real Z-Image QKV shapes (rank=16, in=1536, out=4608 for q+k+v)
  - The real Z-Image FFN shapes (rank=16, in=1536, out=6144)
  - A smaller sanity case (rank=8, in=64, out=128)
  - 3D input (B, N, Cin) — the actual call site passes 3D tensors from ensure_3d

For each case:
  input: BF16 (matches what the trainer passes — already-casted activations)
  lora_a: F32, requires_grad=True  (matches Parameter storage)
  lora_b: F32, requires_grad=True  (matches Parameter storage)
  base_w: BF16, requires_grad=False (frozen model weight, added outside forward_delta)
  loss: MSE(out + base_out, target)   <- real usage: forward_delta result is added
                                          to the frozen base weight output

Outputs grad_lora_a and grad_lora_b for Rust to match.
"""

import json
import os
import sys
import torch
import torch.nn.functional as F


def forward_delta_ref(input_bf16, lora_a_f32, lora_b_f32, alpha: float, rank: int):
    """Exact Python replica of LoRALinear::forward_delta."""
    orig_shape = input_bf16.shape   # e.g. [B, N, Cin] or [B*N, Cin]
    orig_ndim = len(orig_shape)

    # Flatten all leading dims -> [leading, Cin]
    leading = 1
    for d in orig_shape[:-1]:
        leading *= d
    in_features = orig_shape[-1]
    out_features = lora_b_f32.shape[0]

    input_2d = input_bf16.reshape(leading, in_features)

    # Cast F32 params to BF16, transpose
    a_bf16 = lora_a_f32.to(torch.bfloat16)   # [rank, in_features]
    b_bf16 = lora_b_f32.to(torch.bfloat16)   # [out_features, rank]
    a_t = a_bf16.t()                           # [in_features, rank]
    b_t = b_bf16.t()                           # [rank, out_features]

    intermediate = input_2d @ a_t             # [leading, rank]
    out_2d = intermediate @ b_t               # [leading, out_features]

    scale = alpha / rank
    if abs(scale - 1.0) > 1e-9:
        scaled = out_2d * scale
    else:
        scaled = out_2d

    # Reshape back: last dim = out_features, leading dims preserved
    out_shape = list(orig_shape[:-1]) + [out_features]
    return scaled.reshape(out_shape)


def make_case(
    batch, seq, in_features, out_features, rank, alpha,
    seed, tag
):
    """Generate one parity case and return a dict."""
    torch.manual_seed(seed)
    device = torch.device("cuda")

    # BF16 activation input (3D: [B, N, Cin])
    input_bf16 = torch.randn(
        batch, seq, in_features, device=device, dtype=torch.bfloat16
    ) * 0.5

    # F32 LoRA params — requires_grad=True (matching lora.rs Parameter storage)
    bound = 1.0 / (in_features ** 0.5)
    lora_a_f32 = (
        (torch.rand(rank, in_features, device=device, dtype=torch.float32) * 2.0 - 1.0) * bound
    ).requires_grad_(True)
    # lora_b starts zero (Kaiming zero init in new() — but for parity, use small random
    # to get non-zero grad_lora_b from step 1 even without B-zero-init saturation)
    lora_b_f32 = (
        torch.randn(out_features, rank, device=device, dtype=torch.float32) * 0.01
    ).requires_grad_(True)

    # Frozen base weight (BF16, requires_grad=False) — simulates the model's W_q etc.
    base_w_bf16 = torch.randn(
        out_features, in_features, device=device, dtype=torch.bfloat16
    ) * (1.0 / in_features ** 0.5)

    # BF16 target (what the block's output should match — MSE loss)
    target_bf16 = torch.randn(
        batch, seq, out_features, device=device, dtype=torch.bfloat16
    ) * 0.1

    # ─── Forward ───────────────────────────────────────────────────────
    # Base path: input @ base_w^T  (frozen)
    input_2d = input_bf16.reshape(batch * seq, in_features)
    base_out_2d = input_2d @ base_w_bf16.t()
    base_out = base_out_2d.reshape(batch, seq, out_features)

    # LoRA delta
    lora_out = forward_delta_ref(input_bf16, lora_a_f32, lora_b_f32, alpha, rank)

    # Combined output
    out = base_out + lora_out   # [B, N, Cout], BF16

    # MSE loss in F32 to match trainer
    loss = F.mse_loss(out.float(), target_bf16.float())
    loss.backward()

    assert lora_a_f32.grad is not None, f"{tag}: grad_lora_a is None!"
    assert lora_b_f32.grad is not None, f"{tag}: grad_lora_b is None!"

    grad_a = lora_a_f32.grad.detach().float().cpu().tolist()
    grad_b = lora_b_f32.grad.detach().float().cpu().tolist()

    print(
        f"  {tag}: loss={loss.item():.6e}  "
        f"|grad_a|={lora_a_f32.grad.norm().item():.4e}  "
        f"|grad_b|={lora_b_f32.grad.norm().item():.4e}"
    )

    return {
        "tag": tag,
        "batch": batch,
        "seq": seq,
        "in_features": in_features,
        "out_features": out_features,
        "rank": rank,
        "alpha": alpha,
        "seed": seed,
        # Tensors as F32 flat lists (Rust loads, casts to BF16/F32 as appropriate)
        # All tensors flattened to 1D for simple Vec<f32> deserialization in Rust.
        "input_bf16_as_f32": input_bf16.detach().float().cpu().flatten().tolist(),
        "lora_a_f32": lora_a_f32.detach().cpu().flatten().tolist(),
        "lora_b_f32": lora_b_f32.detach().cpu().flatten().tolist(),
        "base_w_bf16_as_f32": base_w_bf16.detach().float().cpu().flatten().tolist(),
        "target_bf16_as_f32": target_bf16.detach().float().cpu().flatten().tolist(),
        # Expected gradients (flat)
        "grad_lora_a_f32": lora_a_f32.grad.detach().float().cpu().flatten().tolist(),
        "grad_lora_b_f32": lora_b_f32.grad.detach().float().cpu().flatten().tolist(),
    }


def main() -> int:
    fixture_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "lora_forward_delta_bwd_ref.json",
    )

    print("Generating lora_forward_delta_bwd_ref.json ...")
    print("  (Tests the EXACT forward_delta code path from lora.rs: F32->BF16 cast + 2 matmuls)")

    cases = []

    # Case 0: Small sanity case (fast, catches basic grad path)
    cases.append(make_case(
        batch=4, seq=8, in_features=64, out_features=128,
        rank=8, alpha=1.0, seed=0xDEAD0001, tag="small(4,8,64->128,r8)"
    ))

    # Case 1: Real Z-Image QKV shape — to_q/to_k/to_v combined projection
    # Z-Image hidden_dim=1536, head_dim=64, num_heads=24 -> out=1536 per branch
    # But all-in-one QKV is 1536 * 3 = 4608 output
    cases.append(make_case(
        batch=2, seq=256, in_features=1536, out_features=4608,
        rank=16, alpha=1.0, seed=0xDEAD0002, tag="zimage_qkv(2,256,1536->4608,r16)"
    ))

    # Case 2: Real Z-Image FFN shape (up-projection: in=1536, out=6144)
    cases.append(make_case(
        batch=2, seq=256, in_features=1536, out_features=6144,
        rank=16, alpha=1.0, seed=0xDEAD0003, tag="zimage_ffn(2,256,1536->6144,r16)"
    ))

    # Case 3: Z-Image to_out projection (out=1536, in=1536)
    cases.append(make_case(
        batch=2, seq=256, in_features=1536, out_features=1536,
        rank=16, alpha=1.0, seed=0xDEAD0004, tag="zimage_to_out(2,256,1536->1536,r16)"
    ))

    fixture = {"cases": cases}
    with open(fixture_path, "w") as f:
        json.dump(fixture, f)

    print(f"\nwrote {fixture_path}")
    print(f"  {len(cases)} cases covering: small sanity + real Z-Image QKV/FFN/to_out shapes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
