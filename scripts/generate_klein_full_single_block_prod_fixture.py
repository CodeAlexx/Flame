#!/usr/bin/env python3
"""
End-to-end Klein single-block parity fixture at production shape.

Composes EVERYTHING in `klein-trainer/src/model.rs::single_block_forward`:
modulate_pre → linear1 (fused QKV+gate_up) → narrow → LoRA(QKV) →
split_qkv (reshape+permute) → head_rms_norm → apply_rope → SDPA → permute
→ reshape → swiglu(gate_up) → cat(attn,mlp) → linear2 → LoRA(out on
attn slice only) → gate_residual.

The three ingredient tests (modulate_pre, swiglu_act, gate_residual) all
pass cos_sim=1.0 individually at production shape with all 3 known
fixes applied. If THIS fixture also passes, the residual bisect chaos
in the live trainer is at the chain-of-blocks level (BlockOffloader
saved-tensor races, AutogradContext interactions across many blocks,
or similar trainer-loop infrastructure), NOT a single-block backward
bug. If THIS fails, the bug is in composition: most likely candidates
are Tensor::cat backward, the dual-narrow on the MLP gate_up split, or
the LoRA-via-Op::Add on a narrowed slice.

Fixture: ~330 MB (linear1 alone is 207 MB BF16). Untracked.

Usage:
    cd flame-core
    python3 scripts/generate_klein_full_single_block_prod_fixture.py
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import save_file

SEED = 42
DEVICE = "cuda"
DTYPE = torch.bfloat16

# Klein single-block production shape:
B = 1
N = 1536
D = 3072
H = 24
HD = D // H          # 128
INNER = D            # 3072
MLP = 12288
RANK = 16
ALPHA = 16.0
HALF = HD // 2

OUT_PATH = Path("tests/pytorch_fixtures/patterns") / "klein_ext_full_single_block_prod.safetensors"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def make(shape, seed, scale=0.05, dtype=DTYPE):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return torch.randn(shape, generator=g, device=DEVICE, dtype=dtype) * scale


def make_f32(shape, seed, scale=0.05):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return torch.randn(shape, generator=g, device=DEVICE, dtype=torch.float32) * scale


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
print(f"Building production-shape Klein single-block fixture "
      f"(B={B} N={N} D={D} H={H} HD={HD} MLP={MLP} R={RANK})...")

x = make((B, N, D), SEED).requires_grad_(True)

# Frozen base weights (BF16 in production after FP8 dequant).
linear1_w = make((3 * INNER + 2 * MLP, D), SEED + 1)        # [33792, 3072]
linear2_w = make((D, INNER + MLP), SEED + 2)                 # [3072, 15360]
q_norm_scale = make((HD,), SEED + 3, scale=1.0)              # init around 1.0
k_norm_scale = make((HD,), SEED + 4, scale=1.0)

# Modulation outputs (frozen in Klein — produced by linear projection of
# timestep embedding which has no LoRA).
shift = make((B, D), SEED + 5, scale=0.1)
scale_mod = make((B, D), SEED + 6, scale=0.1)
gate = make((B, D), SEED + 7, scale=0.1)

# Interleaved RoPE tables built from Klein's actual frequency schedule.
freqs = torch.arange(0.0, HD, 2.0, device=DEVICE, dtype=torch.float32) \
    .mul_(-torch.tensor(10000.0).log() / HD).exp_()
positions = torch.arange(0, N, device=DEVICE, dtype=torch.float32)
angles = positions[:, None] * freqs[None, :]                    # [N, HALF]
pe_cos = angles.cos().to(DTYPE).view(1, 1, N, HALF).contiguous()
pe_sin = angles.sin().to(DTYPE).view(1, 1, N, HALF).contiguous()

# LoRA params: F32 (matches AdamW path).
# QKV LoRA: in=D, out=3*INNER (acts on the QKV slice of linear1's output).
lora_qkv_a = make_f32((RANK, D), SEED + 10).requires_grad_(True)
lora_qkv_b = torch.zeros((3 * INNER, RANK), device=DEVICE,
                          dtype=torch.float32, requires_grad=True)
# OUT LoRA: in=D (acts on attn_out slice), out=D (linear2 output dim).
lora_out_a = make_f32((RANK, D), SEED + 11).requires_grad_(True)
lora_out_b = torch.zeros((D, RANK), device=DEVICE,
                          dtype=torch.float32, requires_grad=True)
# Random upstream gradient for non-trivial backward (matches the bisect
# which has random LoRA seed + random `go` flowing into the block).
go = make((B, N, D), SEED + 20)

# Klein training has B=0 at init (zeros above) ⇒ delta_qkv = 0 at first step
# ⇒ d/d(lora_b) drives most of the gradient signal ⇒ this is the EXACT case
# where the bisect chaos was observed. Sample one B perturbation so the
# fixture exercises non-degenerate LoRA backward (matches the bisect's
# `--b-seed-std` defaulting to nonzero; see klein_finite_diff_test).
with torch.no_grad():
    lora_qkv_b.copy_(make_f32((3 * INNER, RANK), SEED + 12, scale=0.01))
    lora_out_b.copy_(make_f32((D, RANK), SEED + 13, scale=0.01))


# ---------------------------------------------------------------------------
# Forward — exact mirror of single_block_forward in klein-trainer
# ---------------------------------------------------------------------------
def lora_delta(input_3d, lora_a_f32, lora_b_f32):
    """Mirrors LoRALinear::forward_delta: cast to BF16, transpose + matmul,
    scale by alpha/rank. lora_a is [r, in], lora_b is [out, r]."""
    a_bf = lora_a_f32.to(DTYPE)
    b_bf = lora_b_f32.to(DTYPE)
    leading = input_3d.shape[0] * input_3d.shape[1]
    in_features = input_3d.shape[-1]
    out_features = b_bf.shape[0]
    inp_2d = input_3d.reshape(leading, in_features)
    delta_2d = inp_2d @ a_bf.t() @ b_bf.t()
    delta = delta_2d.reshape(input_3d.shape[0], input_3d.shape[1], out_features)
    return delta * (ALPHA / RANK)


def head_rms_norm(t, scale_v):
    """Per-head RMSNorm along the head_dim. Matches flame-core's rms_norm
    which computes `t * rsqrt(mean(t^2) + eps) * scale` in F32 and casts back."""
    eps = 1e-6
    var = t.float().pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    return (t.float() * rstd).to(DTYPE) * scale_v


def apply_rope_interleaved(t, pe_cos, pe_sin):
    """Mirrors bf16_ops::rope_fused_bf16 exactly (Interleaved layout).
    t: [BH, N, HD] reshape; pairs (2d, 2d+1) along HD.
        y_even = x_even * cos - x_odd * sin
        y_odd  = x_even * sin + x_odd * cos
    """
    BH = t.shape[0] * t.shape[1]
    Nl = t.shape[2]
    HDl = t.shape[3]
    HALFl = HDl // 2
    flat = t.reshape(BH, Nl, HDl)
    pairs = flat.float().view(BH, Nl, HALFl, 2)
    xe = pairs[..., 0]
    xo = pairs[..., 1]
    c = pe_cos.float().view(Nl, HALFl)
    s = pe_sin.float().view(Nl, HALFl)
    ye = xe * c - xo * s
    yo = xe * s + xo * c
    out = torch.stack([ye, yo], dim=-1).view(BH, Nl, HDl).to(DTYPE)
    return out.view(t.shape[0], t.shape[1], Nl, HDl)


# 1. modulate_pre
normed = F.layer_norm(x, [D], None, None, 1e-6)
normed.retain_grad()
x_normed = normed * (1.0 + scale_mod.unsqueeze(1)) + shift.unsqueeze(1)
x_normed.retain_grad()

# 2. linear1 fused
qkv_mlp = F.linear(x_normed, linear1_w)         # bias=None to match Klein
qkv_mlp.retain_grad()
qkv_dim = 3 * INNER
qkv_base = qkv_mlp[..., :qkv_dim]
qkv_base.retain_grad()
gate_up = qkv_mlp[..., qkv_dim:]
gate_up.retain_grad()

# 3. LoRA on QKV slice (narrow → add via lora_delta)
delta_qkv = lora_delta(x_normed, lora_qkv_a, lora_qkv_b)
delta_qkv.retain_grad()
qkv = qkv_base + delta_qkv
qkv.retain_grad()

# 4. split_qkv — reshape+permute
def split_head(z):
    return z.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()

q = split_head(qkv[..., :INNER])
k = split_head(qkv[..., INNER:2 * INNER])
v = split_head(qkv[..., 2 * INNER:])

# 5. head_rms_norm
q = head_rms_norm(q, q_norm_scale)
k = head_rms_norm(k, k_norm_scale)

# 6. RoPE
q = apply_rope_interleaved(q, pe_cos, pe_sin)
k = apply_rope_interleaved(k, pe_cos, pe_sin)

# 7. SDPA (uses PyTorch's reference impl — internally cuDNN/efficient/math)
attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
attn_out = attn.permute(0, 2, 1, 3).contiguous().reshape(B, N, INNER)
attn_out.retain_grad()

# 8. swiglu on gate_up
gate_proj = gate_up[..., :MLP]
gate_proj.retain_grad()
up_proj = gate_up[..., MLP:]
up_proj.retain_grad()
silu_gate = F.silu(gate_proj)
silu_gate.retain_grad()
mlp_out = silu_gate * up_proj
mlp_out.retain_grad()

# 9. cat
fused = torch.cat([attn_out, mlp_out], dim=2)
fused.retain_grad()

# 10. linear2 + LoRA on attn_out slice only
out_block_base = F.linear(fused, linear2_w)
delta_out = lora_delta(attn_out, lora_out_a, lora_out_b)
out_block = out_block_base + delta_out
out_block.retain_grad()

# 11. gate_residual
out = x + out_block * gate.unsqueeze(1)

print(f"  forward: out.shape={tuple(out.shape)}  out.dtype={out.dtype}  "
      f"||out||={out.float().norm().item():.3e}")

# ---------------------------------------------------------------------------
# Backward via random upstream `go` (matches `out.backward(go)` semantics).
# ---------------------------------------------------------------------------
loss = (out * go).sum()
loss.backward()

print(f"  backward: ||dx||={x.grad.float().norm().item():.3e}  "
      f"||dlora_qkv_a||={lora_qkv_a.grad.float().norm().item():.3e}  "
      f"||dlora_qkv_b||={lora_qkv_b.grad.float().norm().item():.3e}")
print(f"            ||dlora_out_a||={lora_out_a.grad.float().norm().item():.3e}  "
      f"||dlora_out_b||={lora_out_b.grad.float().norm().item():.3e}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
tensors = {
    "x": x.detach().cpu().contiguous(),
    "linear1_w": linear1_w.detach().cpu().contiguous(),
    "linear2_w": linear2_w.detach().cpu().contiguous(),
    "q_norm_scale": q_norm_scale.detach().cpu().contiguous(),
    "k_norm_scale": k_norm_scale.detach().cpu().contiguous(),
    "shift": shift.detach().cpu().contiguous(),
    "scale": scale_mod.detach().cpu().contiguous(),
    "gate": gate.detach().cpu().contiguous(),
    "pe_cos": pe_cos.detach().cpu().contiguous(),
    "pe_sin": pe_sin.detach().cpu().contiguous(),
    "lora_qkv_a": lora_qkv_a.detach().cpu().contiguous(),
    "lora_qkv_b": lora_qkv_b.detach().cpu().contiguous(),
    "lora_out_a": lora_out_a.detach().cpu().contiguous(),
    "lora_out_b": lora_out_b.detach().cpu().contiguous(),
    "go": go.detach().cpu().contiguous(),
    "output": out.detach().cpu().contiguous(),
    "dx": x.grad.cpu().contiguous(),
    "dlora_qkv_a": lora_qkv_a.grad.cpu().contiguous(),
    "dlora_qkv_b": lora_qkv_b.grad.cpu().contiguous(),
    "dlora_out_a": lora_out_a.grad.cpu().contiguous(),
    "dlora_out_b": lora_out_b.grad.cpu().contiguous(),
    # Intermediate gradients for suspect bisecting (Bug #4 hunt).
    # The dx is corrupted (cos≈0.47); these probes split the chain so
    # the first wrong cos_sim names the op that introduces the error.
    "dnormed":    normed.grad.cpu().contiguous(),
    "dx_normed":  x_normed.grad.cpu().contiguous(),
    "dqkv_mlp":   qkv_mlp.grad.cpu().contiguous(),
    "dqkv_base":  qkv_base.grad.cpu().contiguous(),
    "dgate_up":   gate_up.grad.cpu().contiguous(),
    "ddelta_qkv": delta_qkv.grad.cpu().contiguous(),
    "dqkv":       qkv.grad.cpu().contiguous(),
    "dattn_out":  attn_out.grad.cpu().contiguous(),
    "dgate_proj": gate_proj.grad.cpu().contiguous(),
    "dup_proj":   up_proj.grad.cpu().contiguous(),
    "dsilu_gate": silu_gate.grad.cpu().contiguous(),
    "dmlp_out":   mlp_out.grad.cpu().contiguous(),
    "dfused":     fused.grad.cpu().contiguous(),
    "dout_block": out_block.grad.cpu().contiguous(),
}
save_file(tensors, str(OUT_PATH))
print(f"\nWrote {OUT_PATH} "
      f"({sum(t.numel() * t.element_size() for t in tensors.values()) / 1024 / 1024:.1f} MB).")
