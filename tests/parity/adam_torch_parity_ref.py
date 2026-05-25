#!/usr/bin/env python3
"""Generate the PyTorch-AdamW reference fixture for `flame_core::adam::AdamW`.

The handoff `HANDOFF_2026-05-24_ZIMAGE_VAE_FIXED_ADAM_AMP_OPEN.md` §10 names
this as the single-step decisive test: feed identical p_0 + 5 grads to both
optimizers with identical hyperparameters, dump p after each step, diff.

Output: `flame-core/tests/parity/adam_torch_parity_ref.json` with:
    config:        the hyperparameters used (lr, betas, eps, wd)
    init:          p_0 as Vec<f32>, length N
    grads:         [g_1, ..., g_5], each length N
    p_after_step:  [p_1, ..., p_5], each length N (after torch.optim.AdamW step k)

Reference is `torch.optim.AdamW` from PyTorch (CUDA F32 dtype; CPU and CUDA
agree on this op so CUDA is the strict source). The reference is implemented
INDEPENDENTLY of our kernel — torch.optim.AdamW is the canonical AdamW.

Hyperparameters mirror the Z-Image / Klein LoRA configuration from the
handoff: lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01. Grad
magnitude is calibrated to the observed step-1 LoRA-B grad scale (~1e-4 L2
over ~16 elements => per-element magnitude ~2.5e-5).

Run once; output checked in for the Rust test to read.

Usage:
    python3 tests/parity/adam_torch_parity_ref.py
"""

import json
import os
import sys

import torch


def _run_case(name, lr, beta1, beta2, eps, wd, n_per_tensor, n_tensors, steps, seed):
    """Run one PyTorch AdamW reference scenario and return the JSON dict."""
    device = torch.device("cuda")
    torch.manual_seed(seed)

    inits = [
        torch.randn(n_per_tensor, device=device, dtype=torch.float32) * 0.01
        for _ in range(n_tensors)
    ]
    grads = [
        [
            torch.randn(n_per_tensor, device=device, dtype=torch.float32) * 3e-5
            for _ in range(n_tensors)
        ]
        for _ in range(steps)
    ]

    params = [x.clone().detach().requires_grad_(True) for x in inits]
    opt = torch.optim.AdamW(
        params, lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=wd
    )

    p_after = []
    for step_grads in grads:
        for p, g in zip(params, step_grads):
            p.grad = g.clone()
        opt.step()
        p_after.append([p.detach().cpu().tolist() for p in params])

    return {
        "name": name,
        "config": {
            "lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "eps": eps,
            "weight_decay": wd,
            "n_per_tensor": n_per_tensor,
            "n_tensors": n_tensors,
            "steps": steps,
        },
        "inits": [x.detach().cpu().tolist() for x in inits],
        # grads[step][tensor] -> list[float]
        "grads": [[g.detach().cpu().tolist() for g in step_grads] for step_grads in grads],
        # p_after_step[step][tensor] -> list[float]
        "p_after_step": p_after,
    }


def main() -> int:
    fixture_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "adam_torch_parity_ref.json",
    )

    cases = [
        # (name, lr, b1, b2, eps, wd, n_per, n_tensors, steps, seed)
        # 1) Zimage / Klein LoRA preset (single param, short horizon)
        _run_case("zimage_klein_single", 3e-4, 0.9, 0.999, 1e-8, 0.01, 16, 1, 5, 0xADA7C0DE),
        # 2) Many tensors, exercises the multi-tensor fused path (n>1 params).
        _run_case("multi_tensor_50t", 3e-4, 0.9, 0.999, 1e-8, 0.01, 64, 50, 10, 0xC0FFEE01),
        # 3) WD=0 (isolates the WD-order question — should be identical regardless of order).
        _run_case("zero_wd", 3e-4, 0.9, 0.999, 1e-8, 0.0, 64, 8, 20, 0xBEEF0001),
        # 4) Long horizon (50 steps) at zimage preset — any compounding bug shows up here.
        _run_case("zimage_long_horizon", 3e-4, 0.9, 0.999, 1e-8, 0.01, 64, 8, 50, 0xFEEDFACE),
    ]

    with open(fixture_path, "w") as f:
        json.dump({"cases": cases}, f, indent=2)

    print(f"wrote {fixture_path}")
    for c in cases:
        cfg = c["config"]
        print(
            f"  {c['name']}: n_per={cfg['n_per_tensor']} n_tensors={cfg['n_tensors']} "
            f"steps={cfg['steps']} lr={cfg['lr']} wd={cfg['weight_decay']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
