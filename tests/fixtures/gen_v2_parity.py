"""
Phase 5a — autograd v2 parity fixture generator.

Generates PyTorch reference fixtures for the 13 v2 ops to feed
`tests/autograd_v2_parity.rs`. Per op we emit:

  - `<op>_backward.safetensors`: inputs, grad_output, expected per-input grads.
  - `<op>_jvp.safetensors`: inputs, per-input tangents, expected output tangent.

Conventions:
  - Fixed seed 42 (numpy + torch).
  - Inputs named `input_0`, `input_1`, ... (in op argument order).
  - Backward fixtures: `grad_output` (= upstream grad, all-ones), plus
    `grad_input_0`, `grad_input_1`, ... matching the input order.
  - JVP fixtures: `tangent_0`, `tangent_1`, ... plus `out_fw`
    (= the expected output tangent).
  - Dtype is `bfloat16` for layer_norm (matches flame-core's BF16-only LN
    op); all other ops use `float32` (matches `tests/autograd_v2_fw_mode.rs`
    and `tests/autograd_v2_ops.rs`).
  - Shapes are kept small (≤512 elements) so each fixture stays a few KB
    on disk.

Reference for layer_norm JVP formula (PyTorch ATen):
  aten/src/ATen/native/layer_norm.cpp::layer_norm_backward_cpu
  (the same chain rule pieces, applied in the forward direction).

Run: `python3 tests/fixtures/gen_v2_parity.py` from `flame-core/` root.
"""

import os
import sys

import torch
from safetensors.torch import save_file

torch.manual_seed(42)

OUTDIR = os.path.dirname(os.path.abspath(__file__))


def _save(path: str, tensors: dict) -> None:
    # safetensors requires contiguous CPU tensors with concrete dtype.
    # `.clone()` because storage-sharing siblings (e.g. grad_output and
    # grad_input_* both equal to ones-tensors that share storage) trip the
    # safetensors "shared memory" error.
    out = {k: v.detach().clone().contiguous().cpu() for k, v in tensors.items()}
    save_file(out, path)
    print(f"  wrote {os.path.basename(path)}: {list(out.keys())}")


def _backward_fixture(op_name, inputs, output, dtype=torch.float32):
    """Run torch.autograd.grad with all-ones grad_output, save fixture."""
    grad_output = torch.ones_like(output)
    grads = torch.autograd.grad(
        outputs=output,
        inputs=inputs,
        grad_outputs=grad_output,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )
    tensors = {}
    for i, x in enumerate(inputs):
        tensors[f"input_{i}"] = x.detach().to(dtype)
    tensors["grad_output"] = grad_output.detach().to(dtype)
    for i, g in enumerate(grads):
        tensors[f"grad_input_{i}"] = g.detach().to(dtype)
    _save(os.path.join(OUTDIR, f"{op_name}_backward.safetensors"), tensors)


def _jvp_fixture(op_name, fn, primals, tangents, dtype=torch.float32):
    """Run torch.autograd.functional.jvp, save fixture."""
    _, out_fw = torch.autograd.functional.jvp(fn, primals, tangents)
    tensors = {}
    for i, x in enumerate(primals):
        tensors[f"input_{i}"] = x.detach().to(dtype)
    for i, t in enumerate(tangents):
        tensors[f"tangent_{i}"] = t.detach().to(dtype)
    tensors["out_fw"] = out_fw.detach().to(dtype)
    _save(os.path.join(OUTDIR, f"{op_name}_jvp.safetensors"), tensors)


# ---------------------------------------------------------------------------
# 1. add
# ---------------------------------------------------------------------------
def gen_add():
    print("[add]")
    shape = (8, 16)
    a = torch.randn(shape, dtype=torch.float32, requires_grad=True)
    b = torch.randn(shape, dtype=torch.float32, requires_grad=True)
    out = a + b
    _backward_fixture("add", [a, b], out)
    # JVP
    a2 = torch.randn(shape, dtype=torch.float32)
    b2 = torch.randn(shape, dtype=torch.float32)
    ta = torch.randn(shape, dtype=torch.float32)
    tb = torch.randn(shape, dtype=torch.float32)
    _jvp_fixture("add", lambda x, y: x + y, (a2, b2), (ta, tb))


# ---------------------------------------------------------------------------
# 2. mul
# ---------------------------------------------------------------------------
def gen_mul():
    print("[mul]")
    shape = (8, 16)
    a = torch.randn(shape, dtype=torch.float32, requires_grad=True)
    b = torch.randn(shape, dtype=torch.float32, requires_grad=True)
    out = a * b
    _backward_fixture("mul", [a, b], out)
    a2 = torch.randn(shape, dtype=torch.float32)
    b2 = torch.randn(shape, dtype=torch.float32)
    ta = torch.randn(shape, dtype=torch.float32)
    tb = torch.randn(shape, dtype=torch.float32)
    _jvp_fixture("mul", lambda x, y: x * y, (a2, b2), (ta, tb))


# ---------------------------------------------------------------------------
# 3. sum
# ---------------------------------------------------------------------------
def gen_sum():
    print("[sum]")
    shape = (4, 8)
    a = torch.randn(shape, dtype=torch.float32, requires_grad=True)
    out = a.sum()  # scalar output
    _backward_fixture("sum", [a], out)
    a2 = torch.randn(shape, dtype=torch.float32)
    ta = torch.randn(shape, dtype=torch.float32)
    _jvp_fixture("sum", lambda x: x.sum(), (a2,), (ta,))


# ---------------------------------------------------------------------------
# 4. matmul
# ---------------------------------------------------------------------------
def gen_matmul():
    print("[matmul]")
    a = torch.randn(4, 8, dtype=torch.float32, requires_grad=True)
    b = torch.randn(8, 6, dtype=torch.float32, requires_grad=True)
    out = a @ b
    _backward_fixture("matmul", [a, b], out)
    a2 = torch.randn(4, 8, dtype=torch.float32)
    b2 = torch.randn(8, 6, dtype=torch.float32)
    ta = torch.randn(4, 8, dtype=torch.float32)
    tb = torch.randn(8, 6, dtype=torch.float32)
    _jvp_fixture("matmul", lambda x, y: x @ y, (a2, b2), (ta, tb))


# ---------------------------------------------------------------------------
# 5. silu
# ---------------------------------------------------------------------------
def gen_silu():
    print("[silu]")
    shape = (4, 16)
    a = torch.randn(shape, dtype=torch.float32, requires_grad=True)
    out = torch.nn.functional.silu(a)
    _backward_fixture("silu", [a], out)
    a2 = torch.randn(shape, dtype=torch.float32)
    ta = torch.randn(shape, dtype=torch.float32)
    _jvp_fixture("silu", torch.nn.functional.silu, (a2,), (ta,))


# ---------------------------------------------------------------------------
# 6. layer_norm  (BF16-only in flame-core; affine = (weight, bias))
# ---------------------------------------------------------------------------
def gen_layer_norm():
    print("[layer_norm]")
    # Pick a 2D shape so the kernel's outer-x-norm layout is obvious.
    # normalized_shape = [features]; per-row reduction.
    rows, feats = 4, 16
    eps = 1e-5

    # Backward: BF16 storage; run the ref computation in float32 then
    # cast inputs/outputs to bfloat16 for storage. The op itself is BF16-only
    # in flame-core; allowing the reference to be F32 makes the BF16 noise
    # bound interpretable (we know what the noiseless answer is).
    a = torch.randn(rows, feats, dtype=torch.float32, requires_grad=True)
    w = torch.randn(feats, dtype=torch.float32, requires_grad=True)
    b = torch.randn(feats, dtype=torch.float32, requires_grad=True)
    out = torch.nn.functional.layer_norm(a, [feats], w, b, eps)
    _backward_fixture("layer_norm", [a, w, b], out, dtype=torch.bfloat16)

    # JVP — also save BF16.
    a2 = torch.randn(rows, feats, dtype=torch.float32)
    w2 = torch.randn(feats, dtype=torch.float32)
    b2 = torch.randn(feats, dtype=torch.float32)
    ta = torch.randn(rows, feats, dtype=torch.float32)
    tw = torch.randn(feats, dtype=torch.float32)
    tb = torch.randn(feats, dtype=torch.float32)

    def f(x, ww, bb):
        return torch.nn.functional.layer_norm(x, [feats], ww, bb, eps)

    _jvp_fixture("layer_norm", f, (a2, w2, b2), (ta, tw, tb), dtype=torch.bfloat16)


# ---------------------------------------------------------------------------
# 7. reshape  /  8. view
# ---------------------------------------------------------------------------
def gen_reshape():
    print("[reshape]")
    a = torch.randn(6, 8, dtype=torch.float32, requires_grad=True)
    out = a.reshape(2, 3, 8)
    _backward_fixture("reshape", [a], out)
    a2 = torch.randn(6, 8, dtype=torch.float32)
    ta = torch.randn(6, 8, dtype=torch.float32)
    _jvp_fixture("reshape", lambda x: x.reshape(2, 3, 8), (a2,), (ta,))


def gen_view():
    print("[view]")
    a = torch.randn(4, 6, dtype=torch.float32, requires_grad=True)
    out = a.contiguous().view(8, 3)
    _backward_fixture("view", [a], out)
    a2 = torch.randn(4, 6, dtype=torch.float32)
    ta = torch.randn(4, 6, dtype=torch.float32)
    _jvp_fixture("view", lambda x: x.contiguous().view(8, 3), (a2,), (ta,))


# ---------------------------------------------------------------------------
# 9. transpose  (2D — flame-core's transpose_v2 takes no args; swaps last 2)
# ---------------------------------------------------------------------------
def gen_transpose():
    print("[transpose]")
    a = torch.randn(4, 6, dtype=torch.float32, requires_grad=True)
    out = a.transpose(0, 1)
    _backward_fixture("transpose", [a], out)
    a2 = torch.randn(4, 6, dtype=torch.float32)
    ta = torch.randn(4, 6, dtype=torch.float32)
    _jvp_fixture("transpose", lambda x: x.transpose(0, 1), (a2,), (ta,))


# ---------------------------------------------------------------------------
# 10. narrow
# ---------------------------------------------------------------------------
def gen_narrow():
    print("[narrow]")
    a = torch.randn(8, 10, dtype=torch.float32, requires_grad=True)
    # narrow along dim=1, start=2, length=5 → output [8, 5]
    out = a.narrow(1, 2, 5)
    _backward_fixture("narrow", [a], out)
    a2 = torch.randn(8, 10, dtype=torch.float32)
    ta = torch.randn(8, 10, dtype=torch.float32)
    _jvp_fixture("narrow", lambda x: x.narrow(1, 2, 5), (a2,), (ta,))


# ---------------------------------------------------------------------------
# 11. squeeze  /  12. unsqueeze
# ---------------------------------------------------------------------------
def gen_squeeze():
    print("[squeeze]")
    a = torch.randn(4, 1, 6, dtype=torch.float32, requires_grad=True)
    out = a.squeeze(1)
    _backward_fixture("squeeze", [a], out)
    a2 = torch.randn(4, 1, 6, dtype=torch.float32)
    ta = torch.randn(4, 1, 6, dtype=torch.float32)
    _jvp_fixture("squeeze", lambda x: x.squeeze(1), (a2,), (ta,))


def gen_unsqueeze():
    print("[unsqueeze]")
    a = torch.randn(4, 6, dtype=torch.float32, requires_grad=True)
    out = a.unsqueeze(1)
    _backward_fixture("unsqueeze", [a], out)
    a2 = torch.randn(4, 6, dtype=torch.float32)
    ta = torch.randn(4, 6, dtype=torch.float32)
    _jvp_fixture("unsqueeze", lambda x: x.unsqueeze(1), (a2,), (ta,))


# ---------------------------------------------------------------------------
# 13. permute
# ---------------------------------------------------------------------------
def gen_permute():
    print("[permute]")
    a = torch.randn(2, 3, 4, dtype=torch.float32, requires_grad=True)
    out = a.permute(2, 0, 1)
    _backward_fixture("permute", [a], out)
    a2 = torch.randn(2, 3, 4, dtype=torch.float32)
    ta = torch.randn(2, 3, 4, dtype=torch.float32)
    _jvp_fixture("permute", lambda x: x.permute(2, 0, 1), (a2,), (ta,))


def main():
    print(f"writing fixtures to {OUTDIR}")
    gen_add()
    gen_mul()
    gen_sum()
    gen_matmul()
    gen_silu()
    gen_layer_norm()
    gen_reshape()
    gen_view()
    gen_transpose()
    gen_narrow()
    gen_squeeze()
    gen_unsqueeze()
    gen_permute()
    print("done.")


if __name__ == "__main__":
    main()
