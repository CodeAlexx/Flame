# `parity_helpers` — shared parity comparator for flame-core tests

## Why this exists

Every fused-kernel phase in the Fusion Sprint adds a CUDA kernel and
needs a numerical gate. Without a shared comparator, each phase reinvents
its own atol/rtol policy and "FAILED" prints with no diagnostics.

This module gives every test:

1. **Consistent tolerance defaults** — BF16 element-wise vs FP32-reduction.
2. **Top-K delta print on mismatch** — the index, the actual, the expected,
   the absolute delta. So when a stride bug corrupts one corner of an
   output, you can see WHERE without spelunking.
3. **SHA256 fixture pinning** — catches silent regeneration / corruption
   of the `.safetensors` fixture.

## Usage

```rust
mod parity_helpers;

#[test]
fn my_kernel_parity() {
    let dev = flame_core::global_cuda_device();
    let path = std::path::Path::new("tests/pytorch_fixtures/.../my_op.safetensors");

    // 1. Pin the fixture so silent regeneration is caught.
    assert_eq!(
        parity_helpers::sha256_file(path),
        "abc123...", // bytes-of-fixture SHA256
    );

    // 2. Load.
    let tensors = flame_core::serialization::load_file(path, &dev).unwrap();
    let input = tensors.get("input").unwrap();
    let expected = tensors.get("output").unwrap();

    // 3. Run flame-core's op.
    let got = my_kernel(input).unwrap();

    // 4. Compare.
    parity_helpers::assert_parity_bf16("my_op", &got, expected);
}
```

## Tolerance defaults

| Path | atol | rtol | When to use |
|-----:|:----:|:----:|-------------|
| `compare_bf16` | 1e-2 | 1e-2 | unary/binary ops, fused norms, fused softmax, attention output, MLP epilogue, dropout (with mask check) |
| `compare_fp32_reduction` | 1e-5 | 1e-5 | norm stats `(mean, rstd)`, optimizer moments, grad clip global L2 norm, loss reductions |

If you need something tighter or looser, use `compare_tensor(got, expected, atol, rtol)` directly and **document why** in the test body.

## What it deliberately does NOT do

- It does not compare flame-core against itself. PyTorch is the oracle. If
  you find yourself reaching for "compare to my old impl", stop and add a
  PyTorch fixture instead.
- It does not generate fixtures. That's `scripts/generate_*_fixture.py`.
  Fixtures are committed under `tests/pytorch_fixtures/` and pinned by SHA.
- It does not report timing. Use `cargo bench` / per-phase `benches/*.rs`
  for that.

## Generating new fixtures

```bash
# Smoke fixture (already committed):
python3 -c '
import torch
from safetensors.torch import save_file
torch.manual_seed(0); torch.cuda.manual_seed(0)
a = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
b = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
save_file({"input_a": a.cpu().contiguous(),
           "input_b": b.cpu().contiguous(),
           "output": (a+b).cpu().contiguous()},
          "tests/pytorch_fixtures/smoke/add_4x8_bf16.safetensors")'

# Verify pinned hash still matches:
sha256sum tests/pytorch_fixtures/smoke/add_4x8_bf16.safetensors
# expected: ddde4d62922baf2969a0eba1b8db813580b1dbc9f37fcc66ef9e8110dc1a5086
```

For larger op fixtures, prefer `scripts/generate_pytorch_fixtures.py`
which writes the canonical layout used by `pytorch_parity.rs`.
