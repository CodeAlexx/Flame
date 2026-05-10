"""
Dump PyTorch forward-pass intermediates as a flat .safetensors file
that `flame_core::parity::ParityHarness` can load directly.

This is the reference template — copy it, adapt the model loading and
the layer selection, and you have a parity dump in 30 lines.

Recipe:
1. Build the PyTorch model and load weights (BF16, on CUDA, eval mode).
2. Register a forward hook on each layer of interest. The hook stores
   that layer's output keyed by a stable name (the module's path).
3. Run a forward pass on a fixed input. Convert each captured tensor
   to a contiguous CPU view in a dtype the safetensors loader handles
   (BF16 or F32; pick to match what the Rust side compares against).
4. Save with `safetensors.torch.save_file`.

Naming convention: keep keys identical to the strings the Rust caller
passes to `harness.compare(name, &our_tensor)`. The convention used
across flame-core test fixtures is the PyTorch module path
(`transformer_blocks.0.attn.to_q.output`), but any stable string works.

Two important non-obvious gotchas:
- Hooks fire during the forward pass; the input/output tensors are
  views into autograd-tracked storage. Detach + clone before storing or
  you will hold the entire graph in memory and the safetensors writer
  will crash on non-contiguous strides.
- Match the dtype of the Rust comparison tensor. Saving BF16 from
  PyTorch and comparing against a Rust BF16 tensor preserves the BF16
  rounding behavior; saving F32 from PyTorch lets you measure how much
  precision the Rust BF16 path lost relative to the F32 ground truth.
  Pick deliberately.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from safetensors.torch import save_file


def register_capture_hooks(
    model: torch.nn.Module,
    layer_names: Iterable[str],
    suffix: str = "output",
) -> Tuple[Dict[str, torch.Tensor], List[torch.utils.hooks.RemovableHandle]]:
    """
    Attach a forward hook to every named submodule in `layer_names`.
    The hook stores `output.detach().contiguous().cpu()` into the
    returned dict under the key `f"{name}.{suffix}"`.

    Caller is responsible for `handle.remove()` on every returned
    handle once the dump is captured (or just let them drop with the
    enclosing scope).
    """
    captures: Dict[str, torch.Tensor] = {}
    handles: List[torch.utils.hooks.RemovableHandle] = []

    by_name = dict(model.named_modules())
    for name in layer_names:
        if name not in by_name:
            raise KeyError(f"layer '{name}' not in model.named_modules()")
        module = by_name[name]
        key = f"{name}.{suffix}" if suffix else name

        def make_hook(k: str):
            def hook(_mod, _inp, out):
                # `out` may be a tuple (e.g. attention returns (out, attn_weights));
                # adapt as needed. Default: assume a single tensor.
                if isinstance(out, tuple):
                    out = out[0]
                captures[k] = out.detach().contiguous().cpu()
            return hook

        handles.append(module.register_forward_hook(make_hook(key)))
    return captures, handles


def dump_forward(
    model: torch.nn.Module,
    inputs: Tuple,
    layer_names: Iterable[str],
    out_path: str | Path,
    *,
    save_dtype: torch.dtype | None = None,
    suffix: str = "output",
) -> Dict[str, torch.Tensor]:
    """
    Run `model(*inputs)` once with hooks attached, then write the
    captured tensors to `out_path` as a .safetensors file.

    `save_dtype`:
      - `None`   keep each tensor's native dtype (typical for BF16 dumps)
      - `torch.float32` cast to F32 (use when measuring BF16 loss vs ground truth)
      - `torch.bfloat16` force BF16 even if model ran in F32
    """
    model.eval()
    captures, handles = register_capture_hooks(model, layer_names, suffix=suffix)
    try:
        with torch.inference_mode():
            _ = model(*inputs)
    finally:
        for h in handles:
            h.remove()

    if save_dtype is not None:
        captures = {k: v.to(save_dtype) for k, v in captures.items()}

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(captures, str(out_path))
    print(f"[dump_pytorch_layers] wrote {len(captures)} tensors to {out_path}")
    return captures


# ---------------------------------------------------------------------------
# Example usage (commented; copy and adapt for your model):
#
# from diffusers import QwenImageTransformer2DModel
# device = "cuda"
# model = QwenImageTransformer2DModel.from_pretrained(
#     "Qwen/Qwen-Image-2512", torch_dtype=torch.bfloat16
# ).to(device).eval()
#
# layers = [
#     # The exact names you'll compare against on the Rust side.
#     "transformer_blocks.0.norm1",
#     "transformer_blocks.0.attn.to_q",
#     "transformer_blocks.0.attn.to_k",
#     "transformer_blocks.0.attn.to_v",
#     "transformer_blocks.0.attn",          # attention output
#     "transformer_blocks.0.attn.to_out.0",
#     "transformer_blocks.0",                # full block output
#     "transformer_blocks.1",
#     "norm_out",
#     "proj_out",
# ]
#
# # Fixed inputs — set seed before constructing them so they reproduce.
# torch.manual_seed(0)
# hidden = torch.randn(1, 256, 3072, dtype=torch.bfloat16, device=device)
# encoder = torch.randn(1, 512, 3584, dtype=torch.bfloat16, device=device)
# timestep = torch.tensor([500], device=device)
#
# dump_forward(
#     model,
#     inputs=(hidden, encoder, timestep),
#     layer_names=layers,
#     out_path="dumps/qwen_block0_block1_block59_bf16.safetensors",
#     save_dtype=torch.bfloat16,
# )
