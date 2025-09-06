FLAME Autograd Coverage — GPU Backprop Readiness

- Grad Storage: FP32-only enforced (gradient.rs)
- sum_to_shape: GPU keepdim axis reductions via `sum_dim_keepdim_kernel`
- Softmax: Forward shifted-exp on GPU (GPU max/exp/sum); backward analytic with FP32 reductions
- LayerNorm: Backward present (dγ, dβ, dx); stats in FP32
- GroupNorm: Backward present (dγ, dβ, dx); stats in FP32
- Elementwise: Add/Mul/Div/Scalar ops with broadcasting and GPU reductions for backprop
- Matmul/Linear: dA = dY @ Bᵀ, dB = Aᵀ @ dY wired
- Conv2d: NHWC adapter in place — public API uses NHWC x and [KH,KW,IC,OC] w; internally adapts to NCHW + [OC,IC,KH,KW]; dInput NHWC FP32, dWeight [KH,KW,IC,OC] FP32, dBias [OC] FP32

Image Ops (NHWC) — ✅
- GPU kernels via NVRTC for resize_bilinear_nhwc, center_crop_nhwc, normalize_nhwc
- Public API: `image_ops_nhwc::{resize_bilinear_nhwc, center_crop_nhwc, normalize_nhwc}`
VAE API (minimal) — ✅
- NHWC encode: [N,H,W,3] → [N,H/8,W/8,4] with scale 0.18215; decode inverse
- BF16 params, FP32 compute; uses existing NHWC conv adapter internally
- Public API: `vae::AutoencoderKL::{from_safetensors, encode, decode}`

**Conv2d (NHWC adapter) — ✅**
- Public contract: input NHWC, weights [KH,KW,IC,OC], output NHWC.
- Impl: NHWC↔NCHW + weight [KH,KW,IC,OC]↔[OC,IC,KH,KW] permutations on GPU, reuse NCHW conv kernels.
- Grads: FP32 end-to-end (sum_to_shape keepdim reductions).
- Tests: `conv2d_nhwc_smoke` (shape/finite), `conv2d_nhwc_parity` (forward+backward parity vs NCHW).

**Softmax (GPU) — ✅**
- Forward uses GPU max/exp/sum with keepdim reductions for numerical stability.
- Backward: analytic Jacobian–vector product; FP32 reductions.
- Tests: `softmax_gpu_stability` (rows sum≈1, grads finite).

Tiny Tests
- tests/grad_sanity.rs:
  - softmax_backward_gpu
  - layernorm_backward_gpu
  - groupnorm_backward_gpu
  - matmul_backward_gpu
  - conv2d_layout_asserts (legacy failure check)
  - conv2d_nhwc_smoke.rs: conv2d_nhwc_forward_backward_smoke (forward shape, backward grads present, parity vs NCHW path)

Notes
- No externs added/renamed; kernels compiled at runtime from existing module.
- Remaining CPU paths: some scalar math still CPU; priority is to extend GPU kernels if needed.
