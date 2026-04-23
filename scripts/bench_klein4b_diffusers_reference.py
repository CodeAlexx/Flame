#!/usr/bin/env python3
"""PyTorch Klein 4B DiT baseline on the same GPU flame-core runs on.

Purpose: this is the reference number `klein_infer` (flame-core Klein 4B) is
being compared against. Run on the same 3090 Ti, same 1024×1024, same 50
steps, same guidance=4.0, same BF16, same CFG-doubled batch — the only
difference is the framework.

Measured 2026-04-23 on RTX 3090 Ti (Ampere SM_86):
    diffusers Flux2Transformer2DModel, Klein 4B weights:  1055 ms/step
    flame-core KleinTransformer,        Klein 4B weights:  1180 ms/step
    Ratio: 1.12× (flame-core 12% slower than PyTorch reference)

That's the real gap. It replaces earlier sessions' extrapolations from
proxy-shape pattern benchmarks (which read 1.11× composition — basically
the same number, confirming the extrapolation, but the extrapolation was
an extrapolation). Future sessions comparing flame-core Klein perf should
re-run this script to stay honest.

Requires: diffusers >= 0.38.0.dev0 (Flux2Transformer2DModel landed there).
Weights: /home/alex/.serenity/models/checkpoints/flux-2-klein-base-4b.safetensors
         (native BFL checkpoint; config is pulled from HF repo_id).

Out of scope (same as flame-core's "Stage 3 denoise" number): VAE decode,
text encoder forward, scheduler math. We want the DiT forward, not the
pipeline.
"""
import time
import torch
from diffusers import Flux2Transformer2DModel

DEVICE = "cuda"
DTYPE = torch.bfloat16
CKPT = "/home/alex/.serenity/models/checkpoints/flux-2-klein-base-4b.safetensors"

# Klein 4B config (inferred from checkpoint key shapes)
CONFIG = dict(
    patch_size=1,
    in_channels=128,
    num_layers=5,           # double blocks
    num_single_layers=20,   # single blocks
    attention_head_dim=128,
    num_attention_heads=24,  # 3072 / 128
    joint_attention_dim=7680,
    timestep_guidance_channels=256,
    mlp_ratio=3.0,
    axes_dims_rope=(32, 32, 32, 32),
    rope_theta=2000,
    eps=1e-06,
    guidance_embeds=True,
)

# Matches flame-core klein_infer settings
WIDTH, HEIGHT = 1024, 1024
NUM_STEPS = 50
GUIDANCE = 4.0
BATCH = 2  # CFG: cond + uncond doubled batch

# Klein derives n_img from latent shape: H/16 × W/16 = 64 × 64 = 4096 tokens
N_IMG = (HEIGHT // 16) * (WIDTH // 16)
# Text token count — Klein uses Qwen3-4B with typical ~512 tokens; flame-core
# caches embeddings at this size. Match for apples-to-apples.
N_TXT = 512


def main():
    print(f"Loading Klein 4B transformer from {CKPT} ...")
    t0 = time.time()
    # `from_single_file` wants a repo_id for config (pulls config.json
    # from HF — ~2 KB). The actual weights load from our local file.
    model = Flux2Transformer2DModel.from_single_file(
        CKPT,
        config="black-forest-labs/FLUX.2-klein-4B",
        subfolder="transformer",
        torch_dtype=DTYPE,
    ).to(DEVICE)
    model.eval()
    print(f"  loaded in {time.time() - t0:.1f}s")

    # Build synthetic inputs at the production shape.
    hidden_states = torch.randn(
        BATCH, N_IMG, CONFIG["in_channels"], device=DEVICE, dtype=DTYPE
    )
    encoder_hidden_states = torch.randn(
        BATCH, N_TXT, CONFIG["joint_attention_dim"], device=DEVICE, dtype=DTYPE
    )
    # Position IDs: img uses 2D positions (+ time/channel axes if RoPE 4D),
    # txt uses 1D positions. Klein's RoPE is axes (32,32,32,32) → rank 4,
    # meaning img_ids is [B, N_img, 4] and txt_ids is [B, N_txt, 4].
    img_ids = torch.zeros(BATCH, N_IMG, 4, device=DEVICE, dtype=torch.float32)
    txt_ids = torch.zeros(BATCH, N_TXT, 4, device=DEVICE, dtype=torch.float32)
    timestep = torch.tensor([500.0, 500.0], device=DEVICE, dtype=DTYPE)
    guidance = torch.tensor([GUIDANCE, GUIDANCE], device=DEVICE, dtype=DTYPE)

    print(f"\nShape: hidden_states {list(hidden_states.shape)}, "
          f"encoder_hidden_states {list(encoder_hidden_states.shape)}")
    print(f"Config: {NUM_STEPS} steps, guidance={GUIDANCE}, "
          f"CFG batch={BATCH} (= 2 fwd passes/step in CFG-doubled mode)")

    # Warmup: 3 forwards to clear lazy init / cuBLAS heuristics / memory pool
    print("\nWarmup (3 forwards)...")
    with torch.no_grad():
        for _ in range(3):
            _ = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,
                return_dict=False,
            )
    torch.cuda.synchronize()

    # Time NUM_STEPS forward passes. Each "step" is ONE forward; flame-core's
    # klein_infer at CFG runs the CFG-doubled batch (B=2) as a single forward,
    # so step time = one model() call. Matches flame-core's "2.25 s/step".
    print(f"\nTiming {NUM_STEPS} forwards at B={BATCH} (CFG-doubled) ...")
    per_step = []
    torch.cuda.synchronize()
    t_all = time.time()
    with torch.no_grad():
        for i in range(NUM_STEPS):
            torch.cuda.synchronize()
            t_step = time.time()
            _ = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,
                return_dict=False,
            )
            torch.cuda.synchronize()
            per_step.append(time.time() - t_step)
    total = time.time() - t_all

    per_step.sort()
    median = per_step[len(per_step) // 2]
    mean = sum(per_step) / len(per_step)
    pmin, pmax = per_step[0], per_step[-1]
    print(f"\n--- Results ---")
    print(f"Total wall ({NUM_STEPS} steps): {total:.2f} s")
    print(f"Per-step:")
    print(f"  mean:   {mean*1000:.1f} ms")
    print(f"  median: {median*1000:.1f} ms")
    print(f"  min:    {pmin*1000:.1f} ms")
    print(f"  max:    {pmax*1000:.1f} ms")
    print(f"\nCompare to flame-core klein_infer (Klein 9B, not 4B): 2280 ms/step")
    print(f"                                        (Klein 4B flame-core): [run klein_infer]")


if __name__ == "__main__":
    main()
