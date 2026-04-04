# Flame Inference — Standalone Repo Extraction Plan

Feasibility study for extracting inference code from EriDiffusion into a standalone
pure-Rust inference repo (similar to stable-diffusion.cpp).

## Status: Investigated, not started

## Architecture

The inference code is ~95% untangled from training. flame-core (57K lines) is already
a standalone crate. The inference files in eridiffusion (95K lines total) have almost
zero imports from training modules.

### New repo structure

```
flame-inference/
├── Cargo.toml              # depends on flame-core (git or path dep)
├── src/
│   ├── lib.rs
│   ├── device.rs           # thin flame_core wrapper, copy as-is
│   ├── tokenizers.rs       # copy as-is
│   ├── weight_loader.rs    # re-export from loaders/
│   │
│   ├── loaders/            # full dir: safetensors, lazy loading, format converters
│   │
│   ├── models/             # inference models only
│   │   ├── klein_model.rs
│   │   ├── klein_vae.rs
│   │   ├── zimage_model.rs
│   │   ├── flux_vae.rs     # shared VAE (used by zimage)
│   │   ├── ltx2_model.rs
│   │   ├── ltx2_vae.rs
│   │   ├── gemma3_encoder.rs
│   │   └── qwen3_encoder.rs
│   │
│   ├── sampling/           # renamed from inference/
│   │   ├── klein_sampling.rs
│   │   ├── zimage_sampling.rs
│   │   └── ltx2_sampling.rs
│   │
│   ├── ops/                # rope.rs, qk_norm.rs, attention.rs (no streaming norms)
│   │
│   └── bin/
│       ├── klein_generate.rs
│       ├── zimage_generate.rs
│       └── ltx2_generate.rs
```

### Files to copy (~15 core files + bins)

Source: `/home/alex/EriDiffusion/eridiffusion/eridiffusion/src/`

**Shared utilities (copy as-is):**
- `device.rs`
- `tokenizers.rs`
- `weight_loader.rs`

**Loaders (copy entire dir):**
- `loaders/` — WeightLoader, PrefixedWeightLoader, lazy safetensors, format converters

**Models (cherry-pick):**
- `models/klein_model.rs`, `models/klein_vae.rs`
- `models/zimage_model.rs`
- `models/flux_vae.rs` (shared VAE)
- `models/ltx2_model.rs`, `models/ltx2_vae.rs`
- `models/gemma3_encoder.rs`, `models/qwen3_encoder.rs`

**Sampling (cherry-pick):**
- `inference/klein_sampling.rs`
- `inference/zimage_sampling.rs`
- `inference/ltx2_sampling.rs`

**Ops (cherry-pick):**
- `ops/rope.rs`, `ops/qk_norm.rs`, `ops/attention.rs`
- Skip: `ops/streaming_layer_norm.rs`, `ops/streaming_rms_norm*.rs` (training-only)

**Bins (copy as-is):**
- `bin/klein_generate.rs`
- `bin/zimage_generate.rs`
- `bin/ltx2_generate.rs`

### What stays in EriDiffusion (training-only)

- `config/` — YAML training configs
- `trainers/` — all training loops
- `optimizers/` — Adam, SGD, etc.
- `networks/` — LoRA, gradients
- `data/` — datasets, augmentation
- `memory/block_swapping.rs` — training memory tricks
- `ops/streaming_*` — training-specific norms
- `unified_inference.rs` — tangled with trainers/, skip it

### flame-core dependency options

1. **Git dep (recommended for release):** `flame-core = { git = "https://github.com/CodeAlexx/Flame" }`
2. **Path dep (for dev):** `flame-core = { path = "../EriDiffusion/flame-core" }`
3. **Copy into repo:** fully standalone but maintains two copies

Recommendation: path dep during dev, git dep for release.

### Working models

| Model | Binary | Status | Output |
|-------|--------|--------|--------|
| Klein 4B | `klein_generate` | Working | PNG images |
| Z-Image (NextDiT) | `zimage_generate` | Working | PNG images |
| LTX-2.3 | `ltx2_generate` | In progress | Video latents (safetensors) |

### Adding new architectures

Drop in 3 files per model:
1. `models/{model}_model.rs` — transformer/network definition
2. `sampling/{model}_sampling.rs` — scheduler + denoise loop
3. `bin/{model}_generate.rs` — CLI entry point

All bins follow the same pattern:
load cached text embeddings -> load model weights -> seeded noise -> euler denoise -> VAE decode -> save

### Effort estimate

~1-2 hours. Copy files, set up Cargo.toml, verify it compiles.
