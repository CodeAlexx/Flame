# Vendored: NVIDIA cudnn_frontend

This directory contains a **direct copy** of NVIDIA's `cudnn_frontend`
header-only C++ library. Used by `flame-core/src/cuda/cudnn_sdpa.cpp` to
build cuDNN v9 Flash SDPA graphs without hand-rolling ~500 lines of raw
backend-descriptor code.

## Pinned version

| field | value |
|---|---|
| tag | **v1.22.1** |
| commit | `a91f0e04dcea10515f0f776fc5a89535e316a9c8` |
| upstream | https://github.com/NVIDIA/cudnn-frontend |
| license | BSD-3-Clause (see `LICENSE.txt`) |
| vendored on | 2026-04-22 |
| vendored by | flame-core session working on `PLAN_CUDNN_SDPA.md` |

## Why

- Header-only C++ — no build-time library dependency beyond what cuDNN
  already demands.
- NVIDIA-maintained — tracks cuDNN C API changes authoritatively.
- Graph construction is ~50 lines of code via `Graph::sdpa()` vs ~500 lines
  of raw `cudnnBackendDescriptor*` setup.

## What's here

Only `include/` was copied — ~29 top-level entries, ~3.4 MB, 100% headers.
Samples, docs, tests, Python bindings, CMake glue were **excluded** from the
vendor.

## How to re-vendor (e.g. for a cuDNN bump)

```bash
# Pick a new release tag from the upstream release page.
TAG=v1.23.0  # example
cd /tmp && rm -rf cudnn_frontend
git clone --depth 1 --branch "$TAG" https://github.com/NVIDIA/cudnn-frontend.git cudnn_frontend

cd /home/alex/EriDiffusion/flame-core/third_party/cudnn_frontend
rm -rf include
cp -r /tmp/cudnn_frontend/include .
cp /tmp/cudnn_frontend/LICENSE.txt .

# Record the new tag + commit at the top of this file, rebuild, re-run
# Phase C + E (parity) + G (end-to-end Klein) before committing.
```

## What not to do

- Do **not** edit files under `include/` directly. Any local patches would
  silently drift on re-vendor. If upstream has a bug, open a PR there and
  bump the tag here.
- Do **not** mix this with a cuDNN version older than the one the tag
  supports. cuDNN ≥ 9.0 is required for cudnn_frontend v1.x.
