# Flame Core Notes

## Strict BF16 Overlay

The `strict_bf16` feature only turns on runtime guards and telemetry. It does **not**
pull in CUDA, attention, or v4 autograd support. Toggle it freely alongside or
without `cuda`, `bf16_u16`, `autograd_v4`, `flash_attn`, or other optional stacks.
Use `cargo check --features strict_bf16` (optionally with `--no-default-features`)
to verify the overlay in isolation.
