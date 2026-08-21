# Stage 10 — checkpoint / artifact (dev laptop)

```text
date:              2026-08-20
host:              MacBook-Pro (Apple Silicon)
Zig:               0.16.0
commands:
  zig build test -Dhip=off
  zig build stage10
  ./zig-out/bin/zynfer artifact-compile --out stage10-fixture.zynfer
  ./zig-out/bin/zynfer inspect stage10-fixture.zynfer
```

## Exit criterion

Deterministic Zig fixture → `.zynfer` → validate + load tensors. Passes.

## Notes

- Format: `docs/artifact-format.md` (ZYNF v1, SHA-256).
- Fixture embeds Qwen3-0.6B **metadata** and two tiny f32 tensors — not
  full weights.
- Optional full checkpoint: see **`tools/checkpoint/README.md`**
  (`hf download` → converter → `zynfer inspect`). Converter is BF16-safe;
  `huggingface-cli` is deprecated.
- Python Safetensors converter is optional for local HF trees; CI does not
  download checkpoints.
- Next: Stage 11 full forward + golden logits.
