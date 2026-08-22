# Stage 11 — Qwen forward + golden logits (dev laptop)

```text
date:              2026-08-22
host:              MacBook-Pro (Apple Silicon)
Zig:               0.16.0
commands:
  zig build test -Dhip=off
  zig build stage11 -Dhip=off
  ./zig-out/bin/zynfer artifact-compile --mini --out zig-out/stage11-mini.zynfer
  ./zig-out/bin/zynfer forward-golden zig-out/stage11-mini.zynfer --tokens=2,3
```

## Exit criterion (CI)

Mini f32 fixture → CPU forward → deterministic last-token logits. **Passes.**

## Full Qwen3-0.6B (local only)

```bash
# Weights + artifact (one-time)
hf download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B
python3 tools/checkpoint/safetensors_to_zynfer.py \
  --config models/Qwen3-0.6B/config.json \
  --weights models/Qwen3-0.6B \
  --out models/qwen3-0.6b.zynfer

# Smoke (top logits only)
./zig-out/bin/zynfer forward-golden models/qwen3-0.6b.zynfer --tokens=151643,2,3

# Golden compare
pip install torch transformers
python3 tools/fixtures/gen_golden_logits.py --tokens=151643,2,3 --out ref_logits.f32
./zig-out/bin/zynfer forward-golden models/qwen3-0.6b.zynfer \
  --tokens=151643,2,3 --golden ref_logits.f32
# → golden OK (ref_logits.f32, vocab=151936)
```

Top logits (last token, 2026-08-22): id=400 (13.27), id=25046, id=2, id=4, …

Promotes BF16 weights to F32 at load (~3 GiB resident). Golden file: raw
little-endian f32 blob, `vocab_size × 4` bytes, not in repo.

Debug: `--dump DIR` on `forward-golden`; compare with
`tools/fixtures/ref_forward_numpy.py --compare-dir DIR`. See
`docs/stages/11-qwen-forward.md`.

## Notes

- QK-norm, GQA, SwiGLU, RoPE (theta=1e6 from artifact meta).
- Last-token logits: final RMSNorm uses `viewLastRow`, not prefix `viewAs`.
- Tokenizer / sampling / TTFT → Stage 12.
- HF download never runs in CI.
