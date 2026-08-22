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

## Optional full Qwen (local only; not CI)

```text
hf download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B
python3 tools/checkpoint/safetensors_to_zynfer.py \
  --config models/Qwen3-0.6B/config.json \
  --weights models/Qwen3-0.6B \
  --out models/qwen3-0.6b.zynfer
./zig-out/bin/zynfer inspect models/qwen3-0.6b.zynfer
```

| Metric | Value |
| --- | --- |
| Artifact bytes | 1 503 302 336 |
| Tensors | 311 (BF16) |
| Load storage | mmap |
| sha256 | `f466df0f393bf4d0858d0116d9fbac3276e5bcbbb4c6eba996219b329811a244` |

## Notes

- Format: `docs/artifact-format.md` (ZYNF v1, SHA-256).
- Fixture embeds Qwen3-0.6B **metadata** and two tiny f32 tensors — not
  full weights.
- Converter: `tools/checkpoint/README.md` (BF16-safe, single file or shards).
- Load path: mmap with heap fallback; hot path `findById`.
- **Closed.** Forward / golden logits → Stage 11; tokenizer / TTFT → Stage 12.
