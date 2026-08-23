# Stage 12 — tokenizer + sampling (dev laptop)

```text
date:              2026-08-22
host:              MacBook-Pro (Apple Silicon)
Zig:               0.16.0
commands:
  zig build test -Dhip=off
  zig build stage12 -Dhip=off
  ./zig-out/bin/zynfer run models/qwen3-0.6b.zynfer \
    --prompt "Explain gravity simply." \
    --tokenizer models/Qwen3-0.6B \
    --max-tokens 16
```

## Exit criterion (CI)

- Mini BPE encode/decode unit test
- Sampling greedy + seeded determinism
- `zynfer stage12` ledger
- Real Qwen encode golden (skipped if `models/Qwen3-0.6B` absent)

## Full model (local)

Greedy run (`--max-tokens 8`, 2026-08-22):

```text
Gravity is a force that attracts every object

prompt_tokens=17 generated_tokens=8
ttft_ms=40306.672 prefill_ms=40306.191 decode_tok_s=0.324
```

CPU decode is the Stage 12 correctness path, not a speed target.

## Notes

- Prefer `zynfer setup` then `zynfer chat "…"`.
- Chat wrap defaults to Qwen3 non-thinking (`--raw` for plain continuation).
- Streaming is on by default (`--no-stream` to buffer).
- Tokenizer resolves next to the artifact, then `models/Qwen3-0.6B`.
- Stop ids: `im_end`, `endoftext`, and configured eos.
- Integration optionally runs a short `chat` when local weights exist.
