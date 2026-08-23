# Stage M0 — Metal Qwen forward (dev laptop)

```text
date:              2026-08-23
host:              MacBook-Pro (Apple Silicon)
Zig:               0.16.0
commands:
  zig build test -Dhip=off
  zig build stageM0 -Dhip=off
  ./zig-out/bin/zynfer forward-golden zig-out/stage11-mini.zynfer \
    --tokens 2,3 --backend apple
```

## Exit criterion (CI)

- Unit test: mini Metal logits vs CPU (`Stage M0: mini Metal forward matches CPU logits`)
- `zynfer stageM0` ledger
- `--backend apple` on `forward-golden` (mini fixture)

## Mini fixture parity

`forward-golden --tokens 2,3 --backend apple` matches CPU top logits within test tolerance (3e-3).

Full Qwen3-0.6B Metal generate: local only; embed/LM head on CPU; blocks per-op Metal.
Expect unimpressive tok/s until M3 batched schedule.

## Notes

- Attention: `kv_len ≤ 256` thread-local; `257…2048` device scores (`attention_f32_buf`).
- Context > 2048 on Metal: unsupported until M3.
- Planning: `baoulo/prompts/fable-5-prompt.md`, `docs/roadmap.md` Phase M/R/S.
