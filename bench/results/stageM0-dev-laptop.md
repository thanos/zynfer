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

## Open gate items (do not drop)

Tracked also in [`docs/stages/M0-metal-qwen-forward.md`](../../docs/stages/M0-metal-qwen-forward.md):

| # | Item | Status |
| --- | --- | --- |
| 1 | Full-model Metal greedy == CPU | OPEN |
| 2 | Metal TTFT / decode tok/s in this ledger | OPEN — use `qwen-bench` on full model |
| 3 | Per-layer dump ladder on Metal | OPEN |
| 4 | LM-head GEMV path A/B | OPEN |
| 5 | Attention parity at kv_len > 256 | OPEN |
| 6 | Resident KV / one-CB | → M3 |

## Notes

- Attention: `kv_len ≤ 256` thread-local; `257…2048` device scores (`attention_f32_buf`).
- Context > 2048 on Metal: unsupported until M3.
- Planning: `baoulo/prompts/fable-5-prompt.md`, `docs/roadmap.md` Phase M/R/S.
