# Stage S3 — Speculative decoding

**Status: done (n-gram draft + target verify).** Cheap history n-gram proposals
verified greedily against the target model before each `decodeToken`.
Registered Qwen3-0.6B/4B have **no MTP heads**; MTP is deferred until a
compatible checkpoint is registered.

Part of **Phase S** ([`fable-5-prompt.md`](../../baoulo/prompts/fable-5-prompt.md)).
Old curriculum Stage 23.

## Goal

```bash
zig build stageS3 -Dhip=off
./zig-out/bin/zynfer stageS3
./zig-out/bin/zynfer spec-bench --mini --proposal-depth 4 --max-tokens 8
ZYNFER_QWEN_METAL=int8 ./zig-out/bin/zynfer spec-bench \
  models/qwen3-0.6b-int8.zynfer --proposal-depth 4 --max-tokens 16 --backend apple
```

## What landed

| Piece | Policy |
| --- | --- |
| Draft | N-gram continuation from committed history (order ≥ 2) |
| Verify | Greedy argmax must match draft before decode |
| Reject | Commit target argmax; end round (no bad KV write) |
| Full accept | +1 bonus token from final logits (classic speculative +1) |
| Metrics | proposal depth, acceptance rate, tokens/round, **committed** tok/s |
| Parity | Greedy stream ≡ non-speculative `generate` |

## Explicit non-goals (this stage)

- Separate draft model / EAGLE / Medusa / layer-skip — **future proposal**, not S3
- MTP heads (absent from registered CausalLM artifacts) — same proposal
- Tree attention / multi-position parallel verify — same proposal
- HTTP (S4) — **done**; see `docs/stages/S4-http-server.md`
- Reporting proposed-token throughput as useful tok/s

## Future goals (proposed)

Stronger draft sources are documented as an optional post–S3 / post–S4
campaign — **not** required to keep S3 closed:

→ [`docs/proposals/speculative-draft-followons.md`](../proposals/speculative-draft-followons.md)

Summary priority: (1) separate draft LM, (2) MTP-capable checkpoint, (3)
Medusa, (4) EAGLE, (5) layer-skip, (6) tree verify — each with committed-tok/s
retain criteria and loud failure if artifacts lack heads.

## Gate

1. Greedy token parity vs baseline (mini unit + `spec-bench`)
2. `spec-bench` JSON: depth, acceptance, tokens/round, committed tok/s
3. Tutorial teaches draft/verify/accept and when speculation loses
4. Non-goals documented above; future draft sources linked as proposal only

## Commands

```bash
zig build test -Dhip=off
zig build stageS3 -Dhip=off
zig build spec-bench -Dhip=off
zig build integration -Dhip=off
```

## Files

| Path | Role |
| --- | --- |
| `src/runtime/speculative.zig` | n-gram draft, verify loop, `runSpecBench` |
| `src/main.zig` | `stageS3`, `spec-bench` |
| `docs/tutorials/25-speculative-decoding.md` | Walkthrough |
| `docs/proposals/speculative-draft-followons.md` | Future draft-source goals |
| `bench/results/stageS3-dev-laptop.md` | Ledger |
