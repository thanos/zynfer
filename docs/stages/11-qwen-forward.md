# Stage 11 — Full Qwen3-0.6B forward + golden logits

**Status: not started.**

Stage 10 closed the checkpoint path: `.zynfer` validate/load, optional
local HF convert, mmap + `findById`. Stage 11 runs the model math.

## Goal

Run all 28 transformer layers on Qwen3-0.6B and produce logits that match
a trusted reference (CPU oracle or external golden).

```text
token IDs (fixed test vector — not tokenizer yet)
   ↓
embedding
   ↓
28 × transformer block
   ↓
final RMSNorm
   ↓
LM head
   ↓
logits
```

First milestone:

> **We have built an LLM inference engine** (forward-only; text comes in Stage 12).

## In scope (Stage 11)

| Item | Notes |
| --- | --- |
| **Full Qwen forward** | Bind weights from loaded `.zynfer` via `findById` |
| **Golden logits** | Compare embeddings → block 0 → selected blocks → final hidden → logits |
| **CPU path first** | Scalar/oracle ops; Metal optional once CPU matches |
| **BF16 weights** | Dequant/load policy TBD; may promote to f32 for oracle parity |
| **Named tensor binding** | Map HF tensor names / ids to layer weights |
| **CLI / tests** | e.g. `forward-golden`, differential tests vs reference |
| **Stage report** | `bench/results/stage11-dev-laptop.md` when done |

Absorbs Apple Stage 6 deferred **golden logits / full forward** work.

## Explicitly not Stage 11 (→ Stage 12 or out of CI)

These were **explicitly not Stage 10** and remain outside Stage 11:

| Item | Owner | Notes |
| --- | --- | --- |
| **Tokenizer** | Stage 12 | Stage 11 uses fixed token-id test vectors |
| **Sampling / greedy decode loop** | Stage 12 | Stage 11 compares logits, not generated text |
| **Vocabulary TTFT / tok/s / ITL** | Stage 12 | Benchmark matrix stays N/A until real tokens |
| **HF download in CI** | Never | Weights stay local/gitignored; CI uses tiny fixtures or checked-in golden vectors only — **not** `hf download` in GitHub Actions |

## Prerequisites

- Stage 10 `.zynfer` loader (`Artifact.loadFile`, `findById`)
- Local artifact optional: `tools/checkpoint/README.md`
- `src/model/qwen3.zig` architecture constants

## Exit criterion

For a fixed prompt represented as token IDs, top logits (and optionally
greedy token ids) agree with the trusted reference within stated tolerances.

## Commands (planned)

```bash
# TBD when implemented:
# zig build stage11
# ./zig-out/bin/zynfer forward-golden --artifact models/qwen3-0.6b.zynfer ...
```

See also: `docs/roadmap.md` (Track A), `docs/apple-backend.md` deferred table.
