# Stage S3 — speculative decoding (dev laptop)

N-gram draft + greedy target verify. Committed tok/s only (never proposed
throughput). Registered Qwen3 artifacts have no MTP heads.

```text
date:              2026-09-11
host:              MacBook-Pro (Apple Silicon)
OS:                Darwin 25.5.0 (arm64)
Zig:               0.16.0
commands:
  zig build test -Dhip=off
  zig build stageS3 -Dhip=off
  zig build spec-bench -Dhip=off
  ./zig-out/bin/zynfer spec-bench --mini --proposal-depth 4 --max-tokens 8 --backend cpu
  ZYNFER_QWEN_METAL=int8 ./zig-out/bin/zynfer spec-bench \
    models/qwen3-0.6b-int8.zynfer --proposal-depth 4 --max-tokens 16 --backend apple
```

## Correctness

| Check | Result |
| --- | --- |
| Unit: greedy speculative ≡ baseline (mini) | PASS |
| `spec-bench --mini` token_parity | PASS |
| Apple 0.6B int8 `spec-bench` token_parity | PASS |

## Mini CPU A/B

`proposal_depth=4`, `ngram_order=2`, `max_new=8`, backend=cpu

| Mode | wall_ms | gen | committed_t/s | acceptance | tok/round |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.385 | 8 | 20759 | — | — |
| speculative | 0.299 | 8 | 26719 | 0.50 | 1.60 |

## Apple Qwen3-0.6B int8 A/B

`ZYNFER_QWEN_METAL=int8`, backend=apple, `proposal_depth=4`, `max_new=16`

| Mode | wall_ms | gen | committed_t/s | acceptance | tok/round |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 1008.5 | 16 | 15.87 | — | — |
| speculative | **972.4** | 16 | **16.46** | **0.80** | 1.78 |

**Reading:** with a repetitive prompt that n-grams can draft, acceptance is
high and committed wall improves slightly. This is **not** a claim that
n-gram speculation always wins — open-ended prompts with low acceptance can
lose. MTP / a real draft model remains future work when artifacts support it.

## Policy

1. Propose ≤K tokens via history n-gram.
2. Verify with target greedy **before** `decodeToken`.
3. On mismatch: commit argmax, end round.
4. On full accept: +1 bonus from final logits.
5. Report committed tok/s only.

## Non-goals (this stage)

- Separate draft model / EAGLE / Medusa / layer-skip / MTP on CausalLM artifacts
- Tree / parallel multi-position verify
- HTTP (S4)

## Future goals (proposed)

See [`docs/proposals/speculative-draft-followons.md`](../../docs/proposals/speculative-draft-followons.md)
for optional **S3b** draft-quality campaign (after S4): priority draft LM →
MTP → Medusa → EAGLE → layer-skip, with committed-tok/s retain criteria.

## See also

- `docs/stages/S3-speculative-decoding.md`
- `docs/tutorials/25-speculative-decoding.md`
- `docs/proposals/speculative-draft-followons.md`
