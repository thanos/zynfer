# Proposal — Speculative draft follow-ons (post–S3)

**Status: proposed (not scheduled).** Stage S3 is **closed** on n-gram draft +
target verify. This document names stronger draft sources as **future goals**
only — after S4 (HTTP) unless a product bet explicitly pulls them forward.

Related: [`docs/stages/S3-speculative-decoding.md`](../stages/S3-speculative-decoding.md),
tutorial [`25-speculative-decoding.md`](../tutorials/25-speculative-decoding.md).

## Why not S3 goals

| Reason | Detail |
| --- | --- |
| Exit criterion met | Evaluate speculation vs baseline; n-gram teaches the class |
| Artifacts | Registered Qwen3-0.6B/4B are CausalLM-only (no MTP/Medusa/EAGLE tensors) |
| Scope | Separate draft model = second registry entry, load path, memory A/B |
| Discipline | Measure-then-retain; no half-paths without committed-tok/s ledgers |
| Curriculum | Phase S next milestone is **S4 HTTP**, not a draft-model bakeoff |

## Proposed future goals (priority order)

Suggested label if promoted: **S3b — Draft quality campaign** (optional; after
S4 or when serving tok/s becomes the product goal).

| Priority | Technique | Preconditions | Proposed gate |
| ---: | --- | --- | --- |
| 1 | **Separate draft model** | Register a small paired draft + target (same tokenizer family); `spec-bench` draft=artifact | Greedy (or documented sampling) parity; committed tok/s vs n-gram and vs baseline; memory report |
| 2 | **MTP heads** | Checkpoint with MTP (e.g. Qwen3.5/3.6-class) + converter + `Arch` fields | Head outputs verified; acceptance / tokens/round; no silent CausalLM fallback |
| 3 | **Medusa** | Trained Medusa heads in artifact (or documented train recipe out of runtime) | Multi-head propose + verify; parity; ledger vs n-gram |
| 4 | **EAGLE** | Extra autoregressive draft head on target features | Same metrics; retain only if acceptance×cost beats Medusa/draft-LM |
| 5 | **Layer-skip self-draft** | Early-exit / skip policy with Instruments/profile proof | Full vs skip quality bound; committed tok/s win at equal quality |
| 6 | **Tree / parallel verify** | Multi-position logits or tree attention in one forward | Documented encode/wait structure; no fake “parallel” on sequential decode |

## Technique cheat-sheet

| Technique | Draft source | Extra weights? |
| --- | --- | --- |
| N-gram (**S3 done**) | History statistics | No |
| Separate draft LM | Second small CausalLM | Yes (full draft model) |
| MTP | Multi-token heads on same checkpoint | Yes (heads) |
| Medusa | Parallel heads on last hidden | Yes (heads) |
| EAGLE | Light AR head on features | Yes (draft head) |
| Layer-skip | Same model, fewer layers | Policy only |

## Retain criteria (any follow-on)

All of the following, or the path is **REJECT** with a ledger (same spirit as M7):

1. **Correctness** — committed stream matches agreed sampling policy vs baseline (greedy parity at minimum).
2. **Committed tok/s** — beats n-gram S3 and/or plain `generate` on the same machine/prompt/length (noise bounds).
3. **Cost honesty** — draft FLOPs/bytes and peak RSS reported; no proposed-token throughput as the headline.
4. **Artifact honesty** — missing heads/models fail loud; no zombie half-support.
5. **Teachability** — tutorial update explaining why this draft source won or lost.

## Explicitly still out of scope here

- Turning S3 “done” back into “in progress”
- Shipping untrained Medusa/EAGLE stubs
- Fabricating speedups without `spec-bench` + ledger
- Blocking S4 on this proposal

## Recommended sequence

```text
S3 (done) → S4 HTTP → (optional) S3b draft campaign
                ↘ or skip S3b if serving quality > raw tok/s
```

Pick **one** draft source per campaign; change one variable; re-run
`spec-bench` matrix (mini + Apple 0.6B int8 + optional 4B).
