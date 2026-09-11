# Tutorial — Speculative decoding (Stage S3)

Ordinary autoregressive decoding commits **one** token per expensive target
forward. Speculative decoding tries to commit **more than one** by proposing
cheap draft tokens and verifying them with the target model.

## In plain English

1. **Draft** — guess the next few tokens cheaply (here: n-gram from history).
2. **Verify** — ask the real model’s greedy next token; accept only matches.
3. **Reject** — on the first mismatch, commit what the model actually wanted
   and start a new round.
4. **Bonus** — if every draft matched, sample one more token from the last
   logits (the classic +1).

You measure **committed** tokens/sec — never raw proposed throughput.

## Why not MTP on Qwen3-0.6B/4B?

Those registered artifacts are `Qwen3ForCausalLM` with a single LM head. They
do **not** ship multi-token prediction heads. Stage 23 allows MTP *if* the
checkpoint exposes it; until then, n-gram (or another cheap draft) is the
honest educational path.

## Acceptance rate and speedup

```text
acceptance_rate = accepted_draft_tokens / proposed_tokens
tokens_per_round = committed_tokens / rounds
```

Expected speedup needs both a **cheap** draft and a **high** acceptance rate.
N-gram draft is nearly free, but acceptance can be low on open-ended text —
speculation can **lose** (extra bookkeeping, same number of target forwards
per committed token when every draft rejects). That outcome is still a valid
Stage S3 result: evaluate rigorously against the baseline.

## Greedy parity

With temperature 0, every accepted draft equals the target argmax, and every
reject commits that argmax. The committed stream must match ordinary
`generate` — `spec-bench` asserts `token_parity: PASS`.

## Try it

```bash
zig build stageS3 -Dhip=off
./zig-out/bin/zynfer spec-bench --mini --proposal-depth 4 --max-tokens 8
```

## What is not here (S3)

- Separate draft LM, Medusa, EAGLE, layer-skip, MTP heads — see **future goals**
- HTTP serving — Stage S4
- Packed continuous batch Metal forwards

## Future goals (proposed, not scheduled)

S3 teaches the **class** of speculative decoding. Stronger drafts become
optional goals only when artifacts and product need justify them:

| Technique | Draft source | When it becomes a goal |
| --- | --- | --- |
| Separate draft model | Second small LM | Serving tok/s; register paired draft+target |
| MTP | Multi-token heads in checkpoint | Register an MTP-capable artifact + converter |
| Medusa | Extra parallel heads on last hidden | Heads present/trained; ledger vs n-gram |
| EAGLE | Light AR head on features | Same retain bar as Medusa |
| Layer-skip | Same model, fewer layers | Profiled skip policy; quality-bounded win |

Full proposal, priority order, and retain criteria:
[`docs/proposals/speculative-draft-followons.md`](../proposals/speculative-draft-followons.md).

Do **not** reopen S3 for these; prefer an optional **S3b** after S4.

## See also

- [`docs/stages/S3-speculative-decoding.md`](../stages/S3-speculative-decoding.md)
- [`docs/proposals/speculative-draft-followons.md`](../proposals/speculative-draft-followons.md)
- [`bench/results/stageS3-dev-laptop.md`](../../bench/results/stageS3-dev-laptop.md)
