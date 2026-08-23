# Stage 12 — Tokenizer and sampling

**Status: done (CPU).** Byte-level BPE + greedy/temperature/top-k/top-p sampling
and a prefill → decode generation loop that prints text and TTFT.

Stage 11 produced last-token logits from fixed token IDs. Stage 12 turns a
**text prompt** into tokens, runs autoregressive generation, and decodes text.

## Goal

```bash
./zig-out/bin/zynfer run models/qwen3-0.6b.zynfer \
  --prompt "Explain gravity simply." \
  --tokenizer models/Qwen3-0.6B \
  --max-tokens 64
```

produces coherent continuation text (Qwen3 chat wrap by default).

## In plain English

- **Tokenizer** — turns letters into numbers the model understands, and
  numbers back into letters (`vocab.json` + `merges.txt`, Qwen2 BPE).
- **Sampling** — picks the next token from logits. Greedy = always the top
  score; temperature / top-k / top-p add controlled randomness.
- **`zynfer run`** — encode prompt → prefill → sample → decode one token at
  a time (KV cache) → print text + timing (TTFT, decode tok/s).

## In scope

| Item | Notes |
| --- | --- |
| Qwen2/Qwen3 BPE | `src/model/tokenizer.zig` |
| Sampling | `src/runtime/sample.zig` |
| Prefill + decode generate | `Session.generate` in `qwen_forward.zig` |
| CLI | `zynfer run …`, `zynfer stage12` |
| Metrics | TTFT (ns→ms), prefill_ms, decode_tok_s |

## Explicitly not Stage 12

| Item | Owner |
| --- | --- |
| Metal Qwen generate | Later |
| Server / batching | Later stages |
| HF download in CI | Never |

## Prerequisites

- Stage 11 artifact: `models/qwen3-0.6b.zynfer`
- Tokenizer files: `models/Qwen3-0.6B/vocab.json` + `merges.txt`
- Zig 0.16

## Commands

```bash
zig build stage12 -Dhip=off
zig build test -Dhip=off

# Greedy (default temperature=0)
./zig-out/bin/zynfer run models/qwen3-0.6b.zynfer \
  --prompt "Explain gravity simply." \
  --tokenizer models/Qwen3-0.6B \
  --max-tokens 32

# Sampled
./zig-out/bin/zynfer run models/qwen3-0.6b.zynfer \
  --prompt "Explain gravity simply." \
  --temperature 0.6 --top-k 20 --top-p 0.95 --seed 1 \
  --max-tokens 32
```

Flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--prompt` | required | User text |
| `--tokenizer` | `models/Qwen3-0.6B` | HF dir with vocab + merges |
| `--max-tokens` | 64 | New tokens to generate |
| `--temperature` / `--temp` | 0 | 0 = greedy |
| `--top-k` | 0 | 0 = off |
| `--top-p` | 1.0 | 1 = off |
| `--seed` | 0 | RNG seed |
| `--raw` | off | Skip Qwen3 chat template |

Default chat wrap (non-thinking):

```text
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
<think>

</think>


```

## Exit criterion

- Unit tests: mini BPE round-trip; real Qwen encode of
  `"Explain gravity simply."` → `[840, 20772, 23249, 4936, 13]` when weights
  dir is present
- Sampling tests: greedy + seeded determinism
- `zig build stage12` prints the Stage 12 ledger
- Local `zynfer run` prints text and TTFT metrics

## Implementation map

| Area | Path |
| --- | --- |
| Tokenizer | `src/model/tokenizer.zig` |
| Sampling | `src/runtime/sample.zig` |
| Generate loop | `src/model/qwen_forward.zig` |
| CLI | `src/main.zig` (`run`, `stage12`) |

Tutorial: [`docs/tutorials/12-tokenizer-and-run.md`](../tutorials/12-tokenizer-and-run.md).
Bench: [`bench/results/stage12-dev-laptop.md`](../../bench/results/stage12-dev-laptop.md).
