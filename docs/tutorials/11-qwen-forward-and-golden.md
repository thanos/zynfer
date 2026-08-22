# Tutorial — Qwen forward and golden logits (Stage 11)

This tutorial walks through validating the CPU forward pass against PyTorch.
You need the Stage 10 artifact from
[Tutorial 10](10-checkpoint-and-artifact.md).

Full reference: [`docs/stages/11-qwen-forward.md`](../stages/11-qwen-forward.md).

## What you are proving

Zynfer runs a **prefill** over a fixed list of token ids, applies 28 Qwen3
blocks, normalizes the **last** token, and computes vocab logits. Stage 11
checks that those logits match Hugging Face within tolerance.

Tokenizer and sampling are Stage 12 — here you pass raw ids.

## In plain English

Stage 11 taught zynfer to **actually run the model math**, not just load weights:

1. Look up each token’s embedding
2. Run all **28 transformer layers**
3. Normalize the **last** token
4. Produce **logits** — a score for every vocabulary entry (~152k numbers)

That’s a **forward pass**: tokens in → prediction scores out. No chat text yet.

### What is `forward-golden`?

A **sanity check**: run the network on token IDs and either print the top scores
or compare them to a saved **golden** answer from PyTorch.

- **Forward** = one neural-network pass
- **Golden** = optional answer key (`ref_logits.f32`)

Like checking `2 + 2` on a calculator, or grading against an answer sheet
(`golden OK` vs `NumericalMismatch`).

### The two commands you’ll use

**Smoke test** — run and print top logits (no answer key):

```bash
./zig-out/bin/zynfer forward-golden models/qwen3-0.6b.zynfer --tokens=151643,2,3
```

| Piece | Meaning |
| --- | --- |
| `forward-golden` | Run the model and show (or verify) results |
| `models/qwen3-0.6b.zynfer` | Packaged Qwen weights |
| `--tokens=151643,2,3` | Input token IDs (not English words yet) |

**Golden compare** — same run, graded against PyTorch:

```bash
python3 tools/fixtures/gen_golden_logits.py --tokens=151643,2,3 --out ref_logits.f32
./zig-out/bin/zynfer forward-golden models/qwen3-0.6b.zynfer \
  --tokens=151643,2,3 --golden ref_logits.f32
```

`--golden` means “must match this reference file.” Use the **same tokens** in
both steps. Success: `golden OK`. Failure: usually a real forward bug.

## 1. CI path (no weights)

From the repo root:

```bash
zig build stage11 -Dhip=off
```

This compiles a 1-layer mini artifact, runs forward on tokens `2,3`, and checks
determinism. No `models/` directory required.

## 2. Build the real artifact (local)

If you have not already:

```bash
hf download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B

python3 tools/checkpoint/safetensors_to_zynfer.py \
  --config models/Qwen3-0.6B/config.json \
  --weights models/Qwen3-0.6B \
  --out models/qwen3-0.6b.zynfer

./zig-out/bin/zynfer inspect models/qwen3-0.6b.zynfer
```

Expect `model_id: qwen3-0.6b`, 28 layers, 311 BF16 tensors, `storage: mmap`.

## 3. Smoke test

```bash
zig build -Dhip=off

./zig-out/bin/zynfer forward-golden models/qwen3-0.6b.zynfer --tokens=151643,2,3
```

Default tokens (if you omit `--tokens`) are the same: BOS `151643`, then `2`,
then `3`.

You should see eight lines of top logits. On a correct build the top id is
**400** (~13.27), not an unrelated id with negative logit at 400.

## 4. Golden compare (Option B)

Install Python deps once:

```bash
pip install torch transformers
```

Generate reference logits (last prompt token):

```bash
python3 tools/fixtures/gen_golden_logits.py \
  --model models/Qwen3-0.6B \
  --tokens=151643,2,3 \
  --out ref_logits.f32
```

The script prints top logits to stderr and writes 607 744 bytes (`151936 × 4`).

Compare:

```bash
./zig-out/bin/zynfer forward-golden models/qwen3-0.6b.zynfer \
  --tokens=151643,2,3 \
  --golden ref_logits.f32
```

Success:

```text
golden OK (ref_logits.f32, vocab=151936)
```

**Rule:** use the same `--tokens` in both commands.

## 5. If golden compare fails

### Wrong file or tokens

- `FileNotFound` → run `gen_golden_logits.py` first
- Size mismatch → golden must be exactly `vocab_size` f32 values
- Top id wrong but small drift → regenerate golden; script uses eager attention

### Real forward bug — bisect with dumps

Dump zynfer tensors:

```bash
./zig-out/bin/zynfer forward-golden models/qwen3-0.6b.zynfer \
  --tokens=151643,2,3 \
  --dump=zynfer_dump
```

Replay from safetensors without torch:

```bash
pip install numpy

python3 tools/fixtures/ref_forward_numpy.py \
  --tokens=151643,2,3 \
  --out-dir ref_numpy \
  --compare-dir zynfer_dump
```

Read the printed `max_abs` per stage:

| Stage | Meaning |
| --- | --- |
| `embed_last` | Embedding gather |
| `layer00` … `layer27` | After each transformer block |
| `normed` | After `model.norm` on **last token** |
| `logits` | Tied LM head |

If all layers match but `normed` does not, suspect final norm row selection
(last vs first token). Zynfer uses `Tensor.viewLastRow()`.

More detail: [`tools/fixtures/README.md`](../../tools/fixtures/README.md).

## 6. What's next

| Item | Stage |
| --- | --- |
| Tokenizer / sampling / TTFT | **12** |
| Metal Qwen forward | After CPU golden holds |
| HF download in CI | **Never** |

## Commands cheat sheet

```bash
zig build test -Dhip=off
zig build stage11 -Dhip=off
./zig-out/bin/zynfer forward-golden ARTIFACT [--tokens IDS] [--golden PATH] [--dump DIR]
python3 tools/fixtures/gen_golden_logits.py --tokens=151643,2,3 --out ref_logits.f32
python3 tools/fixtures/ref_forward_numpy.py --compare-dir zynfer_dump
zynfer stage11   # in-repo ledger
```
