# Stage 11 — Full Qwen3-0.6B forward + golden logits

**Status: done (CPU oracle).** Mini CI fixture passes; full local artifact matches
PyTorch golden logits.

Stage 10 closed the checkpoint path: `.zynfer` validate/load, optional local HF
convert, mmap + `findById`. Stage 11 runs the model math and compares logits to
a trusted reference.

## Goal

Run all 28 transformer layers on Qwen3-0.6B and produce **last-token logits**
that match a trusted reference (PyTorch/transformers or numpy replay from
safetensors).

```text
token IDs (fixed test vector — not tokenizer yet)
   ↓
embedding
   ↓
28 × transformer block (QK-norm, GQA, RoPE, SwiGLU)
   ↓
final RMSNorm  (last token row only)
   ↓
LM head (tied to embed)
   ↓
logits for the last prompt token
```

First milestone:

> **We have built an LLM inference engine** (forward-only; text comes in Stage 12).

## In plain English

Stage 11 taught zynfer to **actually run Qwen3-0.6B** — not download weights or
inspect files, but do the math a language model does when you give it numbers
(token IDs):

1. Look up each token’s embedding vector
2. Pass through all **28 transformer layers** (attention, etc.)
3. Normalize the result for the **last** token
4. Produce **logits** — a score for every word/token in the vocabulary (~152k numbers)

That’s a **forward pass**: input tokens in → prediction scores out. No text
generation yet (that’s Stage 12).

### What is `forward-golden`?

It’s a **sanity-check command**: “Run the model on these tokens and show me
(or verify) the output.”

- **Forward** = run the neural network once
- **Golden** = an optional **reference answer** saved ahead of time (from
  PyTorch), used to check zynfer isn’t wrong

Think of it like running a calculator on `2 + 2` and printing `4`, or comparing
your answer to an answer key (`golden OK` vs `NumericalMismatch`).

### Smoke test (print top scores)

```bash
./zig-out/bin/zynfer forward-golden models/qwen3-0.6b.zynfer --tokens=151643,2,3
```

| Piece | Meaning |
| --- | --- |
| `./zig-out/bin/zynfer` | The zynfer program you built |
| `forward-golden` | “Run the model and show results” |
| `models/qwen3-0.6b.zynfer` | The packaged Qwen weights (your “model file”) |
| `--tokens=151643,2,3` | Feed in those token IDs (numbered symbols, not English yet) |

Loads the model, runs those tokens through all layers, and prints the **top
scoring token IDs** for “what comes next after the last token” — without checking
against a reference. Different tokens → different logits (expected).

Default when `--tokens` is omitted: BOS `151643`, then `2`, then `3`.

### Golden compare (check the answer key)

```bash
python3 tools/fixtures/gen_golden_logits.py --tokens=151643,2,3 --out ref_logits.f32

./zig-out/bin/zynfer forward-golden models/qwen3-0.6b.zynfer \
  --tokens=151643,2,3 --golden ref_logits.f32
```

| Piece | Meaning |
| --- | --- |
| Same as above | Same model run |
| `--golden ref_logits.f32` | “Compare my output to this reference file” |
| `ref_logits.f32` | ~152k floats from PyTorch — the “correct” scores for the last token |

1. Runs the forward pass
2. Reads `ref_logits.f32`
3. Checks every score is close enough to the reference

Success prints `golden OK`. Failure prints `NumericalMismatch` (usually a real
bug, not “close enough”). **Use the same `--tokens` in both commands.**

One-sentence summary: **`forward-golden`** runs Qwen on a short list of token
numbers and either **shows** the top predictions or **proves** they match
PyTorch. That’s how we know the Zig engine computes the same thing as the
standard reference before tokenizer and text generation in Stage 12.

## In scope (Stage 11)

| Item | Notes |
| --- | --- |
| **Full Qwen forward** | Bind weights from loaded `.zynfer` |
| **Golden logits** | Compare last-token logits vs PyTorch reference |
| **CPU path first** | Scalar/oracle ops in `src/backends/cpu/ops.zig` |
| **BF16 weights** | Promote to f32 at load (`src/runtime/bf16.zig`) |
| **HF linear layout** | Transpose `[out,in]` → `[in,out]` on load |
| **CLI / tests** | `forward-golden`, `artifact-compile --mini`, `zig build stage11` |
| **Debug dumps** | `--dump DIR` + fixture scripts for layer bisection |

## Explicitly not Stage 11 (→ Stage 12 or out of CI)

| Item | Owner | Notes |
| --- | --- | --- |
| **Tokenizer** | Stage 12 | Stage 11 uses fixed token-id test vectors |
| **Sampling / greedy decode loop** | Stage 12 | Stage 11 compares logits, not generated text |
| **Vocabulary TTFT / tok/s / ITL** | Stage 12 | Benchmark matrix stays N/A until real tokens |
| **HF download in CI** | Never | Weights stay local/gitignored; CI uses mini fixture only |
| **Metal Qwen forward** | After CPU golden | Apple path still uses tiny-block fixture |

## Prerequisites

- Zig 0.16.0 (`zig build`, `zig build test -Dhip=off`)
- Stage 10 artifact loader (`Artifact.loadFile`, `findById`)
- **CI / no weights:** nothing else — `zig build stage11` builds the mini fixture
- **Full model (local):**
  - Hugging Face weights under `models/Qwen3-0.6B/` (gitignored)
  - Compiled artifact `models/qwen3-0.6b.zynfer` (~1.5 GiB BF16, mmap)
  - For golden generation: Python 3.12+ with `torch` and `transformers`
    (`pip install torch transformers`)

See also `tools/checkpoint/README.md` for safetensors → `.zynfer` conversion.

## Quick start

### CI / mini fixture (no HF weights)

```bash
zig build test -Dhip=off
zig build stage11 -Dhip=off
```

This builds `zig-out/stage11-mini.zynfer`, runs forward on tokens `2,3`, and
checks determinism.

Manual:

```bash
./zig-out/bin/zynfer artifact-compile --mini --out zig-out/stage11-mini.zynfer
./zig-out/bin/zynfer forward-golden zig-out/stage11-mini.zynfer --tokens=2,3
```

### Full Qwen3-0.6B (local)

**1. Download weights and compile artifact** (one-time, not in CI):

```bash
hf download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B

python3 tools/checkpoint/safetensors_to_zynfer.py \
  --config models/Qwen3-0.6B/config.json \
  --weights models/Qwen3-0.6B \
  --out models/qwen3-0.6b.zynfer
```

**2. Smoke test** (prints top-8 logits, no golden file):

```bash
./zig-out/bin/zynfer forward-golden models/qwen3-0.6b.zynfer --tokens=151643,2,3
```

Default tokens when `--tokens` is omitted: `151643,2,3` (BOS + ids `2` and `3`
for Qwen3-0.6B).

**3. Golden compare (Option B — recommended for full-model validation):**

Generate a reference logits file with PyTorch, then compare:

```bash
python3 tools/fixtures/gen_golden_logits.py \
  --model models/Qwen3-0.6B \
  --tokens=151643,2,3 \
  --out ref_logits.f32

./zig-out/bin/zynfer forward-golden models/qwen3-0.6b.zynfer \
  --tokens=151643,2,3 \
  --golden ref_logits.f32
```

Success looks like:

```text
golden OK (ref_logits.f32, vocab=151936)
top logits (last token):
  id=400 logit=13.271963
  ...
```

**Important:** `--tokens` must be **identical** in both commands. The golden
file is a raw little-endian `f32` blob of shape `[vocab_size]` (607 744 bytes
for Qwen3-0.6B). It is not checked into git.

Tolerances in `forward-golden`: absolute `1e-3`, relative `1e-2`
(`src/runtime/compare.zig`).

## CLI reference

```text
zynfer forward-golden ARTIFACT.zynfer [--tokens IDS] [--golden PATH] [--dump DIR]
```

| Flag | Description |
| --- | --- |
| `ARTIFACT` | Path to `.zynfer` (mini fixture or full model) |
| `--tokens=ID,ID,...` | Comma-separated token ids (default: BOS,2,3 from artifact meta) |
| `--golden=PATH` | Raw f32 last-token logits; run fails on `NumericalMismatch` if wrong |
| `--dump=DIR` | Write intermediate f32 blobs for debugging (see below) |

Other commands:

```bash
zynfer stage11              # ledger + pointers
zynfer inspect PATH.zynfer  # metadata + tensor list
zynfer artifact-compile --mini --out PATH
```

## Golden generation (`gen_golden_logits.py`)

Development-time only; requires PyTorch + transformers.

```bash
pip install torch transformers

python3 tools/fixtures/gen_golden_logits.py --help
```

Defaults:

- `--model models/Qwen3-0.6B`
- `--tokens 151643,2,3`
- `--out ref_logits.f32`

The script loads `AutoModelForCausalLM` in float32 with
`attn_implementation="eager"` (matches the scalar CPU attention path; SDPA/flash
can diverge slightly). It writes **last prompt token** logits and prints top-8
to stderr.

## Debugging mismatches

If `--golden` fails with `NumericalMismatch`, treat it as a real forward bug
(not tolerance noise) when top-token ids differ.

### Step 1 — verify tokens and golden file

- Same `--tokens` in `gen_golden_logits.py` and `forward-golden`
- Golden file size = `vocab_size × 4` bytes (151936 × 4 = 607744)
- Regenerate golden after any PyTorch/transformers upgrade if needed

### Step 2 — dump zynfer intermediates

```bash
rm -rf zynfer_dump
./zig-out/bin/zynfer forward-golden models/qwen3-0.6b.zynfer \
  --tokens=151643,2,3 \
  --dump=zynfer_dump
```

Writes (last token row unless noted):

| File | Contents |
| --- | --- |
| `embed_last.f32` | Embedding output for last token `[1024]` |
| `layer00.f32` … `layer27.f32` | Hidden state after each block `[1024]` |
| `normed.f32` | After final RMSNorm `[1024]` |
| `logits.f32` | Full vocab logits `[151936]` |

### Step 3 — compare against numpy replay (no torch)

`tools/fixtures/ref_forward_numpy.py` reimplements the CPU math from
safetensors (BF16 → f32, transposed linears, QK-norm, RoPE, GQA, SwiGLU):

```bash
pip install numpy   # safetensors read is manual; no torch required

python3 tools/fixtures/ref_forward_numpy.py \
  --tokens=151643,2,3 \
  --out-dir ref_numpy \
  --compare-dir zynfer_dump
```

Prints per-tensor `max_abs` / `rms` vs zynfer dumps. Use this to bisect:
embed → layer00 → … → normed → logits.

### Step 4 — PyTorch layer hook (optional)

```bash
pip install torch transformers

python3 tools/fixtures/dump_forward_refs.py \
  --tokens=151643,2,3 \
  --out-dir ref_forward
```

Writes `layer00.f32`, `normed.f32`, `logits.f32` from Hugging Face forward
with hooks.

### Interpreting results

| Symptom | Likely cause |
| --- | --- |
| `embed_last` mismatch | Weight load / BF16 decode / wrong token id |
| `layer00` ok, later layers drift | Block math (attention, RoPE, MLP, QK-norm) |
| All `layerNN` ok, `normed` wrong | Final norm on wrong token row or bad `model.norm.weight` |
| `normed` ok, `logits` wrong | LM head / tied embed matvec |

**Last-token pitfall:** prefill stores `[num_tokens, hidden]`. Final norm must
use the **last row**, not row 0. Zynfer uses `Tensor.viewLastRow()` for this
(`src/runtime/tensor.zig`); `viewAs(&.{1, hidden})` only views the prefix.

## Exit criterion

For token ids `151643,2,3` on the full local artifact:

- `forward-golden --golden ref_logits.f32` prints `golden OK`
- Top logits match PyTorch within stated tolerances
- `zig build stage11` passes on CI without HF weights

## Implementation map

| Area | Path |
| --- | --- |
| Architecture constants | `src/model/qwen3.zig` |
| Weight load / transpose | `src/model/qwen_weights.zig` |
| Transformer block | `src/model/qwen_block.zig` |
| Full forward + mini fixture | `src/model/qwen_forward.zig` |
| CPU ops (oracle) | `src/backends/cpu/ops.zig` |
| BF16 decode | `src/runtime/bf16.zig` |
| Golden compare | `src/runtime/compare.zig` |
| CLI | `src/main.zig` (`forward-golden`, `--dump`) |
| Build / CI step | `build.zig` (`stage11`) |

## Bench report

See `bench/results/stage11-dev-laptop.md`.

Hands-on walkthrough: [`docs/tutorials/11-qwen-forward-and-golden.md`](../tutorials/11-qwen-forward-and-golden.md).
Fixture scripts: [`tools/fixtures/README.md`](../../tools/fixtures/README.md).
