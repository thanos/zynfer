# Checkpoint conversion tools

`safetensors_to_zynfer.py` turns a local Hugging Face Safetensors tree into a
`.zynfer` artifact the Zig runtime can `inspect` / load.

This path is **optional** for Stage 10. CI and `artifact-compile` use a tiny
Zig fixture only. Full Qwen weights live under `models/` (gitignored).

See also: `docs/artifact-format.md`, `docs/stages/10-checkpoint-artifact.md`.

## Prerequisites

| Need | Notes |
| --- | --- |
| Python 3.10+ | Stdlib only for the converter (no NumPy, no `safetensors` pip package) |
| ~3+ GiB free disk | HF download ~1.5 GiB + `.zynfer` ~1.5 GiB |
| ~2+ GiB free RAM | Converter loads the Safetensors file into memory once |
| `hf` CLI | From `huggingface_hub` — **not** deprecated `huggingface-cli` |
| Built `zynfer` | Only to `inspect` afterward (`zig build -Dhip=off`) |

Install the Hub CLI once:

```bash
pip install -U "huggingface_hub[cli]"
# confirm: hf --help
```

## Steps

### 1. Download Qwen3-0.6B

From the repo root:

```bash
mkdir -p models
hf download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B
```

Expect at least:

- `models/Qwen3-0.6B/config.json`
- `models/Qwen3-0.6B/model.safetensors` (single file for this repo; BF16)

### 2. Convert to `.zynfer`

```bash
python3 tools/checkpoint/safetensors_to_zynfer.py \
  --config models/Qwen3-0.6B/config.json \
  --weights models/Qwen3-0.6B/model.safetensors \
  --out models/qwen3-0.6b.zynfer
```

Successful stderr looks like:

```text
wrote models/qwen3-0.6b.zynfer (… bytes, 311 tensors)
```

Qwen weights are **BF16**. The converter copies raw Safetensors bytes, so it
does not need NumPy/`ml_dtypes` (those fail with `data type 'bfloat16' not
understood` if you use `safe_open(..., framework="np")`).

### 3. Validate with Zig

```bash
zig build -Dhip=off
./zig-out/bin/zynfer inspect models/qwen3-0.6b.zynfer
```

You should see `model_id: qwen3-0.6b`, `dtype=bf16`, and 311 tensors.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `huggingface-cli` deprecated warning | Use `hf download …` |
| `FileNotFoundError: path/to/config.json` | Use real paths under `models/Qwen3-0.6B/` |
| `data type 'bfloat16' not understood` | Update to the raw-byte converter (current `tools/checkpoint/safetensors_to_zynfer.py`) |
| `inspect` / `StreamTooLong` / panic on large file | Rebuild after Stage 10 loadFile fix (`zig build -Dhip=off`) |
| Sharded `model-00001-of-0000N.safetensors` | This script takes one file; merge or pass the single-file checkpoint |

## What this does *not* do

- No forward pass (Stage 11)
- No tokenizer / sampling (Stage 12)
- Does not commit weights — keep `models/` local
