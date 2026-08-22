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
| ~2+ GiB free RAM | Converter loads shard(s) into memory once |
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

Expect at least `config.json` and either:

- one `model.safetensors`, or
- sharded `model-0000N-of-0000M.safetensors` plus optional
  `model.safetensors.index.json`

### 2. Convert to `.zynfer`

**Single file:**

```bash
python3 tools/checkpoint/safetensors_to_zynfer.py \
  --config models/Qwen3-0.6B/config.json \
  --weights models/Qwen3-0.6B/model.safetensors \
  --out models/qwen3-0.6b.zynfer
```

**Directory (auto-discovers shards / index.json):**

```bash
python3 tools/checkpoint/safetensors_to_zynfer.py \
  --config models/Qwen3-0.6B/config.json \
  --weights models/Qwen3-0.6B \
  --out models/qwen3-0.6b.zynfer
```

**Explicit shard list:**

```bash
python3 tools/checkpoint/safetensors_to_zynfer.py \
  --config models/Qwen3-0.6B/config.json \
  --weights models/Qwen3-0.6B/model-00001-of-00002.safetensors \
            models/Qwen3-0.6B/model-00002-of-00002.safetensors \
  --out models/qwen3-0.6b.zynfer
```

Successful stderr looks like:

```text
loading 1 safetensors file(s)
wrote models/qwen3-0.6b.zynfer (… bytes, 311 tensors, ids 1..311)
```

Tensor **ids are 1..N in sorted name order** (stable for `Artifact.findById`).

Qwen weights are **BF16**. The converter copies raw Safetensors bytes (no
NumPy / `ml_dtypes`).

### 3. Validate with Zig

```bash
zig build -Dhip=off
./zig-out/bin/zynfer inspect models/qwen3-0.6b.zynfer
```

Expect `storage: mmap` on macOS/Linux, `model_id: qwen3-0.6b`, `dtype=bf16`.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `huggingface-cli` deprecated warning | Use `hf download …` |
| `FileNotFoundError: path/to/config.json` | Use real paths under `models/Qwen3-0.6B/` |
| `data type 'bfloat16' not understood` | Use current raw-byte converter (not `safe_open` + NumPy) |
| `duplicate tensor … across shards` | Overlapping shards / bad index — fix the checkpoint tree |
| `inspect` fails on huge file | Rebuild current tree (`loadFile` mmap + 4 GiB cap) |

## What this does *not* do

- No forward pass (Stage 11)
- No tokenizer / sampling (Stage 12)
- Does not commit weights — keep `models/` local
