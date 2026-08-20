# Checkpoint conversion tools

`safetensors_to_zynfer.py` — optional Hugging Face → `.zynfer` converter.
See `docs/artifact-format.md`.

## Download Qwen3-0.6B

Use the Hugging Face `hf` CLI. Do **not** use `huggingface-cli` (deprecated).

```bash
pip install -U "huggingface_hub[cli]" safetensors numpy

hf download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B
```

`models/` is gitignored and is not required for Stage 10 fixture tests.

## Convert

```bash
python3 tools/checkpoint/safetensors_to_zynfer.py \
  --config models/Qwen3-0.6B/config.json \
  --weights models/Qwen3-0.6B/model.safetensors \
  --out models/qwen3-0.6b.zynfer

./zig-out/bin/zynfer inspect models/qwen3-0.6b.zynfer
```
