# Stage 10 — Checkpoint inspection and artifact compiler

Exit: a deterministic conversion produces a `.zynfer` file the Zig
runtime validates and loads.

## Commands

```bash
zig build test -Dhip=off
zig build stage10
./zig-out/bin/zynfer artifact-compile --out stage10-fixture.zynfer
./zig-out/bin/zynfer inspect stage10-fixture.zynfer
```

### Optional: Hugging Face Qwen3-0.6B → `.zynfer`

Not required for Stage 10 CI. Full checklist (Python, `hf`, disk/RAM,
BF16 notes): **`tools/checkpoint/README.md`**.

```bash
pip install -U "huggingface_hub[cli]"
hf download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B

python3 tools/checkpoint/safetensors_to_zynfer.py \
  --config models/Qwen3-0.6B/config.json \
  --weights models/Qwen3-0.6B/model.safetensors \
  --out models/qwen3-0.6b.zynfer

zig build -Dhip=off
./zig-out/bin/zynfer inspect models/qwen3-0.6b.zynfer
```

Do not use `huggingface-cli` (deprecated). Converter is BF16-safe (raw
bytes; no NumPy). See `docs/artifact-format.md` and
`bench/results/stage10-dev-laptop.md`.
