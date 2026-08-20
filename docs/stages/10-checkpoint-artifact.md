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

Not required for Stage 10 CI. `models/` is gitignored. Use `hf` (the
`huggingface-cli` entry point is deprecated):

```bash
pip install -U "huggingface_hub[cli]" safetensors numpy

hf download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B

python3 tools/checkpoint/safetensors_to_zynfer.py \
  --config models/Qwen3-0.6B/config.json \
  --weights models/Qwen3-0.6B/model.safetensors \
  --out models/qwen3-0.6b.zynfer

./zig-out/bin/zynfer inspect models/qwen3-0.6b.zynfer
```

See `docs/artifact-format.md`, `docs/tutorials/10-checkpoint-and-artifact.md`,
and `bench/results/stage10-dev-laptop.md`.
