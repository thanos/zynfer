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
  --weights models/Qwen3-0.6B \
  --out models/qwen3-0.6b.zynfer

zig build -Dhip=off
./zig-out/bin/zynfer inspect models/qwen3-0.6b.zynfer
```

`--weights` accepts a file, a directory (shards + optional
`model.safetensors.index.json`), or multiple shard paths. See
`tools/checkpoint/README.md`.

Zig `Artifact.loadFile` uses **mmap** when available (`storage: mmap` in
`inspect`). Hot path: `findById` / `tensorBytesById` (ids 1..N sorted by
name). Names remain for inspect/debug.

