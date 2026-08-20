# Tutorial 10 — Checkpoints and the `.zynfer` artifact

## What we are building

A tiny native model artifact (`.zynfer`) plus inspect/load tooling so
Stage 11 can run a real forward without teaching the runtime Safetensors.

## Why it exists

Upstream checkpoints are convenient for research and awkward for a
specialized inference engine: sharding, string tensor names, Python-centric
dtypes, and slow cold-start parsing. NInfer-style runtimes compile once into
an engine-owned layout.

## The underlying concept

```text
Hugging Face (Safetensors + config.json)
        │  development-time converter
        ▼
     .zynfer  (header + meta + directory + aligned payloads + SHA-256)
        │  Zig validate / load
        ▼
   Stage 11 forward
```

## How the hardware sees it

Weights become ordinary host (and later GPU-shared) byte ranges. Alignment
and contiguous payloads matter for mmap and Metal/HIP buffer creation;
string parsing does not belong in the decode loop.

## How the math maps to code

Stage 10 does not run matmul. It stores **architecture metadata** (Qwen3-0.6B
dims from HF) and **named tensor blobs** the forward pass will bind by name
or numeric id.

## How the Zig implementation works

- `src/model/qwen3.zig` — architecture constants
- `src/model/artifact.zig` — build / validate / `Artifact.load*`
- CLI: `artifact-compile`, `inspect`, `stage10`

## How the GPU implementation works

Not yet. Stage 10 is host I/O. Metal/HIP consume loaded buffers in later
stages.

## How correctness is checked

Unit tests round-trip a fixture, reject bad magic/checksum, and integration
tests compile then inspect via the installed binary.

## How performance is measured

Artifact size and load path only. TTFT stays N/A until Stage 12.

## What surprised us

Zig 0.16 `extern struct` padding: the v1 `Header` is **88** bytes because of
alignment before the `u64` payload fields—document sizes from `@sizeOf`, not
hand sums.

## Exercises

1. Corrupt one payload byte and confirm `inspect` exits 2.
2. Add a third fixture tensor and re-run round-trip tests.
3. Optional — download Qwen3-0.6B and convert (use `hf`, not
   `huggingface-cli`):

```bash
pip install -U "huggingface_hub[cli]" safetensors numpy
hf download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B
python3 tools/checkpoint/safetensors_to_zynfer.py \
  --config models/Qwen3-0.6B/config.json \
  --weights models/Qwen3-0.6B/model.safetensors \
  --out models/qwen3-0.6b.zynfer
./zig-out/bin/zynfer inspect models/qwen3-0.6b.zynfer
```

## Further experiments

Map HF tensor names → stable `tensor_id` values for Stage 11 hot path.
