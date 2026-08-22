# Forward fixtures and golden references

Development-time helpers for Stage 11 (Qwen CPU forward). **Not used in CI**
except indirectly via the checked-in mini artifact built by Zig.

Full Qwen3-0.6B weights and golden blobs stay local / gitignored.

## In plain English

`forward-golden` runs Qwen on a short list of token numbers and either **shows**
the top prediction scores or **proves** they match PyTorch (the “golden”
answer key in `ref_logits.f32`). These scripts build that answer key and help
debug mismatches. Full layman walkthrough:
[`docs/stages/11-qwen-forward.md`](../../docs/stages/11-qwen-forward.md#in-plain-english)
and [`docs/tutorials/11-qwen-forward-and-golden.md`](../../docs/tutorials/11-qwen-forward-and-golden.md#in-plain-english).

## Scripts

| Script | Requires | Purpose |
| --- | --- | --- |
| `gen_golden_logits.py` | torch, transformers | Last-token logits from Hugging Face → `ref_logits.f32` |
| `ref_forward_numpy.py` | numpy | Replay forward from safetensors; bisect vs zynfer `--dump` |
| `dump_forward_refs.py` | torch, transformers | HF hooks: layer0 / normed / logits f32 dumps |

Canonical workflow and troubleshooting: [`docs/stages/11-qwen-forward.md`](../../docs/stages/11-qwen-forward.md).

## Generate golden logits

```bash
pip install torch transformers

python3 tools/fixtures/gen_golden_logits.py \
  --model models/Qwen3-0.6B \
  --tokens=151643,2,3 \
  --out ref_logits.f32
```

Compare with zynfer (same tokens):

```bash
./zig-out/bin/zynfer forward-golden models/qwen3-0.6b.zynfer \
  --tokens=151643,2,3 \
  --golden ref_logits.f32
```

Output format: raw little-endian `f32[vocab_size]`.

Uses `attn_implementation="eager"` so attention matches the scalar CPU path.

## Dump zynfer intermediates

```bash
./zig-out/bin/zynfer forward-golden models/qwen3-0.6b.zynfer \
  --tokens=151643,2,3 \
  --dump=zynfer_dump
```

Files: `embed_last.f32`, `layer00.f32` … `layer27.f32`, `normed.f32`, `logits.f32`
(last-token hidden states are 1024 floats; logits are full vocab).

## Numpy reference (no torch)

```bash
pip install numpy

python3 tools/fixtures/ref_forward_numpy.py \
  --model models/Qwen3-0.6B/model.safetensors \
  --tokens=151643,2,3 \
  --out-dir ref_numpy \
  --compare-dir zynfer_dump
```

Slow (~1 min) — reloads weights from safetensors each layer; fine for debugging.

## PyTorch intermediate dumps

```bash
python3 tools/fixtures/dump_forward_refs.py \
  --tokens=151643,2,3 \
  --out-dir ref_forward
```
