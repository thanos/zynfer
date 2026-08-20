# `.zynfer` artifact format (v1)

Native model weights for the Zig runtime. Hot path never parses Safetensors;
conversion is a development-time step (`tools/checkpoint/`).

## Layout (little-endian)

```text
[Header 88 bytes]
[Meta 116 bytes]
[TensorEntry × N]   N = Header.tensor_count; each entry 120 bytes
[padding to 64-byte alignment]
[payload bytes]     each tensor starts on a 64-byte boundary
```

### Header

| Field | Type | Notes |
| --- | --- | --- |
| magic | `[4]u8` | `ZYNF` |
| version | `u16` | `1` |
| endian | `u8` | `1` = little |
| flags | `u8` | `0` (reserved) |
| header_bytes | `u32` | `88` |
| meta_offset | `u32` | `88` |
| meta_bytes | `u32` | `116` |
| dir_offset | `u32` | `204` |
| dir_bytes | `u32` | `N * 120` |
| tensor_count | `u32` | |
| reserved0 | `u32` | `0` |
| payload_offset | `u64` | 64-byte aligned |
| payload_bytes | `u64` | |
| sha256 | `[32]u8` | hash of whole file with this field zeroed |

### Meta

Architecture parameters (Qwen3-0.6B values match HF `config.json`):
`model_id` (NUL-padded 64), vocab/hidden/intermediate/layers/heads/kv/head_dim/
max_position/bos/eos, `rope_theta`, `rms_norm_eps`, `tie_word_embeddings`.

### TensorEntry

NUL-padded `name[64]`, optional numeric `tensor_id`, `dtype` (`0=f32`,
`1=f16`, `2=bf16`), `rank`, `shape[8]`, `offset` (relative to payload),
`nbytes`.

## Commands

```bash
zynfer artifact-compile --out stage10-fixture.zynfer
zynfer inspect stage10-fixture.zynfer
zynfer stage10
```

### Optional: real Qwen3-0.6B weights

`models/` is gitignored. Use the Hugging Face `hf` CLI (not the deprecated
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

If the hub ships sharded `*.safetensors` files, point `--weights` at the
primary shard the converter supports, or merge first — Stage 10’s Zig
fixture path does not need a download.

## Stage boundary

Stage 10: validate + load. Stage 11: forward using loaded tensors.
