#!/usr/bin/env python3
"""Convert a Hugging Face Safetensors checkpoint into a .zynfer artifact.

Development-time only. The Zig runtime never imports this module.

Requires: Python 3.10+ only for the converter core. Parses Safetensors
headers and copies raw tensor bytes (including BF16) without NumPy.
Optional: `safetensors` is unused; install nothing beyond stdlib for convert.

Example:
  # Prefer `hf` — `huggingface-cli` is deprecated.
  hf download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B
  python3 tools/checkpoint/safetensors_to_zynfer.py \\
    --config models/Qwen3-0.6B/config.json \\
    --weights models/Qwen3-0.6B/model.safetensors \\
    --out models/qwen3-0.6b.zynfer
"""

from __future__ import annotations

import argparse
import json
import struct
import hashlib
import sys
from pathlib import Path

MAGIC = b"ZYNF"
VERSION = 1
ENDIAN = 1
PAYLOAD_ALIGN = 64
HEADER_SIZE = 88
META_SIZE = 116
ENTRY_SIZE = 120

DTYPE_MAP = {
    "F32": (0, 4),
    "F16": (1, 2),
    "BF16": (2, 2),
}


def align_up(v: int, a: int) -> int:
    return (v + a - 1) // a * a


def pad_name(name: str) -> bytes:
    raw = name.encode("utf-8")
    if not raw or len(raw) >= 64:
        raise ValueError(f"bad tensor name: {name!r}")
    return raw + b"\x00" * (64 - len(raw))


def write_meta(cfg: dict) -> bytes:
    model_id = b"qwen3-0.6b" + b"\x00" * (64 - len(b"qwen3-0.6b"))
    tie = 1 if cfg.get("tie_word_embeddings", True) else 0
    return struct.pack(
        "<64s10I2fB3x",
        model_id,
        int(cfg["vocab_size"]),
        int(cfg["hidden_size"]),
        int(cfg["intermediate_size"]),
        int(cfg["num_hidden_layers"]),
        int(cfg["num_attention_heads"]),
        int(cfg["num_key_value_heads"]),
        int(cfg["head_dim"]),
        int(cfg["max_position_embeddings"]),
        int(cfg.get("bos_token_id", 0)),
        int(cfg.get("eos_token_id", 0)),
        float(cfg.get("rope_theta", 10000.0)),
        float(cfg.get("rms_norm_eps", 1e-6)),
        tie,
    )


def load_safetensors(path: Path) -> list[tuple[str, int, list[int], bytes]]:
    """Return (name, dtype_tag, shape, payload_bytes).

    Reads the Safetensors file layout directly so BF16 works without NumPy
    or ml_dtypes (Qwen3 ships BF16; `safe_open(..., framework=\"np\")` fails).
    """
    data = path.read_bytes()
    if len(data) < 8:
        raise SystemExit(f"truncated safetensors file: {path}")
    header_len = struct.unpack_from("<Q", data, 0)[0]
    header_end = 8 + header_len
    if header_end > len(data):
        raise SystemExit(f"truncated safetensors header: {path}")
    header = json.loads(data[8:header_end].decode("utf-8"))

    out: list[tuple[str, int, list[int], bytes]] = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype = info["dtype"]
        if dtype not in DTYPE_MAP:
            raise SystemExit(f"unsupported safetensors dtype {dtype!r} for {name}")
        tag, width = DTYPE_MAP[dtype]
        shape = [int(x) for x in info["shape"]]
        start, stop = info["data_offsets"]
        start = int(start)
        stop = int(stop)
        abs_start = header_end + start
        abs_stop = header_end + stop
        if abs_stop > len(data) or abs_start < header_end:
            raise SystemExit(f"bad data_offsets for {name}")
        raw = data[abs_start:abs_stop]
        expect = width
        for d in shape:
            expect *= d
        if len(raw) != expect:
            raise SystemExit(
                f"size mismatch for {name}: got {len(raw)} want {expect} "
                f"(dtype={dtype} shape={shape})"
            )
        out.append((name, tag, shape, raw))

    # Stable order for deterministic artifacts.
    out.sort(key=lambda t: t[0])
    return out


def build_artifact(meta: bytes, tensors: list[tuple[str, int, list[int], bytes]]) -> bytes:
    assert len(meta) == META_SIZE
    dir_bytes = len(tensors) * ENTRY_SIZE
    meta_offset = HEADER_SIZE
    dir_offset = meta_offset + META_SIZE
    after_dir = dir_offset + dir_bytes
    payload_offset = align_up(after_dir, PAYLOAD_ALIGN)

    entries = bytearray()
    payload = bytearray()
    cursor = 0
    for i, (name, tag, shape, raw) in enumerate(tensors):
        if len(shape) == 0 or len(shape) > 8:
            raise SystemExit(f"bad rank for {name}")
        off = align_up(cursor, PAYLOAD_ALIGN)
        if off > cursor:
            payload.extend(b"\x00" * (off - cursor))
        payload.extend(raw)
        cursor = off + len(raw)
        shape_pad = shape + [0] * (8 - len(shape))
        entries.extend(
            struct.pack(
                "<64sIBB2x8IQQ",
                pad_name(name),
                i + 1,
                tag,
                len(shape),
                *shape_pad,
                off,
                len(raw),
            )
        )

    header = struct.pack(
        "<4sHBBIIIIIIIxxxxQQ32x",
        MAGIC,
        VERSION,
        ENDIAN,
        0,
        HEADER_SIZE,
        meta_offset,
        META_SIZE,
        dir_offset,
        dir_bytes,
        len(tensors),
        0,
        payload_offset,
        len(payload),
    )
    # struct above may not match Zig padding — build header explicitly
    hdr = bytearray(HEADER_SIZE)
    struct.pack_into("<4sHBB", hdr, 0, MAGIC, VERSION, ENDIAN, 0)
    struct.pack_into("<IIIIII", hdr, 8, HEADER_SIZE, meta_offset, META_SIZE, dir_offset, dir_bytes, len(tensors))
    struct.pack_into("<I", hdr, 32, 0)  # reserved0
    # 4 bytes pad then payload_offset at 40
    struct.pack_into("<QQ", hdr, 40, payload_offset, len(payload))
    # sha256 at 56

    body = bytearray()
    body.extend(hdr)
    body.extend(meta)
    body.extend(entries)
    if payload_offset > len(body):
        body.extend(b"\x00" * (payload_offset - len(body)))
    body.extend(payload)

    # SHA-256 with checksum field zeroed (already zero)
    digest = hashlib.sha256(body).digest()
    body[56:88] = digest
    return bytes(body)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True, help="HF config.json")
    ap.add_argument("--weights", type=Path, required=True, help="model.safetensors")
    ap.add_argument("--out", type=Path, required=True, help="output .zynfer path")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    meta = write_meta(cfg)
    tensors = load_safetensors(args.weights)
    blob = build_artifact(meta, tensors)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(blob)
    print(f"wrote {args.out} ({len(blob)} bytes, {len(tensors)} tensors)", file=sys.stderr)


if __name__ == "__main__":
    main()
