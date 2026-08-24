#!/usr/bin/env python3
"""Pack projection weights in a .zynfer artifact to per-row int8 + f32 scales.

Development-time only. Validates the M5 scheme against a CPU dequant round-trip
in this script; Zig `qwen_quant.dequantRowQ8` is the runtime decoder twin.

Scheme (matches src/model/qwen_quant.zig):
  - Keep norms / embed as original dtype
  - For each *proj* / lm_head linear: store HF layout [out, in] as i8 (dtype=3)
    plus sibling `{name}.qscale` f32 [out]
  - Original float weight tensor is replaced by the i8 payload

TensorEntry layout must match Zig `artifact.TensorEntry` / safetensors converter:
  `<64s I B B 2x 8I Q Q>` (2-byte pad after rank).

Example:
  python3 tools/checkpoint/quantize_zynfer_int8.py \\
    --in models/qwen3-0.6b.zynfer \\
    --out models/qwen3-0.6b-int8.zynfer
"""

from __future__ import annotations

import argparse
import hashlib
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = b"ZYNF"
VERSION = 1
ENDIAN = 1
PAYLOAD_ALIGN = 64
HEADER_SIZE = 88
META_SIZE = 116
ENTRY_SIZE = 120
# name(64) + id(4) + dtype(1) + rank(1) + pad(2) + shape(32) + off(8) + nbytes(8)
ENTRY_FMT = "<64sIBB2x8IQQ"

PROJ_SUFFIXES = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
)


def align_up(v: int, a: int) -> int:
    return (v + a - 1) // a * a


def is_proj(name: str) -> bool:
    if name == "lm_head.weight":
        return True
    return any(name.endswith(s) for s in PROJ_SUFFIXES)


def decode_to_f32(tag: int, raw: bytes, rows: int, cols: int) -> np.ndarray:
    n = rows * cols
    if tag == 0:  # f32
        return np.frombuffer(raw, dtype="<f4", count=n).reshape(rows, cols).astype(np.float32, copy=False)
    if tag == 1:  # f16
        return np.frombuffer(raw, dtype="<f2", count=n).reshape(rows, cols).astype(np.float32)
    if tag == 2:  # bf16 stored as u16
        u = np.frombuffer(raw, dtype="<u2", count=n).astype(np.uint32)
        f = (u << 16).view(np.float32)
        return f.reshape(rows, cols)
    raise SystemExit(f"unsupported dtype tag {tag}")


def pack_matrix_q8(f32: np.ndarray) -> tuple[bytes, bytes, float]:
    """Per-row symmetric int8: scale = max_abs/127; q = round(w/scale)."""
    rows, cols = f32.shape
    max_abs = np.max(np.abs(f32), axis=1)
    scale = np.where(max_abs == 0, 1.0, max_abs / 127.0).astype(np.float32)
    q = np.clip(np.rint(f32 / scale[:, None]), -127, 127).astype(np.int8)
    recon = q.astype(np.float32) * scale[:, None]
    worst = float(np.max(np.abs(f32 - recon)))
    return q.tobytes(), scale.astype("<f4", copy=False).tobytes(), worst


def read_artifact(path: Path) -> tuple[bytes, list[tuple]]:
    data = path.read_bytes()
    if data[:4] != MAGIC:
        raise SystemExit("bad magic")
    tensor_count = struct.unpack_from("<I", data, 28)[0]
    dir_offset = struct.unpack_from("<I", data, 20)[0]
    payload_offset = struct.unpack_from("<Q", data, 40)[0]
    meta = data[88 : 88 + META_SIZE]
    tensors = []
    for i in range(tensor_count):
        off = dir_offset + i * ENTRY_SIZE
        name_b, tensor_id, dtype, rank, *rest = struct.unpack_from(ENTRY_FMT, data, off)
        shape = rest[:8][:rank]
        toff, nbytes = rest[8], rest[9]
        name = name_b.split(b"\x00", 1)[0].decode()
        raw = data[payload_offset + toff : payload_offset + toff + nbytes]
        if len(raw) != nbytes:
            raise SystemExit(f"truncated payload for {name}")
        tensors.append((name, tensor_id, dtype, list(shape), raw))
    return meta, tensors


def build_artifact(meta: bytes, tensors: list[tuple]) -> bytes:
    tensors = sorted(tensors, key=lambda t: t[0])
    dir_offset = HEADER_SIZE + META_SIZE
    dir_bytes = len(tensors) * ENTRY_SIZE
    payload_offset = align_up(dir_offset + dir_bytes, PAYLOAD_ALIGN)
    payload = bytearray()
    entries = bytearray()
    for i, (name, _old_id, tag, shape, raw) in enumerate(tensors):
        tensor_id = i + 1
        while len(payload) % PAYLOAD_ALIGN:
            payload.append(0)
        off = len(payload)
        payload.extend(raw)
        name_b = name.encode() + b"\x00" * (64 - len(name.encode()))
        shape_pad = list(shape) + [0] * (8 - len(shape))
        entries.extend(
            struct.pack(
                ENTRY_FMT,
                name_b,
                tensor_id,
                tag,
                len(shape),
                *shape_pad,
                off,
                len(raw),
            )
        )
    hdr = bytearray(HEADER_SIZE)
    struct.pack_into("<4sHBB", hdr, 0, MAGIC, VERSION, ENDIAN, 0)
    struct.pack_into(
        "<IIIIII",
        hdr,
        8,
        HEADER_SIZE,
        88,
        META_SIZE,
        dir_offset,
        dir_bytes,
        len(tensors),
    )
    struct.pack_into("<I", hdr, 32, 0)
    struct.pack_into("<QQ", hdr, 40, payload_offset, len(payload))
    body = bytearray()
    body.extend(hdr)
    body.extend(meta)
    body.extend(entries)
    if payload_offset > len(body):
        body.extend(b"\x00" * (payload_offset - len(body)))
    body.extend(payload)
    digest = hashlib.sha256(body).digest()
    body[56:88] = digest
    return bytes(body)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-err", type=float, default=0.05, help="abort if dequant max abs err exceeds")
    args = ap.parse_args()

    meta, tensors = read_artifact(args.inp)
    out_tensors: list[tuple] = []
    n_quant = 0
    worst = 0.0
    for name, tid, tag, shape, raw in tensors:
        if not is_proj(name) or len(shape) != 2:
            out_tensors.append((name, tid, tag, shape, raw))
            continue
        rows, cols = shape[0], shape[1]
        expect = rows * cols * (4 if tag == 0 else 2)
        if len(raw) != expect:
            raise SystemExit(f"{name}: nbytes {len(raw)} != {expect} for dtype {tag} shape {shape}")
        f32 = decode_to_f32(tag, raw, rows, cols)
        q, scales, err = pack_matrix_q8(f32)
        worst = max(worst, err)
        if err > args.max_err:
            raise SystemExit(f"{name}: dequant max abs err {err} > {args.max_err}")
        out_tensors.append((name, tid, 3, shape, q))  # i8
        out_tensors.append((name + ".qscale", tid, 0, [rows], scales))  # f32 scales
        n_quant += 1
        print(f"  quantized {name} shape={shape} err={err:.6g}", file=sys.stderr)

    blob = build_artifact(meta, out_tensors)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(blob)
    print(
        f"wrote {args.out} ({len(blob)} bytes, quantized {n_quant} projections, "
        f"worst_dequant_abs_err={worst:.6g})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
