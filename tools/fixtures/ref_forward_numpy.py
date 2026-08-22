#!/usr/bin/env python3
"""Numpy Qwen3 forward from safetensors (no torch) for zynfer bisection.

Matches zynfer CPU math: BF16 weights, transposed linears, QK-norm, RoPE, GQA.

  python3 tools/fixtures/ref_forward_numpy.py --tokens 151643,2,3 --out-dir zynfer_dump

Compare zynfer dumps (--dump DIR) against this script's output in the same layout:
  embed_last.f32, layer00.f32 … layer27.f32, normed.f32, logits.f32
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def bf16_bytes_to_f32(raw: bytes) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype="<u2")
    u32 = u16.astype(np.uint32) << 16
    return u32.view("<f4")


def load_bf16_tensor(path: Path, name: str) -> np.ndarray:
    data = path.read_bytes()
    header_len = struct.unpack_from("<Q", data, 0)[0]
    header = json.loads(data[8 : 8 + header_len])
    info = header[name]
    start = 8 + header_len + info["data_offsets"][0]
    end = 8 + header_len + info["data_offsets"][1]
    shape = info["shape"]
    raw = data[start:end]
    return bf16_bytes_to_f32(raw).reshape(shape)


def load_bf16_matrix(path: Path, name: str) -> np.ndarray:
    return load_bf16_tensor(path, name)


def load_bf16_vec(path: Path, name: str) -> np.ndarray:
    return load_bf16_tensor(path, name).reshape(-1)


def rmsnorm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    # x: [..., d]
    var = (x * x).mean(axis=-1, keepdims=True)
    return w * x / np.sqrt(var + eps)


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a [m,k], b [k,n] row-major like zynfer
    return a @ b


def rope_thd(x: np.ndarray, pos0: int, theta: float) -> None:
    """In-place RoPE on [tokens, heads, d]."""
    t, h, d = x.shape
    half = d // 2
    for ti in range(t):
        pos = pos0 + ti
        for hi in range(h):
            for i in range(half):
                freq = 1.0 / (theta ** (i / half))
                angle = pos * freq
                c, s = np.cos(angle), np.sin(angle)
                a, b = x[ti, hi, i], x[ti, hi, i + half]
                x[ti, hi, i] = a * c - b * s
                x[ti, hi, i + half] = a * s + b * c


def attention_gqa(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """q [n_q,t,d], k/v [n_kv,t,d] full causal prefill."""
    n_q, t, d = q.shape
    n_kv = k.shape[0]
    group = n_q // n_kv
    scale = 1.0 / np.sqrt(d)
    out = np.zeros_like(q)
    for h in range(n_q):
        kv_h = h // group
        for tq in range(t):
            scores = np.full(t, -np.inf, dtype=np.float32)
            for tk in range(t):
                if (t - t + tq) >= tk:  # causal: (kv_len - q_len + tq) >= tk, kv_len=q_len=t
                    scores[tk] = np.dot(q[h, tq], k[kv_h, tk]) * scale
            m = np.max(scores)
            w = np.exp(scores - m)
            w /= w.sum()
            out[h, tq] = (w[:, None] * v[kv_h, :t]).sum(axis=0)
    return out


def write_f32(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(arr.astype("<f4", copy=False).tobytes())


def compare(name: str, a: np.ndarray, b: np.ndarray) -> None:
    diff = np.abs(a - b)
    idx = int(diff.argmax())
    print(
        f"{name}: max_abs={diff.max():.6g} rms={np.sqrt((diff * diff).mean()):.6g} "
        f"idx={idx} a={a.flat[idx]:.6g} b={b.flat[idx]:.6g}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path("models/Qwen3-0.6B/model.safetensors"))
    ap.add_argument("--tokens", default="151643,2,3")
    ap.add_argument("--out-dir", type=Path, default=Path("ref_numpy"))
    ap.add_argument("--compare-dir", type=Path, default=None, help="zynfer --dump dir")
    args = ap.parse_args()

    token_ids = [int(x.strip()) for x in args.tokens.split(",") if x.strip()]
    hidden, n_q, n_kv, d, inter = 1024, 16, 8, 128, 3072
    eps, theta = 1e-6, 1_000_000.0
    n_layers = 28

    embed = load_bf16_matrix(args.model, "model.embed_tokens.weight")  # [vocab, hidden]
    final_norm = load_bf16_vec(args.model, "model.norm.weight")

    x = embed[token_ids].astype(np.float32)  # [t, hidden]
    write_f32(args.out_dir / "embed_last.f32", x[-1])

    for layer in range(n_layers):
        prefix = f"model.layers.{layer}"
        w = {
            "in": load_bf16_vec(args.model, f"{prefix}.input_layernorm.weight"),
            "q_norm": load_bf16_vec(args.model, f"{prefix}.self_attn.q_norm.weight"),
            "k_norm": load_bf16_vec(args.model, f"{prefix}.self_attn.k_norm.weight"),
            "q": load_bf16_matrix(args.model, f"{prefix}.self_attn.q_proj.weight").T,
            "k": load_bf16_matrix(args.model, f"{prefix}.self_attn.k_proj.weight").T,
            "v": load_bf16_matrix(args.model, f"{prefix}.self_attn.v_proj.weight").T,
            "o": load_bf16_matrix(args.model, f"{prefix}.self_attn.o_proj.weight").T,
            "post": load_bf16_vec(args.model, f"{prefix}.post_attention_layernorm.weight"),
            "gate": load_bf16_matrix(args.model, f"{prefix}.mlp.gate_proj.weight").T,
            "up": load_bf16_matrix(args.model, f"{prefix}.mlp.up_proj.weight").T,
            "down": load_bf16_matrix(args.model, f"{prefix}.mlp.down_proj.weight").T,
        }

        t = x.shape[0]
        xn = rmsnorm(x, w["in"], eps)
        q_lin = matmul(xn, w["q"])
        k_lin = matmul(xn, w["k"])
        v_lin = matmul(xn, w["v"])
        q = q_lin.reshape(t, n_q, d)
        k = k_lin.reshape(t, n_kv, d)
        v = v_lin.reshape(t, n_kv, d)
        q = rmsnorm(q, w["q_norm"], eps)
        k = rmsnorm(k, w["k_norm"], eps)
        rope_thd(q, 0, theta)
        rope_thd(k, 0, theta)
        q_htd = np.transpose(q, (1, 0, 2))
        k_htd = np.transpose(k, (1, 0, 2))
        v_htd = np.transpose(v, (1, 0, 2))
        attn = attention_gqa(q_htd, k_htd, v_htd)
        attn_thd = np.transpose(attn, (1, 0, 2)).reshape(t, n_q * d)
        ao = matmul(attn_thd, w["o"])
        x1 = x + ao
        mlp_n = rmsnorm(x1, w["post"], eps)
        gate = matmul(mlp_n, w["gate"])
        up = matmul(mlp_n, w["up"])
        hid = silu(gate) * up
        down = matmul(hid, w["down"])
        x = x1 + down
        write_f32(args.out_dir / f"layer{layer:02d}.f32", x[-1])

    normed = rmsnorm(x[-1:], final_norm, eps)[0]
    write_f32(args.out_dir / "normed.f32", normed)
    logits = embed @ normed
    write_f32(args.out_dir / "logits.f32", logits)

    top = sorted(enumerate(logits), key=lambda z: -z[1])[:8]
    print(f"wrote {args.out_dir}/")
    print("top logits:")
    for tid, val in top:
        print(f"  id={tid} logit={val:.6f}")

    if args.compare_dir:
        for name in ["embed_last", "layer00", "layer01", "normed", "logits"]:
            p = args.compare_dir / f"{name}.f32"
            if not p.exists():
                continue
            n = p.stat().st_size // 4
            b = np.fromfile(p, dtype="<f4", count=n)
            a = np.fromfile(args.out_dir / f"{name}.f32", dtype="<f4")
            compare(name, a, b)


if __name__ == "__main__":
    main()
