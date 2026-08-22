#!/usr/bin/env python3
"""Generate a golden last-token logits file for `zynfer forward-golden --golden`.

Development-time only. Requires PyTorch + transformers (not used in CI).

Writes a raw little-endian f32 blob: vocab_size × 4 bytes (last prompt token).

Example (must match zynfer token IDs exactly):

  pip install torch transformers
  python3 tools/fixtures/gen_golden_logits.py \\
    --model models/Qwen3-0.6B \\
    --tokens 151643,2,3 \\
    --out ref_logits.f32

  ./zig-out/bin/zynfer forward-golden models/qwen3-0.6b.zynfer \\
    --tokens=151643,2,3 --golden ref_logits.f32

Default zynfer tokens (when --tokens omitted): bos,2,3 → 151643,2,3 for Qwen3-0.6B.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path


def parse_tokens(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part, 10))
    if not out:
        raise SystemExit("empty --tokens")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        type=Path,
        default=Path("models/Qwen3-0.6B"),
        help="HF model directory (default: models/Qwen3-0.6B)",
    )
    ap.add_argument(
        "--tokens",
        default="151643,2,3",
        help="comma-separated token IDs (default: bos,2,3 for Qwen3-0.6B)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("ref_logits.f32"),
        help="output path (default: ref_logits.f32)",
    )
    args = ap.parse_args()

    if not args.model.is_dir():
        raise SystemExit(f"model directory not found: {args.model}")

    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as e:
        raise SystemExit(
            "requires: pip install torch transformers\n" f"import error: {e}"
        ) from e

    token_ids = parse_tokens(args.tokens)
    print(f"model:  {args.model}", file=sys.stderr)
    print(f"tokens: {token_ids}", file=sys.stderr)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        trust_remote_code=False,
        attn_implementation="eager",
    )
    model.eval()

    input_ids = torch.tensor([token_ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids).logits[0, -1, :].to(torch.float32).cpu().numpy()

    vocab = logits.shape[0]
    expected_bytes = vocab * 4
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(logits.astype("<f4", copy=False).tobytes())

    if args.out.stat().st_size != expected_bytes:
        raise SystemExit("write size mismatch")

    top = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)[:8]
    print(f"wrote {args.out} ({expected_bytes} bytes, vocab={vocab})", file=sys.stderr)
    print("top logits:", file=sys.stderr)
    for tid, val in top:
        print(f"  id={tid} logit={val:.6f}", file=sys.stderr)


if __name__ == "__main__":
    main()
