#!/usr/bin/env python3
"""Dump intermediate hidden states + logits for zynfer forward debugging.

  pip install torch transformers
  python3 tools/fixtures/dump_forward_refs.py --tokens 151643,2,3 --out-dir ref_forward

Produces:
  ref_forward/normed.f32      final hidden after model.norm [1024]
  ref_forward/logits.f32        last-token logits [vocab]
  ref_forward/layer00.f32       hidden after decoder layer 0 [1024] (last token)
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


def parse_tokens(text: str) -> list[int]:
    return [int(p.strip()) for p in text.split(",") if p.strip()]


def write_f32(path: Path, arr) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(arr.astype("<f4", copy=False).tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path("models/Qwen3-0.6B"))
    ap.add_argument("--tokens", default="151643,2,3")
    ap.add_argument("--out-dir", type=Path, default=Path("ref_forward"))
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    token_ids = parse_tokens(args.tokens)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()

    captured: dict[str, torch.Tensor] = {}

    def hook_layer0(_module, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        captured["layer00"] = hs[0, -1, :].detach().cpu().float()

    model.model.layers[0].register_forward_hook(hook_layer0)

    input_ids = torch.tensor([token_ids], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)

    hs = out.hidden_states[-1][0, -1, :].cpu().float()
    logits = out.logits[0, -1, :].cpu().float()

    write_f32(args.out_dir / "normed.f32", hs.numpy())
    write_f32(args.out_dir / "logits.f32", logits.numpy())
    if "layer00" in captured:
        write_f32(args.out_dir / "layer00.f32", captured["layer00"].numpy())

    top = logits.topk(8)
    print(f"wrote {args.out_dir}/")
    print("top logits:")
    for v, i in zip(top.values.tolist(), top.indices.tolist()):
        print(f"  id={i} logit={v:.6f}")


if __name__ == "__main__":
    main()
