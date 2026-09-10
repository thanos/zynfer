#!/usr/bin/env python3
"""Build a tiny fixed-shape Core ML toy model for Stage M7 load smoke.

Dev-time only (same policy as checkpoint converters). Runtime never depends
on coremltools — it only loads the compiled .mlpackage via the ObjC bridge.

  ASDF_PYTHON_VERSION=3.12.9 python3 tools/fixtures/make_coreml_toy.py
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb

IN_DIM = 8
OUT_DIM = 4


def build_model() -> ct.models.MLModel:
    """y = x @ W + b  with fixed shapes [1, IN] -> [1, OUT]."""
    w = np.arange(IN_DIM * OUT_DIM, dtype=np.float32).reshape(IN_DIM, OUT_DIM) * 0.01
    b = np.linspace(-0.1, 0.1, OUT_DIM, dtype=np.float32)

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, IN_DIM))],
        opset_version=ct.target.iOS16,
    )
    def prog(x):
        ww = mb.const(val=w, name="W")
        bb = mb.const(val=b, name="B")
        xw = mb.matmul(x=x, y=ww, name="xw")
        return mb.add(x=xw, y=bb, name="y")

    return ct.convert(
        prog,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("tools/fixtures/coreml_toy.mlpackage"),
        help="Output .mlpackage path",
    )
    args = ap.parse_args()
    model = build_model()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        shutil.rmtree(args.out)
    model.save(str(args.out))

    x = np.ones((1, IN_DIM), dtype=np.float32)
    out = model.predict({"x": x})
    y = np.asarray(next(iter(out.values()))).reshape(-1)
    print(f"wrote {args.out}")
    print(f"  inputs:  x float32 [1,{IN_DIM}]")
    print(f"  outputs: y float32 [1,{OUT_DIM}]  sample={y.tolist()}")


if __name__ == "__main__":
    main()
