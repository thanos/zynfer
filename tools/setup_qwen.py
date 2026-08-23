#!/usr/bin/env python3
"""One-shot local setup for Qwen3-0.6B + zynfer artifact (+ optional golden).

Development-time only. Never run in CI (downloads ~1.5 GiB weights).

  python3 tools/setup_qwen.py
  python3 tools/setup_qwen.py --skip-golden
  python3 tools/setup_qwen.py --skip-pip

Steps:
  1. pip install huggingface_hub safetensors numpy [torch transformers]
  2. hf download / snapshot_download → models/Qwen3-0.6B
  3. convert safetensors → models/qwen3-0.6b.zynfer
  4. optional: gen_golden_logits.py → ref_logits.f32
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3-0.6B"
HF_DIR = ROOT / "models" / "Qwen3-0.6B"
ARTIFACT = ROOT / "models" / "qwen3-0.6b.zynfer"
GOLDEN = ROOT / "ref_logits.f32"
CONVERTER = ROOT / "tools" / "checkpoint" / "safetensors_to_zynfer.py"
GEN_GOLDEN = ROOT / "tools" / "fixtures" / "gen_golden_logits.py"


def run(argv: list[str], *, check: bool = True) -> int:
    print("+", " ".join(argv), flush=True)
    completed = subprocess.run(argv, cwd=ROOT)
    if check and completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return completed.returncode


def pip_install(pkgs: list[str]) -> None:
    run([sys.executable, "-m", "pip", "install", "-U", *pkgs])


def ensure_hf_dir() -> None:
    HF_DIR.mkdir(parents=True, exist_ok=True)
    config = HF_DIR / "config.json"
    if config.is_file() and (
        (HF_DIR / "model.safetensors").is_file()
        or any(HF_DIR.glob("model-*.safetensors"))
    ):
        print(f"hf weights already present: {HF_DIR}", flush=True)
        return

    hf = shutil.which("hf")
    if hf:
        run([hf, "download", MODEL_ID, "--local-dir", str(HF_DIR)])
        return

    # Fallback without CLI entrypoint.
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise SystemExit(
            "huggingface_hub missing; re-run without --skip-pip or: "
            "pip install -U 'huggingface_hub[cli]'\n"
            f"import error: {e}"
        ) from e
    print(f"snapshot_download({MODEL_ID!r}) → {HF_DIR}", flush=True)
    snapshot_download(MODEL_ID, local_dir=str(HF_DIR))


def convert_artifact() -> None:
    if ARTIFACT.is_file() and ARTIFACT.stat().st_size > 1_000_000_000:
        print(f"artifact already present: {ARTIFACT}", flush=True)
        return
    weights = HF_DIR / "model.safetensors"
    weights_arg = str(weights if weights.is_file() else HF_DIR)
    run(
        [
            sys.executable,
            str(CONVERTER),
            "--config",
            str(HF_DIR / "config.json"),
            "--weights",
            weights_arg,
            "--out",
            str(ARTIFACT),
        ]
    )


def gen_golden() -> None:
    if GOLDEN.is_file() and GOLDEN.stat().st_size == 151936 * 4:
        print(f"golden already present: {GOLDEN}", flush=True)
        return
    run(
        [
            sys.executable,
            str(GEN_GOLDEN),
            "--model",
            str(HF_DIR),
            "--tokens=151643,2,3",
            "--out",
            str(GOLDEN),
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-pip", action="store_true", help="do not pip install deps")
    ap.add_argument("--skip-download", action="store_true", help="assume HF dir exists")
    ap.add_argument("--skip-convert", action="store_true", help="skip .zynfer convert")
    ap.add_argument("--skip-golden", action="store_true", help="skip ref_logits.f32")
    ap.add_argument(
        "--with-torch",
        action="store_true",
        default=True,
        help="install torch+transformers for golden (default on unless --skip-golden)",
    )
    ap.add_argument("--no-torch", action="store_true", help="never install torch")
    args = ap.parse_args()

    if not args.skip_pip:
        base = ["huggingface_hub[cli]", "safetensors", "numpy"]
        pip_install(base)
        if not args.skip_golden and not args.no_torch:
            pip_install(["torch", "transformers"])

    if not args.skip_download:
        ensure_hf_dir()
    elif not (HF_DIR / "config.json").is_file():
        raise SystemExit(f"--skip-download but missing {HF_DIR}/config.json")

    if not args.skip_convert:
        convert_artifact()

    if not args.skip_golden:
        if args.no_torch:
            print("skipping golden (--no-torch)", flush=True)
        else:
            gen_golden()

    print("\nsetup complete:", flush=True)
    print(f"  hf dir:    {HF_DIR}", flush=True)
    print(f"  artifact:  {ARTIFACT}", flush=True)
    if GOLDEN.is_file():
        print(f"  golden:    {GOLDEN}", flush=True)
    print(
        "\ntry:\n"
        "  ./zig-out/bin/zynfer chat \"Explain gravity simply.\"\n"
        "  ./zig-out/bin/zynfer forward-golden models/qwen3-0.6b.zynfer "
        "--tokens=151643,2,3 --golden ref_logits.f32\n",
        flush=True,
    )


if __name__ == "__main__":
    main()
