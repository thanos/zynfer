#!/usr/bin/env python3
"""One-shot local setup for registered Qwen3 models + zynfer artifacts.

Development-time only. Never run in CI (downloads multi-GiB weights).

  python3 tools/setup_qwen.py                 # Qwen3-0.6B (default)
  python3 tools/setup_qwen.py --model 4b      # Qwen3-4B (Stage M8)
  python3 tools/setup_qwen.py --model 4b --quantize
  python3 tools/setup_qwen.py --skip-golden

Steps:
  1. pip install huggingface_hub safetensors numpy [torch transformers]
  2. hf download / snapshot_download → models/Qwen3-*
  3. convert safetensors → models/qwen3-*.zynfer
  4. optional: int8 pack → models/qwen3-*-int8.zynfer
  5. optional golden (0.6B only by default)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONVERTER = ROOT / "tools" / "checkpoint" / "safetensors_to_zynfer.py"
QUANTIZE = ROOT / "tools" / "checkpoint" / "quantize_zynfer_int8.py"
GEN_GOLDEN = ROOT / "tools" / "fixtures" / "gen_golden_logits.py"

MODELS = {
    "0.6b": {
        "hf_repo": "Qwen/Qwen3-0.6B",
        "hf_dir": ROOT / "models" / "Qwen3-0.6B",
        "artifact": ROOT / "models" / "qwen3-0.6b.zynfer",
        "int8": ROOT / "models" / "qwen3-0.6b-int8.zynfer",
        "model_id": "qwen3-0.6b",
        "min_artifact_bytes": 1_000_000_000,
    },
    "4b": {
        "hf_repo": "Qwen/Qwen3-4B",
        "hf_dir": ROOT / "models" / "Qwen3-4B",
        "artifact": ROOT / "models" / "qwen3-4b.zynfer",
        "int8": ROOT / "models" / "qwen3-4b-int8.zynfer",
        "model_id": "qwen3-4b",
        "min_artifact_bytes": 4_000_000_000,
    },
}


def run(argv: list[str], *, check: bool = True) -> int:
    print("+", " ".join(argv), flush=True)
    completed = subprocess.run(argv, cwd=ROOT)
    if check and completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return completed.returncode


def pip_install(pkgs: list[str]) -> None:
    run([sys.executable, "-m", "pip", "install", "-U", *pkgs])


def ensure_hf_dir(spec: dict) -> None:
    hf_dir: Path = spec["hf_dir"]
    hf_dir.mkdir(parents=True, exist_ok=True)
    config = hf_dir / "config.json"
    if config.is_file() and (
        (hf_dir / "model.safetensors").is_file()
        or any(hf_dir.glob("model-*.safetensors"))
    ):
        print(f"hf weights already present: {hf_dir}", flush=True)
        return

    model_id = spec["hf_repo"]
    hf = shutil.which("hf")
    if hf:
        run([hf, "download", model_id, "--local-dir", str(hf_dir)])
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise SystemExit(
            "huggingface_hub missing; re-run without --skip-pip or: "
            "pip install -U 'huggingface_hub[cli]'\n"
            f"import error: {e}"
        ) from e
    print(f"snapshot_download({model_id!r}) → {hf_dir}", flush=True)
    snapshot_download(model_id, local_dir=str(hf_dir))


def convert_artifact(spec: dict) -> None:
    artifact: Path = spec["artifact"]
    if artifact.is_file() and artifact.stat().st_size > spec["min_artifact_bytes"]:
        print(f"artifact already present: {artifact}", flush=True)
        return
    hf_dir: Path = spec["hf_dir"]
    weights = hf_dir / "model.safetensors"
    weights_arg = str(weights if weights.is_file() else hf_dir)
    run(
        [
            sys.executable,
            str(CONVERTER),
            "--config",
            str(hf_dir / "config.json"),
            "--weights",
            weights_arg,
            "--out",
            str(artifact),
            "--model-id",
            spec["model_id"],
        ]
    )


def quantize_artifact(spec: dict) -> None:
    src: Path = spec["artifact"]
    dst: Path = spec["int8"]
    if dst.is_file() and dst.stat().st_size > src.stat().st_size // 4:
        print(f"int8 artifact already present: {dst}", flush=True)
        return
    if not src.is_file():
        raise SystemExit(f"missing bf16 artifact for quantize: {src}")
    run(
        [
            sys.executable,
            str(QUANTIZE),
            "--in",
            str(src),
            "--out",
            str(dst),
        ]
    )


def gen_golden(spec: dict) -> None:
    golden = ROOT / "ref_logits.f32"
    if golden.is_file() and golden.stat().st_size == 151936 * 4:
        print(f"golden already present: {golden}", flush=True)
        return
    run(
        [
            sys.executable,
            str(GEN_GOLDEN),
            "--model",
            str(spec["hf_dir"]),
            "--tokens=151643,2,3",
            "--out",
            str(golden),
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        choices=sorted(MODELS.keys()),
        default="0.6b",
        help="registered model size (default: 0.6b)",
    )
    ap.add_argument("--skip-pip", action="store_true", help="do not pip install deps")
    ap.add_argument("--skip-download", action="store_true", help="assume HF dir exists")
    ap.add_argument("--skip-convert", action="store_true", help="skip .zynfer convert")
    ap.add_argument("--quantize", action="store_true", help="also write *-int8.zynfer")
    ap.add_argument("--skip-golden", action="store_true", help="skip ref_logits.f32")
    ap.add_argument("--no-torch", action="store_true", help="never install torch")
    args = ap.parse_args()
    spec = MODELS[args.model]

    if not args.skip_pip:
        base = ["huggingface_hub[cli]", "safetensors", "numpy"]
        pip_install(base)
        if not args.skip_golden and not args.no_torch and args.model == "0.6b":
            pip_install(["torch", "transformers"])

    if not args.skip_download:
        ensure_hf_dir(spec)
    elif not (spec["hf_dir"] / "config.json").is_file():
        raise SystemExit(f"--skip-download but missing {spec['hf_dir']}/config.json")

    if not args.skip_convert:
        convert_artifact(spec)

    if args.quantize:
        quantize_artifact(spec)

    if not args.skip_golden and args.model == "0.6b":
        if args.no_torch:
            print("skipping golden (--no-torch)", flush=True)
        else:
            gen_golden(spec)
    elif not args.skip_golden and args.model != "0.6b":
        print("skipping golden (only wired for 0.6b)", flush=True)

    print("\nsetup complete:", flush=True)
    print(f"  model:     {spec['model_id']}", flush=True)
    print(f"  hf dir:    {spec['hf_dir']}", flush=True)
    print(f"  artifact:  {spec['artifact']}", flush=True)
    if spec["int8"].is_file():
        print(f"  int8:      {spec['int8']}", flush=True)
    art = spec["int8"] if spec["int8"].is_file() else spec["artifact"]
    print(
        f"\ntry:\n"
        f"  ZYNFER_QWEN_METAL=int8 ./zig-out/bin/zynfer chat {art} \"Explain gravity simply.\"\n"
        f"  ./zig-out/bin/zynfer stageM8\n",
        flush=True,
    )


if __name__ == "__main__":
    main()
