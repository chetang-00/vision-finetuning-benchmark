#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a TensorRT engine with NVIDIA trtexec")
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--output", default="artifacts/model.engine")
    parser.add_argument("--precision", choices=["fp16", "int8", "fp32"], default="fp16")
    parser.add_argument("--max-batch", type=int, default=32)
    args = parser.parse_args()
    executable = shutil.which("trtexec")
    if not executable:
        raise SystemExit(
            "trtexec was not found. Run this script inside an NVIDIA TensorRT container."
        )
    onnx = Path(args.onnx).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    shape = "images:1x3x224x224"
    optimal = f"images:{min(8, args.max_batch)}x3x224x224"
    maximum = f"images:{args.max_batch}x3x224x224"
    command = [
        executable,
        f"--onnx={onnx}",
        f"--saveEngine={output}",
        f"--minShapes={shape}",
        f"--optShapes={optimal}",
        f"--maxShapes={maximum}",
    ]
    if args.precision != "fp32":
        command.append(f"--{args.precision}")
    subprocess.run(command, check=True)
    print(f"TensorRT engine written to {output}")


if __name__ == "__main__":
    main()
