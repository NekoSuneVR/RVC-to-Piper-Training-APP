from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


def run_stage(name: str, command: list[str], cwd: Path | None = None) -> None:
    print(f"\n=== {name} ===", flush=True)
    print("> " + subprocess.list2cmdline(command), flush=True)
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n", flush=True)
    if result.returncode:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="One-utterance Colab preflight for RVC -> Piper")
    p.add_argument("--repo-root", required=True)
    p.add_argument("--rvc-root", required=True)
    p.add_argument("--rvc-python", required=True)
    p.add_argument("--piper-python", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--base-config", required=True)
    p.add_argument("--rvc-model", required=True)
    p.add_argument("--rvc-index", default="")
    p.add_argument("--pitch", type=int, default=12)
    p.add_argument("--index-rate", type=float, default=0.75)
    p.add_argument("--protect", type=float, default=0.33)
    p.add_argument("--f0-method", choices=["rmvpe", "pm"], default="rmvpe")
    return p


def absolute_without_resolving_symlink(value: str) -> Path:
    """Make a path absolute without following uv venv interpreter symlinks."""

    return Path(os.path.abspath(os.path.expanduser(value)))


def main() -> int:
    args = parser().parse_args()
    repo = Path(args.repo_root).resolve()
    rvc_root = Path(args.rvc_root).resolve()

    # Important: uv's venv/bin/python is a symlink to its managed interpreter.
    # Path.resolve() would follow it and lose the virtual environment's
    # site-packages. Keep the venv path intact when launching child Python.
    rvc_python = absolute_without_resolving_symlink(args.rvc_python)
    piper_python = absolute_without_resolving_symlink(args.piper_python)

    base_model = Path(args.base_model).resolve()
    base_config = Path(args.base_config).resolve()
    rvc_model = Path(args.rvc_model).resolve()
    rvc_index = Path(args.rvc_index).resolve() if args.rvc_index else None

    checks = [
        (repo / "colab" / "colab_pipeline.py", "Colab pipeline"),
        (rvc_root / "infer" / "cli.py", "RVC CLI"),
        (rvc_python, "RVC Python"),
        (piper_python, "Piper Python"),
        (base_model, "Base Piper ONNX"),
        (base_config, "Base Piper JSON"),
        (rvc_model, "RVC model"),
    ]
    if rvc_index:
        checks.append((rvc_index, "RVC index"))

    print("=== Colab build preflight ===")
    for path, label in checks:
        print(f"{label}: {path}")
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    if args.index_rate > 0 and not rvc_index:
        print("No RVC index supplied; using index_rate=0 for preflight.")
        args.index_rate = 0.0

    with tempfile.TemporaryDirectory(prefix="rvc-piper-preflight-") as raw:
        temp = Path(raw)
        base_dir = temp / "base"
        rvc_out = temp / "rvc"
        base_dir.mkdir()
        rvc_out.mkdir()
        jobs = temp / "jobs.json"
        jobs.write_text(
            json.dumps([
                {
                    "file": "preflight.wav",
                    "text": "Hello. This is a short Piper and RVC Colab preflight test.",
                }
            ]),
            encoding="utf-8",
        )

        run_stage(
            "1/3 Base Piper synthesis",
            [
                str(piper_python),
                str(repo / "colab" / "colab_pipeline.py"),
                "synthesize-base",
                "--model", str(base_model),
                "--config", str(base_config),
                "--jobs-json", str(jobs),
                "--output-dir", str(base_dir),
            ],
            cwd=repo,
        )

        base_wav = base_dir / "preflight.wav"
        if not base_wav.is_file() or base_wav.stat().st_size < 1024:
            raise RuntimeError("Base Piper synthesis returned success but produced no usable WAV")
        print(f"Base WAV OK: {base_wav.stat().st_size / 1024:.1f} KiB")

        rvc_cmd = [
            str(rvc_python),
            "-m", "infer.cli",
            "--model", str(rvc_model),
            "--input", str(base_wav),
            "--output", str(rvc_out / "preflight.wav"),
            "--pitch", str(args.pitch),
            "--f0-method", args.f0_method,
            "--index-rate", str(args.index_rate),
            "--protect", str(args.protect),
            "--format", "wav",
            "--overwrite",
        ]
        if rvc_index and args.index_rate > 0:
            rvc_cmd += ["--index", str(rvc_index)]

        run_stage("2/3 RVC conversion", rvc_cmd, cwd=rvc_root)

        rvc_wav = rvc_out / "preflight.wav"
        if not rvc_wav.is_file() or rvc_wav.stat().st_size < 1024:
            raise RuntimeError("RVC returned success but produced no usable WAV")
        print(f"RVC WAV OK: {rvc_wav.stat().st_size / 1024:.1f} KiB")

        run_stage(
            "3/3 Runtime version check",
            [
                str(piper_python),
                "-c",
                (
                    "import sys,torch,numpy,scipy,piper; "
                    "print('python',sys.version.split()[0]); "
                    "print('executable',sys.executable); "
                    "print('piper',piper.__file__); "
                    "print('torch',torch.__version__,'cuda',torch.version.cuda); "
                    "print('gpu',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); "
                    "print('numpy',numpy.__version__,'scipy',scipy.__version__)"
                ),
            ],
            cwd=repo,
        )

    print("\nPREFLIGHT PASSED: base Piper + RVC + CUDA runtime are working.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
