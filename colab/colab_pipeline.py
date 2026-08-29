from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import wave
from pathlib import Path

VAL_MEL_RE = re.compile(r"val_mel=([0-9]+(?:\.[0-9]+)?)")
DEFAULT_WARMSTART_URL = (
    "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/"
    "en/en_US/lessac/medium/epoch%3D2164-step%3D1355540.ckpt?download=true"
)


def log(message: str) -> None:
    print(message, flush=True)


def run(command: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    log("> " + subprocess.list2cmdline(command))
    subprocess.run(command, cwd=cwd, env=env, check=True)


def load_prompts(path: Path, limit: int | None) -> list[str]:
    prompts: list[str] = []
    seen: set[str] = set()
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        text = " ".join(raw.strip().split())
        if not text or text.startswith("#") or text in seen:
            continue
        if "|" in text:
            raise ValueError("Prompt lines cannot contain the | character.")
        seen.add(text)
        prompts.append(text)
        if limit and len(prompts) >= limit:
            break
    if not prompts:
        raise ValueError(f"No usable prompts found in {path}")
    return prompts


def read_metadata(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="|"):
            if len(row) >= 2:
                result[row[0]] = row[-1]
    return result


def write_metadata(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="|", lineterminator="\n")
        writer.writerows(rows)
    temp.replace(path)


def download(url: str, destination: Path, minimum_size: int = 1) -> Path:
    if destination.is_file() and destination.stat().st_size >= minimum_size:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + ".part")
    part.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "RVC-Piper-Colab/1.0"})
    log(f"Downloading {destination.name} ...")
    try:
        with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as handle:
            total = int(response.headers.get("Content-Length") or 0)
            copied = 0
            next_report = 64 * 1024 * 1024
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                copied += len(chunk)
                if copied >= next_report:
                    if total:
                        log(f"  {copied / 1024**2:.0f} / {total / 1024**2:.0f} MB")
                    else:
                        log(f"  {copied / 1024**2:.0f} MB")
                    next_report += 64 * 1024 * 1024
    except Exception:
        part.unlink(missing_ok=True)
        raise
    if not part.is_file() or part.stat().st_size < minimum_size:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"Download was incomplete: {destination}")
    part.replace(destination)
    return destination


def newest_last_checkpoint(root: Path) -> Path | None:
    candidates = [p for p in root.rglob("last.ckpt") if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime_ns)


def best_checkpoint_newest_run(root: Path) -> Path | None:
    candidates = [p for p in root.rglob("*.ckpt") if p.is_file()]
    if not candidates:
        return None

    runs: dict[Path, list[Path]] = {}
    for path in candidates:
        runs.setdefault(path.parent, []).append(path)
    newest_run = max(
        runs.values(),
        key=lambda paths: max(path.stat().st_mtime_ns for path in paths),
    )

    scored: list[tuple[float, int, Path]] = []
    for path in newest_run:
        match = VAL_MEL_RE.search(path.name)
        if match:
            scored.append((float(match.group(1)), -path.stat().st_mtime_ns, path))
    if scored:
        return min(scored, key=lambda item: (item[0], item[1]))[2]
    return max(newest_run, key=lambda p: p.stat().st_mtime_ns)


def synthesize_base(args: argparse.Namespace) -> int:
    from piper.voice import PiperVoice

    jobs = json.loads(Path(args.jobs_json).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    voice = PiperVoice.load(args.model, args.config)

    log(f"Loaded base Piper voice once; synthesizing {len(jobs)} missing prompt(s)...")
    for i, job in enumerate(jobs, start=1):
        output = output_dir / job["file"]
        with wave.open(str(output), "wb") as wav_file:
            voice.synthesize_wav(job["text"], wav_file)
        if i == 1 or i == len(jobs) or i % 10 == 0:
            log(f"  base {i}/{len(jobs)}: {output.name}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Google Colab RVC -> Piper headless builder")
    subparsers = parser.add_subparsers(dest="subcommand")

    synth = subparsers.add_parser("synthesize-base", help=argparse.SUPPRESS)
    synth.add_argument("--model", required=True)
    synth.add_argument("--config", required=True)
    synth.add_argument("--jobs-json", required=True)
    synth.add_argument("--output-dir", required=True)

    parser.add_argument("--repo-root")
    parser.add_argument("--rvc-root")
    parser.add_argument("--rvc-python")
    parser.add_argument("--piper-python")
    parser.add_argument("--drive-root")
    parser.add_argument("--voice-name")
    parser.add_argument("--rvc-model")
    parser.add_argument("--rvc-index", default="")
    parser.add_argument("--prompts")
    parser.add_argument("--prompt-limit", type=int, default=120)
    parser.add_argument("--pitch", type=int, default=12)
    parser.add_argument("--index-rate", type=float, default=0.75)
    parser.add_argument("--protect", type=float, default=0.33)
    parser.add_argument("--f0-method", choices=["rmvpe", "pm"], default="rmvpe")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--espeak-voice", default="en-GB-x-rp")
    parser.add_argument("--base-model", required=False)
    parser.add_argument("--base-config", required=False)
    parser.add_argument("--warmstart-url", default=DEFAULT_WARMSTART_URL)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.subcommand == "synthesize-base":
        return synthesize_base(args)

    required = (
        "repo_root",
        "rvc_root",
        "rvc_python",
        "piper_python",
        "drive_root",
        "voice_name",
        "rvc_model",
        "prompts",
        "base_model",
        "base_config",
    )
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        parser.error("Missing required arguments: " + ", ".join("--" + name.replace("_", "-") for name in missing))

    repo_root = Path(args.repo_root).resolve()
    rvc_root = Path(args.rvc_root).resolve()
    rvc_python = Path(args.rvc_python).resolve()
    piper_python = Path(args.piper_python).resolve()
    drive_root = Path(args.drive_root).resolve()
    rvc_model = Path(args.rvc_model).resolve()
    rvc_index = Path(args.rvc_index).resolve() if args.rvc_index else None
    prompts_path = Path(args.prompts).resolve()
    base_model = Path(args.base_model).resolve()
    base_config = Path(args.base_config).resolve()

    for path, label in (
        (repo_root / "piper_train_wrapper.py", "Studio training wrapper"),
        (rvc_root / "infer" / "cli.py", "RVC CLI"),
        (rvc_python, "RVC Python"),
        (piper_python, "Piper Python"),
        (rvc_model, "RVC model"),
        (prompts_path, "Prompt file"),
        (base_model, "Base Piper ONNX"),
        (base_config, "Base Piper config"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    if rvc_index and not rvc_index.is_file():
        raise FileNotFoundError(f"RVC index not found: {rvc_index}")
    if not 0 <= args.index_rate <= 1:
        raise ValueError("--index-rate must be between 0 and 1")
    if not 0 <= args.protect <= 0.5:
        raise ValueError("--protect must be between 0 and 0.5")

    prompts = load_prompts(prompts_path, args.prompt_limit or None)
    project = drive_root / args.voice_name
    dataset_dir = project / "dataset"
    audio_dir = dataset_dir / "audio"
    metadata_csv = dataset_dir / "metadata.csv"
    piper_out = project / "piper"
    training_root = piper_out / "training"
    local_root = Path("/content/rvc-piper-colab") / args.voice_name
    base_dir = local_root / "base"
    cache_dir = local_root / "cache"
    jobs_json = local_root / "jobs.json"

    audio_dir.mkdir(parents=True, exist_ok=True)
    piper_out.mkdir(parents=True, exist_ok=True)
    local_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_dataset:
        existing = read_metadata(metadata_csv)
        missing_jobs: list[dict[str, str]] = []
        rows: list[tuple[str, str]] = []
        for i, text in enumerate(prompts, start=1):
            name = f"utt-{i:05d}.wav"
            final = audio_dir / name
            rows.append((name, text))
            if not (final.is_file() and final.stat().st_size > 0 and existing.get(name) == text):
                missing_jobs.append({"file": name, "text": text})

        if missing_jobs:
            if base_dir.exists():
                shutil.rmtree(base_dir)
            base_dir.mkdir(parents=True)
            jobs_json.write_text(json.dumps(missing_jobs, ensure_ascii=False, indent=2), encoding="utf-8")
            run(
                [
                    str(piper_python),
                    str(Path(__file__).resolve()),
                    "synthesize-base",
                    "--model", str(base_model),
                    "--config", str(base_config),
                    "--jobs-json", str(jobs_json),
                    "--output-dir", str(base_dir),
                ],
                cwd=repo_root,
            )

            rvc_cmd = [
                str(rvc_python),
                "-m", "infer.cli",
                "--model", str(rvc_model),
                "--input", str(base_dir),
                "--output", str(audio_dir),
                "--pitch", str(args.pitch),
                "--f0-method", args.f0_method,
                "--index-rate", str(args.index_rate),
                "--protect", str(args.protect),
                "--format", "wav",
                "--overwrite",
            ]
            if rvc_index and args.index_rate > 0:
                rvc_cmd += ["--index", str(rvc_index)]
            run(rvc_cmd, cwd=rvc_root)
            log(f"RVC dataset generation complete: {len(missing_jobs)} new WAV(s).")
        else:
            log(f"Dataset already complete: {len(prompts)} WAV(s) reused from Google Drive.")
        write_metadata(metadata_csv, rows)
    else:
        if not metadata_csv.is_file() or not audio_dir.is_dir():
            raise RuntimeError("--skip-dataset was selected, but the Drive dataset is missing.")
        log("Using existing Google Drive dataset without RVC generation.")

    env = os.environ.copy()
    env["RVC_PIPER_PITCH"] = str(args.pitch)
    env["PIPER_COLAB"] = "1"
    env["PIPER_COLAB_CHECKPOINT_EVERY"] = str(max(1, args.checkpoint_every))
    env["PYTHONUNBUFFERED"] = "1"

    shared_ckpt = drive_root / "_checkpoints" / "en_US-lessac-medium.ckpt"
    resume_ckpt = None if args.no_resume else newest_last_checkpoint(training_root)
    if resume_ckpt:
        log(f"Resuming Colab training from: {resume_ckpt}")
    else:
        download(args.warmstart_url, shared_ckpt, minimum_size=100_000_000)
        log(f"Starting a fresh fine-tune from warm-start: {shared_ckpt}")

    train_cmd = [
        str(piper_python),
        str(repo_root / "piper_train_wrapper.py"),
        "fit",
        "--data.voice_name", args.voice_name,
        "--data.csv_path", str(metadata_csv),
        "--data.audio_dir", str(audio_dir),
        "--model.sample_rate", "22050",
        "--data.espeak_voice", args.espeak_voice,
        "--data.cache_dir", str(cache_dir),
        "--data.config_path", str(piper_out / f"{args.voice_name}.onnx.json"),
        "--data.batch_size", str(args.batch_size),
        "--data.num_workers", str(args.num_workers),
        "--trainer.max_epochs", str(args.max_epochs),
        "--trainer.accelerator", "gpu",
        "--trainer.devices", "1",
        "--trainer.default_root_dir", str(training_root),
        "--model.mos_metric", "none",
    ]
    if resume_ckpt:
        train_cmd += ["--ckpt_path", str(resume_ckpt)]
    else:
        train_cmd += ["--model.warmstart_ckpt", str(shared_ckpt)]

    log("")
    log("=== Piper training on Colab GPU ===")
    run(train_cmd, cwd=repo_root, env=env)

    checkpoint = best_checkpoint_newest_run(training_root)
    if checkpoint is None:
        raise RuntimeError(f"Training finished but no checkpoint was found under {training_root}")
    log(f"Exporting best checkpoint from newest run: {checkpoint}")

    onnx_path = piper_out / f"{args.voice_name}.onnx"
    run(
        [
            str(piper_python),
            "-m", "piper.train.export_onnx",
            "--checkpoint", str(checkpoint),
            "--output-file", str(onnx_path),
        ],
        cwd=repo_root,
        env=env,
    )

    config_path = piper_out / f"{args.voice_name}.onnx.json"
    if not onnx_path.is_file() or onnx_path.stat().st_size == 0:
        raise RuntimeError(f"ONNX export failed: {onnx_path}")
    if not config_path.is_file():
        raise RuntimeError(f"Piper config is missing: {config_path}")

    log("")
    log("=== DONE ===")
    log(f"ONNX:   {onnx_path}")
    log(f"Config: {config_path}")
    log(f"Drive project: {project}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
