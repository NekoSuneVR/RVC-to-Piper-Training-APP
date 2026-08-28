from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PiperTrainPlan:
    voice_name: str
    trainer_python: Path
    trainer_source: Path
    dataset_dir: Path
    output_dir: Path
    espeak_voice: str = "en-gb"
    sample_rate: int = 22050
    batch_size: int = 8
    max_epochs: int = 1000
    accelerator: str = "auto"
    checkpoint: Path | None = None

    @property
    def audio_dir(self) -> Path:
        return self.dataset_dir / "audio"

    @property
    def metadata_csv(self) -> Path:
        return self.dataset_dir / "metadata.csv"

    @property
    def cache_dir(self) -> Path:
        return self.output_dir / "cache"

    @property
    def config_path(self) -> Path:
        return self.output_dir / f"{self.voice_name}.onnx.json"

    @property
    def lightning_dir(self) -> Path:
        return self.output_dir / "training"

    @property
    def onnx_path(self) -> Path:
        return self.output_dir / f"{self.voice_name}.onnx"


def safe_voice_name(value: str) -> str:
    value = value.strip().replace(" ", "-")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "", value)
    value = re.sub(r"[-_.]{2,}", "-", value).strip("-_.")
    if not value:
        raise ValueError("Voice name must contain at least one letter or number.")
    return value


def load_prompts(path: Path, limit: int | None = None) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Training prompt file not found: {path}")
    prompts: list[str] = []
    seen: set[str] = set()
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        text = " ".join(raw.strip().split())
        if not text or text.startswith("#") or text in seen:
            continue
        if "|" in text:
            raise ValueError("Training prompts cannot contain the | character.")
        seen.add(text)
        prompts.append(text)
        if limit and len(prompts) >= limit:
            break
    if not prompts:
        raise ValueError("The training prompt file does not contain any usable sentences.")
    return prompts


def read_metadata(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="|"):
            if len(row) >= 2:
                rows[row[0]] = row[-1]
    return rows


def write_metadata(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="|", lineterminator="\n")
        writer.writerows(rows)
    temp.replace(path)


def training_command(plan: PiperTrainPlan) -> list[str]:
    command = [
        str(plan.trainer_python),
        "-m", "piper.train", "fit",
        "--data.voice_name", plan.voice_name,
        "--data.csv_path", str(plan.metadata_csv),
        "--data.audio_dir", str(plan.audio_dir),
        "--model.sample_rate", str(plan.sample_rate),
        "--data.espeak_voice", plan.espeak_voice,
        "--data.cache_dir", str(plan.cache_dir),
        "--data.config_path", str(plan.config_path),
        "--data.batch_size", str(plan.batch_size),
        "--data.num_workers", "0",
        "--trainer.max_epochs", str(plan.max_epochs),
        "--trainer.accelerator", plan.accelerator,
        "--trainer.devices", "1",
        "--trainer.default_root_dir", str(plan.lightning_dir),
    ]
    if plan.checkpoint:
        command += ["--ckpt_path", str(plan.checkpoint)]
    return command


def export_command(plan: PiperTrainPlan, checkpoint: Path) -> list[str]:
    return [
        str(plan.trainer_python),
        "-m", "piper.train.export_onnx",
        "--checkpoint", str(checkpoint),
        "--output-file", str(plan.onnx_path),
    ]


def latest_checkpoint(root: Path) -> Path | None:
    candidates = [p for p in root.rglob("*.ckpt") if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime_ns)


def validate_trainer(plan: PiperTrainPlan) -> list[str]:
    errors: list[str] = []
    if not plan.trainer_python.is_file():
        errors.append(f"Piper trainer Python not found: {plan.trainer_python}")
    if not (plan.trainer_source / "src" / "piper" / "train").is_dir():
        errors.append(f"Piper training source not found: {plan.trainer_source}")
    if not plan.metadata_csv.is_file():
        errors.append(f"Dataset metadata not found: {plan.metadata_csv}")
    if not plan.audio_dir.is_dir():
        errors.append(f"Dataset audio folder not found: {plan.audio_dir}")
    if plan.checkpoint and not plan.checkpoint.is_file():
        errors.append(f"Warm-start checkpoint not found: {plan.checkpoint}")
    return errors
