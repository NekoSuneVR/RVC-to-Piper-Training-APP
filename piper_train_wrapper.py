from __future__ import annotations

"""Run Piper training with Studio/Colab compatibility and high-pitch safeguards.

For RVC pitch shifts of 10 semitones or more, the generated WAV dataset is
mastered to Piper's native 22.05 kHz format before librosa caches it. The
profile removes rumble, suppresses broadband hiss, rolls off unusable top-end
energy, and leaves headroom.

High-pitch builds warm-start from an official Piper medium checkpoint when no
resume/warm-start argument is already present. The optional UTMOS checkpoint is
disabled because it is not required for training/export.

Google Colab can opt into Drive-friendly checkpointing with PIPER_COLAB=1.
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Older official Piper checkpoints may contain pathlib.PosixPath metadata.
# Let torch.load unpickle those checkpoints on Windows.
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

from piper.train import __main__ as piper_train_main


APP_DIR = Path(__file__).resolve().parent
SETTINGS_FILE = APP_DIR / "data" / "settings.json"
HIGH_PITCH_THRESHOLD = 10
CLEAN_PROFILE_VERSION = "piper-high-pitch-clean-v1"
CLEAN_FILTER = (
    "highpass=f=70,"
    "lowpass=f=9000,"
    "afftdn=nr=10:nf=-35,"
    "aresample=22050,"
    "volume=0.95"
)
FALLBACK_CLEAN_FILTER = (
    "highpass=f=70,"
    "lowpass=f=9000,"
    "aresample=22050,"
    "volume=0.95"
)

WARMSTART_DIR = APP_DIR / "models" / "piper" / "checkpoints"
WARMSTART_CHECKPOINT = WARMSTART_DIR / "en_US-lessac-medium.ckpt"
WARMSTART_URL = (
    "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/"
    "en/en_US/lessac/medium/epoch%3D2164-step%3D1355540.ckpt?download=true"
)


def _arg_value(name: str) -> str | None:
    try:
        index = sys.argv.index(name)
    except ValueError:
        return None
    if index + 1 >= len(sys.argv):
        return None
    return sys.argv[index + 1]


def _has_any_arg(*names: str) -> bool:
    return any(name in sys.argv for name in names)


def _studio_pitch() -> int:
    env_pitch = os.environ.get("RVC_PIPER_PITCH")
    if env_pitch not in (None, ""):
        try:
            return int(env_pitch)
        except ValueError:
            pass

    try:
        raw = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        return int(raw.get("pitch", 0))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return 0


def _find_ffmpeg() -> Path | None:
    bundled = APP_DIR / "tools" / "rvc" / "ffmpeg.exe"
    if bundled.is_file():
        return bundled
    found = shutil.which("ffmpeg")
    return Path(found) if found else None


def _dataset_needs_cleaning(audio_dir: Path, marker: Path, pitch: int) -> bool:
    wavs = [path for path in audio_dir.glob("*.wav") if path.is_file()]
    if not wavs:
        return False
    if not marker.is_file():
        return True
    try:
        info = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return True
    if info.get("profile") != CLEAN_PROFILE_VERSION or int(info.get("pitch", 9999)) != pitch:
        return True
    if int(info.get("files", -1)) != len(wavs):
        return True
    try:
        marker_time = marker.stat().st_mtime_ns
        return any(path.stat().st_mtime_ns > marker_time for path in wavs)
    except OSError:
        return True


def _clean_high_pitch_dataset(pitch: int) -> None:
    audio_raw = _arg_value("--data.audio_dir")
    cache_raw = _arg_value("--data.cache_dir")
    if not audio_raw:
        return

    audio_dir = Path(audio_raw)
    if not audio_dir.is_dir():
        return

    marker = audio_dir.parent / ".piper-audio-master.json"
    if not _dataset_needs_cleaning(audio_dir, marker, pitch):
        print(
            f"Piper Studio {pitch:+d} audio mastering: current; reusing cleaned dataset.",
            flush=True,
        )
        return

    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError(
            "High-pitch Piper training needs FFmpeg for dataset cleanup, but "
            "no bundled/system ffmpeg executable was found."
        )

    wavs = sorted(path for path in audio_dir.glob("*.wav") if path.is_file())
    if not wavs:
        return

    print(
        f"Piper Studio high-pitch mode: keeping RVC pitch {pitch:+d} and mastering "
        f"{len(wavs)} WAV files for clean 22.05 kHz Piper training...",
        flush=True,
    )
    print(
        "Audio cleanup: 70 Hz high-pass, RVC hiss reduction, 9 kHz low-pass, "
        "mono 22050 Hz, 5% headroom.",
        flush=True,
    )

    for number, wav_path in enumerate(wavs, start=1):
        temp_path = wav_path.with_name(wav_path.stem + ".piper-clean.tmp.wav")

        def run_filter(filter_chain: str) -> subprocess.CompletedProcess[str]:
            temp_path.unlink(missing_ok=True)
            command = [
                str(ffmpeg),
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(wav_path),
                "-af",
                filter_chain,
                "-ar",
                "22050",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(temp_path),
            ]
            return subprocess.run(command, capture_output=True, text=True, check=False)

        try:
            result = run_filter(CLEAN_FILTER)
            if result.returncode != 0:
                result = run_filter(FALLBACK_CLEAN_FILTER)

            if result.returncode != 0 or not temp_path.is_file() or temp_path.stat().st_size == 0:
                detail = (result.stderr or result.stdout or "FFmpeg returned no diagnostic").strip()
                raise RuntimeError(f"Could not clean {wav_path.name}: {detail[-2000:]}")
            temp_path.replace(wav_path)
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

        if number == 1 or number == len(wavs) or (number % 10) == 0:
            print(f"  mastered {number}/{len(wavs)}: {wav_path.name}", flush=True)

    marker.write_text(
        json.dumps(
            {
                "profile": CLEAN_PROFILE_VERSION,
                "pitch": pitch,
                "files": len(wavs),
                "sample_rate": 22050,
                "lowpass_hz": 9000,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if cache_raw:
        cache_dir = Path(cache_raw)
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print(
                "Deleted Piper audio/spectrogram cache so training uses the newly "
                "mastered high-pitch dataset.",
                flush=True,
            )


def _download_warmstart() -> Path:
    if WARMSTART_CHECKPOINT.is_file() and WARMSTART_CHECKPOINT.stat().st_size > 100_000_000:
        return WARMSTART_CHECKPOINT

    WARMSTART_DIR.mkdir(parents=True, exist_ok=True)
    part = WARMSTART_CHECKPOINT.with_suffix(".ckpt.part")
    part.unlink(missing_ok=True)

    print(
        "High-pitch Piper build has no warm-start checkpoint. Downloading the "
        "official Piper medium Lessac checkpoint (~846 MB) once...",
        flush=True,
    )
    request = urllib.request.Request(
        WARMSTART_URL,
        headers={"User-Agent": "RVC-Piper-Studio/1.0"},
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response, part.open("wb") as handle:
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
                    if total > 0:
                        print(
                            f"  warm-start download: {copied / 1024**2:.0f} / "
                            f"{total / 1024**2:.0f} MB",
                            flush=True,
                        )
                    else:
                        print(
                            f"  warm-start download: {copied / 1024**2:.0f} MB",
                            flush=True,
                        )
                    next_report += 64 * 1024 * 1024
    except Exception:
        part.unlink(missing_ok=True)
        raise

    if not part.is_file() or part.stat().st_size < 100_000_000:
        part.unlink(missing_ok=True)
        raise RuntimeError("Warm-start checkpoint download was incomplete.")

    part.replace(WARMSTART_CHECKPOINT)
    print(f"Warm-start checkpoint ready: {WARMSTART_CHECKPOINT}", flush=True)
    return WARMSTART_CHECKPOINT


def _enable_high_pitch_warmstart(pitch: int) -> None:
    if _has_any_arg(
        "--ckpt_path",
        "--model.warmstart_ckpt",
        "--model.vocoder_warmstart_ckpt",
    ):
        return

    checkpoint = _download_warmstart()
    sys.argv.extend(["--model.warmstart_ckpt", str(checkpoint)])
    print(
        "Piper Studio high-pitch warm-start: loading matching pretrained VITS "
        "weights with a fresh optimizer (not resuming the old checkpoint epoch).",
        flush=True,
    )


def _prepare_training() -> None:
    if len(sys.argv) < 2 or sys.argv[1] != "fit":
        return

    if not _has_any_arg("--model.mos_metric"):
        sys.argv.extend(["--model.mos_metric", "none"])

    pitch = _studio_pitch()
    if abs(pitch) < HIGH_PITCH_THRESHOLD:
        return

    _clean_high_pitch_dataset(pitch)
    _enable_high_pitch_warmstart(pitch)


def _metric_text(trainer, name: str) -> str | None:
    value = trainer.callback_metrics.get(name)
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().item()
        return f"{float(value):.4f}"
    except (TypeError, ValueError, RuntimeError):
        return str(value)


class _ConsoleProgressCallback:
    """Created dynamically as a Lightning Callback-compatible object."""

    def __init__(self, callback_base, progress_file: Path, batch_every: int) -> None:
        self.__class__ = type(
            "PiperColabConsoleProgress",
            (callback_base, self.__class__),
            {},
        )
        self.progress_file = progress_file
        self.batch_every = max(1, batch_every)
        self.fit_started = time.monotonic()
        self.epoch_started = self.fit_started

    def _write_progress(self, trainer, status: str) -> None:
        try:
            self.progress_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "status": status,
                "epoch": int(trainer.current_epoch),
                "epoch_display": int(trainer.current_epoch) + 1,
                "max_epochs": int(trainer.max_epochs),
                "global_step": int(trainer.global_step),
                "device": str(getattr(trainer.strategy, "root_device", "unknown")),
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            }
            for metric in ("train_mel", "loss_g", "loss_d", "val_mel", "val_loss"):
                text = _metric_text(trainer, metric)
                if text is not None:
                    payload[metric] = text
            temp = self.progress_file.with_suffix(".json.tmp")
            temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            temp.replace(self.progress_file)
        except Exception as exc:
            print(f"[progress] Could not update {self.progress_file}: {exc}", flush=True)

    def on_fit_start(self, trainer, pl_module) -> None:
        self.fit_started = time.monotonic()
        self.epoch_started = self.fit_started
        device = getattr(trainer.strategy, "root_device", pl_module.device)
        print("", flush=True)
        print("================ PIPER TRAINING PROGRESS ================", flush=True)
        print(f"Device:       {device}", flush=True)
        print(f"Start epoch:  {int(trainer.current_epoch) + 1}/{trainer.max_epochs}", flush=True)
        print(f"Global step:  {trainer.global_step}", flush=True)
        print(f"Progress JSON:{self.progress_file}", flush=True)
        if int(trainer.current_epoch) > 0 or int(trainer.global_step) > 0:
            print("RESUME CONFIRMED: checkpoint state, optimizer and step counter were restored.", flush=True)
        else:
            print("Fresh training run: no previous training step was restored.", flush=True)
        print("=========================================================", flush=True)
        self._write_progress(trainer, "fit_started")

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self.epoch_started = time.monotonic()
        print(
            f"\n[EPOCH {int(trainer.current_epoch) + 1}/{trainer.max_epochs}] START "
            f"global_step={trainer.global_step} device={getattr(trainer.strategy, 'root_device', pl_module.device)}",
            flush=True,
        )
        self._write_progress(trainer, "training")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        try:
            total = int(trainer.num_training_batches)
        except (TypeError, ValueError, OverflowError):
            total = 0
        batch_no = int(batch_idx) + 1
        if batch_no != 1 and (batch_no % self.batch_every) != 0 and (not total or batch_no != total):
            return

        fields = [
            f"[train] epoch={int(trainer.current_epoch) + 1}/{trainer.max_epochs}",
            f"batch={batch_no}/{total if total else '?'}",
            f"step={trainer.global_step}",
        ]
        for metric in ("train_mel", "loss_g", "loss_d"):
            value = _metric_text(trainer, metric)
            if value is not None:
                fields.append(f"{metric}={value}")
        fields.append(f"epoch_time={time.monotonic() - self.epoch_started:.1f}s")
        print(" ".join(fields), flush=True)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if getattr(trainer, "sanity_checking", False):
            print("[validation] sanity check complete", flush=True)
            return
        fields = [
            f"[validation] epoch={int(trainer.current_epoch) + 1}/{trainer.max_epochs}",
            f"step={trainer.global_step}",
        ]
        for metric in ("val_mel", "val_loss", "val_kl", "val_dur"):
            value = _metric_text(trainer, metric)
            if value is not None:
                fields.append(f"{metric}={value}")
        print(" ".join(fields), flush=True)
        self._write_progress(trainer, "validated")

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        elapsed = time.monotonic() - self.epoch_started
        total = time.monotonic() - self.fit_started
        print(
            f"[EPOCH {int(trainer.current_epoch) + 1}/{trainer.max_epochs}] COMPLETE "
            f"global_step={trainer.global_step} epoch_time={elapsed:.1f}s total_time={total / 60:.1f}m",
            flush=True,
        )
        print("[resume] A fresh last.ckpt is scheduled for this epoch.", flush=True)
        self._write_progress(trainer, "epoch_complete")

    def on_fit_end(self, trainer, pl_module) -> None:
        print(
            f"[training] FIT COMPLETE at epoch={int(trainer.current_epoch) + 1} "
            f"global_step={trainer.global_step} elapsed={(time.monotonic() - self.fit_started) / 60:.1f}m",
            flush=True,
        )
        self._write_progress(trainer, "fit_complete")


def _configure_callbacks() -> None:
    callbacks = list(getattr(piper_train_main, "_DEFAULT_CALLBACKS", []))

    if os.environ.get("PIPER_COLAB", "").strip().lower() in {"1", "true", "yes", "on"}:
        from lightning.pytorch.callbacks import Callback, ModelCheckpoint

        try:
            best_every = max(1, int(os.environ.get("PIPER_COLAB_CHECKPOINT_EVERY", "5")))
        except ValueError:
            best_every = 5
        try:
            batch_every = max(1, int(os.environ.get("PIPER_CONSOLE_BATCH_EVERY", "2")))
        except ValueError:
            batch_every = 2

        root_raw = _arg_value("--trainer.default_root_dir")
        training_root = Path(root_raw) if root_raw else Path.cwd() / "training"
        checkpoint_dir = training_root / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        progress_file = checkpoint_dir / "training-progress.json"

        class VerboseModelCheckpoint(ModelCheckpoint):
            def _save_checkpoint(self, trainer, filepath: str) -> None:
                super()._save_checkpoint(trainer, filepath)
                path = Path(filepath)
                try:
                    size = f"{path.stat().st_size / 1024**2:.1f} MB"
                except OSError:
                    size = "size unknown"
                print(
                    f"[checkpoint] SAVED {path.name} ({size}) "
                    f"epoch={int(trainer.current_epoch) + 1} step={trainer.global_step}",
                    flush=True,
                )

        # Always create a real full-state checkpoint at each epoch. The fixed
        # filename is overwritten so repeated Colab sessions do not accumulate
        # an unlimited number of resume checkpoints. save_last=True also keeps
        # the deterministic last.ckpt path that the headless pipeline searches.
        resume_checkpoint = VerboseModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="resume-latest",
            save_top_k=-1,
            save_last=True,
            every_n_epochs=1,
            save_on_train_epoch_end=False,
            auto_insert_metric_name=False,
            enable_version_counter=False,
        )
        best_checkpoint = VerboseModelCheckpoint(
            dirpath=str(checkpoint_dir),
            monitor="val_mel",
            mode="min",
            save_top_k=2,
            save_last=False,
            every_n_epochs=best_every,
            filename="epoch={epoch}-val_mel={val_mel:.4f}",
            auto_insert_metric_name=False,
        )
        progress = _ConsoleProgressCallback(Callback, progress_file, batch_every)

        piper_train_main._DEFAULT_CALLBACKS = [
            progress,
            resume_checkpoint,
            best_checkpoint,
        ]
        print(
            "Piper Colab checkpoint mode: resume-latest.ckpt + last.ckpt EVERY epoch; "
            f"best 2 val_mel every {best_every} epoch(s).",
            flush=True,
        )
        print(
            "Full checkpoint state includes model weights, optimizer state, epoch and global step; "
            "it can resume on CPU after a GPU session ends.",
            flush=True,
        )
        print(
            f"Piper Colab console progress: every {batch_every} training batch(es).",
            flush=True,
        )
        print(f"Persistent checkpoint directory: {checkpoint_dir}", flush=True)
        return

    piper_train_main._DEFAULT_CALLBACKS = [
        callback for callback in callbacks if getattr(callback, "monitor", None) != "val_mos"
    ]
    print(
        "Piper Studio checkpoint mode: val_mel + last.ckpt "
        "(optional val_mos/UTMOS checkpoint disabled).",
        flush=True,
    )


def main() -> None:
    _prepare_training()
    _configure_callbacks()
    piper_train_main.main()


if __name__ == "__main__":
    main()
