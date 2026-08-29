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
import urllib.request
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
    # Colab/headless callers can explicitly provide the pitch without needing a
    # Windows Studio settings file.
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
                # Some minimal FFmpeg builds omit afftdn. Keep the important
                # anti-alias/high-frequency cleanup instead of failing setup.
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

    # The MOS checkpoint callback is removed below, so avoid loading/downloading
    # the optional UTMOS predictor as well.
    if not _has_any_arg("--model.mos_metric"):
        sys.argv.extend(["--model.mos_metric", "none"])

    pitch = _studio_pitch()
    if abs(pitch) < HIGH_PITCH_THRESHOLD:
        return

    _clean_high_pitch_dataset(pitch)
    _enable_high_pitch_warmstart(pitch)


def _configure_callbacks() -> None:
    callbacks = list(getattr(piper_train_main, "_DEFAULT_CALLBACKS", []))

    if os.environ.get("PIPER_COLAB", "").strip().lower() in {"1", "true", "yes", "on"}:
        # Google Drive is far slower than a local SSD, and Piper checkpoints are
        # large. Save fewer checkpoints, less often, while keeping a resumable
        # last.ckpt plus the best validation checkpoints.
        from lightning.pytorch.callbacks import ModelCheckpoint

        try:
            every_n_epochs = max(1, int(os.environ.get("PIPER_COLAB_CHECKPOINT_EVERY", "5")))
        except ValueError:
            every_n_epochs = 5

        piper_train_main._DEFAULT_CALLBACKS = [
            ModelCheckpoint(
                monitor="val_mel",
                mode="min",
                save_top_k=2,
                save_last=True,
                every_n_epochs=every_n_epochs,
                filename="epoch={epoch}-val_mel={val_mel:.4f}",
                auto_insert_metric_name=False,
            )
        ]
        print(
            "Piper Colab checkpoint mode: best 2 val_mel + last.ckpt, "
            f"saving every {every_n_epochs} epoch(s).",
            flush=True,
        )
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
