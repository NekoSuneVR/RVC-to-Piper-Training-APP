from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from urllib.parse import quote

APP_ROOT = Path("/app")
RVC_ROOT = Path(os.environ.get("RVC_ROOT", "/opt/rvc"))
PIPER_ROOT = Path(os.environ.get("PIPER_ROOT", "/opt/piper1-gpl"))
RVC_PYTHON = Path(os.environ.get("RVC_PYTHON", "/opt/rvc-venv/bin/python"))
PIPER_PYTHON = Path(os.environ.get("PIPER_PYTHON", "/opt/piper-venv/bin/python"))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/data"))
WORK_ROOT = Path(os.environ.get("WORK_ROOT", str(DATA_ROOT / "work")))
CACHE_ROOT = Path(os.environ.get("CACHE_ROOT", str(DATA_ROOT / "cache")))
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", str(DATA_ROOT / "output")))

VOICE_ALIASES = {
    "alba": "en_GB-alba-medium",
    "amy": "en_US-amy-medium",
}


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def env_bool(name: str, default: bool = False) -> bool:
    raw = env(name, "1" if default else "0").lower()
    return raw in {"1", "true", "yes", "on"}


def data_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else DATA_ROOT / path


def run(command: list[str], *, cwd: Path | None = None, check: bool = True) -> int:
    print("> " + subprocess.list2cmdline(command), flush=True)
    result = subprocess.run(command, cwd=cwd, check=False)
    if check and result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    return result.returncode


def check_gpu() -> None:
    print("=== Docker GPU check ===", flush=True)
    run(["nvidia-smi"], check=True)
    for label, python in (("RVC", RVC_PYTHON), ("Piper", PIPER_PYTHON)):
        run(
            [
                str(python),
                "-c",
                (
                    "import torch; "
                    f"print('{label} torch:', torch.__version__); "
                    "print('CUDA runtime:', torch.version.cuda); "
                    "print('CUDA available:', torch.cuda.is_available()); "
                    "print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); "
                    "assert torch.cuda.is_available(), 'CUDA is not available inside the container'"
                ),
            ]
        )


def download(url: str, destination: Path) -> Path:
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + ".part")
    part.unlink(missing_ok=True)
    print(f"Downloading {destination.name} ...", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "RVC-Piper-Docker/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
    except Exception:
        part.unlink(missing_ok=True)
        raise
    if not part.is_file() or part.stat().st_size == 0:
        raise RuntimeError(f"Download returned an empty file: {destination}")
    part.replace(destination)
    return destination


def resolve_base_voice() -> tuple[Path, Path, str]:
    custom_model = env("BASE_PIPER_MODEL")
    custom_config = env("BASE_PIPER_CONFIG")
    if custom_model or custom_config:
        if not custom_model or not custom_config:
            raise RuntimeError(
                "Set both BASE_PIPER_MODEL and BASE_PIPER_CONFIG when using a custom base voice."
            )
        model = data_path(custom_model)
        config = data_path(custom_config)
        if not model.is_file():
            raise FileNotFoundError(f"Custom Piper ONNX not found: {model}")
        if not config.is_file():
            raise FileNotFoundError(f"Custom Piper JSON not found: {config}")
        return model, config, "custom"

    voice_key = env("BASE_PIPER_VOICE", "en_GB-alba-medium")
    voice_key = VOICE_ALIASES.get(voice_key.lower(), voice_key)
    manifest_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/voices.json"
    manifest_path = CACHE_ROOT / "piper-voices" / "voices.json"

    if not manifest_path.is_file():
        download(manifest_url, manifest_path)
    try:
        voices = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        manifest_path.unlink(missing_ok=True)
        download(manifest_url, manifest_path)
        voices = json.loads(manifest_path.read_text(encoding="utf-8"))

    voice = voices.get(voice_key)
    if voice is None:
        matches = [key for key in voices if voice_key.lower() in key.lower()][:20]
        raise KeyError(
            f"Piper voice key not found: {voice_key}. "
            f"Close matches: {', '.join(matches) if matches else 'none'}"
        )

    files = list(voice.get("files", {}).keys())
    onnx_rel = next((path for path in files if path.endswith(".onnx")), None)
    json_rel = next((path for path in files if path.endswith(".onnx.json")), None)
    if not onnx_rel or not json_rel:
        raise RuntimeError(f"Piper manifest has no ONNX/JSON pair for {voice_key}")

    voice_dir = CACHE_ROOT / "base-voices" / voice_key
    model = voice_dir / Path(onnx_rel).name
    config = voice_dir / Path(json_rel).name
    hf_root = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
    download(hf_root + quote(onnx_rel, safe="/._-") + "?download=true", model)
    download(hf_root + quote(json_rel, safe="/._-") + "?download=true", config)
    return model, config, voice_key


def build_pipeline_args() -> tuple[list[str], str]:
    voice_name = env("VOICE_NAME", "en_GB-rvc-custom-medium")
    rvc_model = data_path(env("RVC_MODEL", "models/voice.pth"))
    rvc_index_raw = env("RVC_INDEX")
    rvc_index = data_path(rvc_index_raw) if rvc_index_raw else None
    prompts = data_path(env("PROMPT_FILE")) if env("PROMPT_FILE") else APP_ROOT / "data" / "piper_training_prompts.txt"

    if not rvc_model.is_file():
        raise FileNotFoundError(
            f"RVC model not found: {rvc_model}. Put the .pth under the mounted /data/models folder."
        )
    if rvc_index and not rvc_index.is_file():
        raise FileNotFoundError(f"RVC index not found: {rvc_index}")
    if not prompts.is_file():
        raise FileNotFoundError(f"Prompt file not found: {prompts}")

    base_model, base_config, base_key = resolve_base_voice()
    index_rate = env("INDEX_RATE", "0.75")
    if not rvc_index and float(index_rate) > 0:
        print("No RVC .index selected; forcing INDEX_RATE=0.", flush=True)
        index_rate = "0"

    args = [
        "--repo-root", str(APP_ROOT),
        "--rvc-root", str(RVC_ROOT),
        "--rvc-python", str(RVC_PYTHON),
        "--piper-python", str(PIPER_PYTHON),
        "--drive-root", str(WORK_ROOT),
        "--voice-name", voice_name,
        "--rvc-model", str(rvc_model),
        "--prompts", str(prompts),
        "--prompt-limit", env("PROMPT_LIMIT", "120"),
        "--pitch", env("PITCH", "12"),
        "--index-rate", index_rate,
        "--protect", env("PROTECT", "0.33"),
        "--f0-method", env("F0_METHOD", "rmvpe"),
        "--batch-size", env("BATCH_SIZE", "8"),
        "--max-epochs", env("MAX_EPOCHS", "1000"),
        "--checkpoint-every", env("CHECKPOINT_EVERY", "5"),
        "--num-workers", env("NUM_WORKERS", "2"),
        "--espeak-voice", env("ESPEAK_VOICE", "en-GB-x-rp"),
        "--base-model", str(base_model),
        "--base-config", str(base_config),
    ]
    if rvc_index:
        args += ["--rvc-index", str(rvc_index)]
    if env_bool("SKIP_DATASET"):
        args.append("--skip-dataset")
    if env_bool("NO_RESUME"):
        args.append("--no-resume")

    print("=== Docker build settings ===", flush=True)
    print(f"Voice:       {voice_name}", flush=True)
    print(f"RVC model:   {rvc_model}", flush=True)
    print(f"RVC index:   {rvc_index or 'none'}", flush=True)
    print(f"Base Piper:  {base_key}", flush=True)
    print(f"Pitch:       {env('PITCH', '12')}", flush=True)
    print(f"Work root:   {WORK_ROOT}", flush=True)
    print(f"Output root: {OUTPUT_ROOT}", flush=True)
    return args, voice_name


def run_preflight(args: list[str]) -> None:
    wanted = {
        "--repo-root",
        "--rvc-root",
        "--rvc-python",
        "--piper-python",
        "--base-model",
        "--base-config",
        "--rvc-model",
        "--rvc-index",
        "--pitch",
        "--index-rate",
        "--protect",
        "--f0-method",
    }
    preflight_args: list[str] = []
    index = 0
    while index < len(args):
        key = args[index]
        if key in wanted and index + 1 < len(args):
            preflight_args.extend([key, args[index + 1]])
            index += 2
        else:
            index += 1
    run([sys.executable, str(APP_ROOT / "colab" / "preflight.py"), *preflight_args])


def copy_final_model(voice_name: str) -> None:
    source = WORK_ROOT / voice_name / "piper"
    target = OUTPUT_ROOT / voice_name
    target.mkdir(parents=True, exist_ok=True)
    for suffix in (".onnx", ".onnx.json"):
        file = source / f"{voice_name}{suffix}"
        if not file.is_file():
            raise FileNotFoundError(f"Expected final model file is missing: {file}")
        shutil.copy2(file, target / file.name)
    print(f"Final Docker output copied to: {target}", flush=True)


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else "build"
    if command == "shell":
        return subprocess.call(["/bin/bash"])

    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    os.environ.setdefault("RVC_PIPER_WORK_ROOT", "/tmp/rvc-piper-work")
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    check_gpu()
    args, voice_name = build_pipeline_args()

    if command in {"preflight", "build"} and env_bool("RUN_PREFLIGHT", True):
        run_preflight(args)
        if command == "preflight":
            return 0

    if command != "build":
        raise SystemExit("Usage: docker/entrypoint.py [build|preflight|shell]")

    run([sys.executable, str(APP_ROOT / "colab" / "colab_pipeline.py"), *args])
    copy_final_model(voice_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
