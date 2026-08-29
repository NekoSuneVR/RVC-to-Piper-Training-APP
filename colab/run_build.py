from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


RUNTIME_ROOT = Path("/content/rvc-piper-runtime")
LOG_PATH = RUNTIME_ROOT / "build.log"


def value_after(args: list[str], name: str, default: str = "") -> str:
    try:
        index = args.index(name)
    except ValueError:
        return default
    if index + 1 >= len(args):
        return default
    return args[index + 1]


def stream(command: list[str], env: dict[str, str]) -> int:
    print("> " + subprocess.list2cmdline(command), flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return process.wait()


def print_log_tail(lines: int = 160) -> None:
    if not LOG_PATH.is_file():
        return
    content = LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
    print("\n================ BUILD LOG TAIL ================", flush=True)
    for line in content[-lines:]:
        print(line, flush=True)
    print("================================================", flush=True)
    print(f"Full Colab build log: {LOG_PATH}", flush=True)


def main() -> int:
    pipeline_args = sys.argv[1:]
    if not pipeline_args:
        raise SystemExit("Pass the normal colab_pipeline.py arguments to this runner.")

    repo_root = value_after(pipeline_args, "--repo-root")
    rvc_root = value_after(pipeline_args, "--rvc-root")
    rvc_python = value_after(pipeline_args, "--rvc-python")
    piper_python = value_after(pipeline_args, "--piper-python")
    base_model = value_after(pipeline_args, "--base-model")
    base_config = value_after(pipeline_args, "--base-config")
    rvc_model = value_after(pipeline_args, "--rvc-model")
    rvc_index = value_after(pipeline_args, "--rvc-index")
    pitch = value_after(pipeline_args, "--pitch", "12")
    index_rate = value_after(pipeline_args, "--index-rate", "0.75")
    protect = value_after(pipeline_args, "--protect", "0.33")
    f0_method = value_after(pipeline_args, "--f0-method", "rmvpe")

    required = {
        "--repo-root": repo_root,
        "--rvc-root": rvc_root,
        "--rvc-python": rvc_python,
        "--piper-python": piper_python,
        "--base-model": base_model,
        "--base-config": base_config,
        "--rvc-model": rvc_model,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise SystemExit("Missing required build arguments: " + ", ".join(missing))

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # PyTorch 2.6+ defaults torch.load to weights_only=True. Classic RVC
    # checkpoints are trusted local pickle checkpoints and RVC upstream still
    # calls torch.load without an explicit weights_only argument. Force the
    # legacy behavior only for this Colab RVC/Piper child-process tree.
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    LOG_PATH.unlink(missing_ok=True)
    print("=== RVC + Piper Colab guarded build ===", flush=True)
    print("Legacy RVC checkpoint compatibility: enabled for trusted local .pth files", flush=True)
    print(f"Build log: {LOG_PATH}", flush=True)

    preflight = [
        sys.executable,
        str(Path(repo_root) / "colab" / "preflight.py"),
        "--repo-root", repo_root,
        "--rvc-root", rvc_root,
        "--rvc-python", rvc_python,
        "--piper-python", piper_python,
        "--base-model", base_model,
        "--base-config", base_config,
        "--rvc-model", rvc_model,
        "--pitch", pitch,
        "--index-rate", index_rate,
        "--protect", protect,
        "--f0-method", f0_method,
    ]
    if rvc_index:
        preflight += ["--rvc-index", rvc_index]

    print("\n=== PRE-FLIGHT: one sentence only ===", flush=True)
    code = stream(preflight, env)
    if code:
        print(f"\nPRE-FLIGHT FAILED with exit code {code}.", flush=True)
        print_log_tail()
        return code

    print("\n=== PRE-FLIGHT PASSED: starting full build ===", flush=True)
    pipeline = [
        sys.executable,
        str(Path(repo_root) / "colab" / "colab_pipeline.py"),
        *pipeline_args,
    ]
    code = stream(pipeline, env)
    if code:
        print(f"\nFULL BUILD FAILED with exit code {code}.", flush=True)
        print_log_tail()
        return code

    print("\nFULL COLAB BUILD COMPLETED SUCCESSFULLY.", flush=True)
    print(f"Build log: {LOG_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
