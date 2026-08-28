from __future__ import annotations

import argparse
from pathlib import Path

from app import APP_DIR
from piper_builder_gui import PiperBuilder


TRAINER_SETUP = APP_DIR / "setup-piper-training.ps1"


class GuardedPiperBuilder(PiperBuilder):
    """Builder that bootstraps the Piper training environment when needed."""

    def _trainer_ready(self) -> bool:
        plan = self._plan()
        return (
            plan.trainer_python.is_file()
            and (plan.trainer_source / "src" / "piper" / "train").is_dir()
        )

    def _install_trainer_sync(self) -> None:
        if self._trainer_ready():
            return
        if not TRAINER_SETUP.is_file():
            raise RuntimeError(f"Piper trainer setup script not found: {TRAINER_SETUP}")

        accelerator = str(self._vars["accelerator"].get()).strip().lower()
        engine = "cpu" if accelerator == "cpu" else "cuda"
        self.events.put(("status", "Piper trainer is missing. Installing it first…"))
        self.events.put(("log", "Piper trainer environment was not found; starting automatic bootstrap."))
        self.events.put(("log", f"Selected trainer engine: {engine}"))

        command = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(TRAINER_SETUP),
            "-Engine",
            engine,
            "-InstallBuildTools",
        ]
        self._run_stream(command, APP_DIR)

        if not self._trainer_ready():
            plan = self._plan()
            missing = []
            if not plan.trainer_python.is_file():
                missing.append(f"Piper trainer Python not found after setup: {plan.trainer_python}")
            train_source = plan.trainer_source / "src" / "piper" / "train"
            if not train_source.is_dir():
                missing.append(f"Piper training source not found after setup: {plan.trainer_source}")
            raise RuntimeError("Trainer setup finished but is incomplete:\n" + "\n".join(missing))

        self.events.put(("log", "Piper trainer bootstrap completed successfully."))

    def install_trainer(self) -> None:
        def worker():
            try:
                self._install_trainer_sync()
                self.events.put(("complete", "Piper training environment is ready."))
            except Exception as exc:
                self.events.put(("error", str(exc)))

        self._start(worker, "Installing Piper trainer and required Windows build tools…")

    def train(self) -> None:
        def worker():
            try:
                self._install_trainer_sync()
                self._train_worker()
                self.events.put(("complete", "Piper training finished. Export the newest checkpoint next."))
            except Exception as exc:
                self.events.put(("error", str(exc)))

        self._start(worker, "Checking Piper trainer before training…")

    def export(self) -> None:
        def worker():
            try:
                self._install_trainer_sync()
                model = self._export_worker()
                self.events.put(("complete", f"Exported Piper voice: {model}"))
            except Exception as exc:
                self.events.put(("error", str(exc)))

        self._start(worker, "Checking Piper trainer before ONNX export…")

    def build_all(self) -> None:
        def worker():
            try:
                self._install_trainer_sync()
                self.events.put(("status", "Trainer ready. Building the RVC dataset…"))
                self._build_dataset_worker()
                self.events.put(("status", "Dataset ready. Starting Piper training…"))
                self._train_worker()
                self.events.put(("status", "Training complete. Exporting ONNX…"))
                model = self._export_worker()
                self.events.put(("complete", f"Full Piper build complete: {model}"))
            except Exception as exc:
                self.events.put(("error", str(exc)))

        self._start(worker, "Checking Piper trainer before the full build…")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Piper model builder and optionally start one build step.")
    parser.add_argument(
        "--action",
        choices=("open", "install", "dataset", "train", "export", "all"),
        default="open",
        help="Builder action to run after the window opens.",
    )
    args = parser.parse_args()

    app = GuardedPiperBuilder()
    actions = {
        "install": app.install_trainer,
        "dataset": app.build_dataset,
        "train": app.train,
        "export": app.export,
        "all": app.build_all,
    }
    action = actions.get(args.action)
    if action is not None:
        # Give Tk a moment to finish drawing the builder before a long task starts.
        app.after(350, action)
    app.mainloop()


if __name__ == "__main__":
    main()
