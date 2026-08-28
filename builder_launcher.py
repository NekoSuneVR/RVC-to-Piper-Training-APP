from __future__ import annotations

import argparse

from app import APP_DIR
from piper_builder_gui import PiperBuilder


TRAINER_SETUP = APP_DIR / "setup-piper-training.ps1"
TRAINER_MARKER = APP_DIR / "tools" / "piper-trainer" / ".setup-complete"
REQUIRED_TRAINER_MARKER = "piper-trainer-v6"


class GuardedPiperBuilder(PiperBuilder):
    """Builder that bootstraps and repairs the Piper training environment when needed."""

    def _marker_current(self) -> bool:
        if not TRAINER_MARKER.is_file():
            return False
        try:
            return TRAINER_MARKER.read_text(encoding="utf-8-sig").strip().startswith(REQUIRED_TRAINER_MARKER)
        except OSError:
            return False

    def _trainer_ready(self) -> bool:
        plan = self._plan()
        piper_dir = plan.trainer_source / "src" / "piper"
        align_dir = piper_dir / "train" / "vits" / "monotonic_align"
        nested_align_dir = align_dir / "monotonic_align"
        has_espeak_bridge = any(piper_dir.glob("espeakbridge*.pyd"))
        has_alignment = any(nested_align_dir.glob("core*.pyd"))
        has_espeak_data = (piper_dir / "espeak-ng-data").is_dir()
        return (
            plan.trainer_python.is_file()
            and (piper_dir / "train").is_dir()
            and has_espeak_bridge
            and has_alignment
            and has_espeak_data
            and self._marker_current()
        )

    def _missing_trainer_parts(self) -> list[str]:
        plan = self._plan()
        piper_dir = plan.trainer_source / "src" / "piper"
        align_dir = piper_dir / "train" / "vits" / "monotonic_align"
        nested_align_dir = align_dir / "monotonic_align"
        missing: list[str] = []
        if not plan.trainer_python.is_file():
            missing.append(f"Piper trainer Python: {plan.trainer_python}")
        if not (piper_dir / "train").is_dir():
            missing.append(f"Piper training source: {plan.trainer_source}")
        if not any(piper_dir.glob("espeakbridge*.pyd")):
            missing.append(f"Piper eSpeak native bridge: {piper_dir / 'espeakbridge*.pyd'}")
        if not (piper_dir / "espeak-ng-data").is_dir():
            missing.append(f"Piper eSpeak data: {piper_dir / 'espeak-ng-data'}")
        if not any(nested_align_dir.glob("core*.pyd")):
            missing.append(f"Piper nested monotonic alignment extension: {nested_align_dir / 'core*.pyd'}")
        if not self._marker_current():
            if TRAINER_MARKER.is_file():
                try:
                    found = TRAINER_MARKER.read_text(encoding="utf-8-sig").strip().split(";", 1)[0]
                except OSError:
                    found = "unreadable marker"
                missing.append(
                    f"Current trainer setup marker {REQUIRED_TRAINER_MARKER} (found {found}): {TRAINER_MARKER}"
                )
            else:
                missing.append(f"Successful trainer setup marker {REQUIRED_TRAINER_MARKER}: {TRAINER_MARKER}")
        return missing

    def _install_trainer_sync(self) -> None:
        if self._trainer_ready():
            return
        if not TRAINER_SETUP.is_file():
            raise RuntimeError(f"Piper trainer setup script not found: {TRAINER_SETUP}")

        accelerator = str(self._vars["accelerator"].get()).strip().lower()
        engine = "cpu" if accelerator == "cpu" else "cuda"
        missing_before = self._missing_trainer_parts()
        self.events.put(("status", "Piper trainer is incomplete or outdated. Repairing it first…"))
        self.events.put(("log", "Piper trainer readiness check failed; starting automatic repair."))
        for item in missing_before:
            self.events.put(("log", f"Missing/outdated: {item}"))
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
            missing = self._missing_trainer_parts()
            raise RuntimeError("Trainer repair finished but is still incomplete:\n" + "\n".join(missing))

        self.events.put(("log", "Piper trainer repair completed successfully."))

    def install_trainer(self) -> None:
        def worker():
            try:
                self._install_trainer_sync()
                self.events.put(("complete", "Piper training environment is ready."))
            except Exception as exc:
                self.events.put(("error", str(exc)))

        self._start(worker, "Installing / repairing Piper trainer and native extensions…")

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
        app.after(350, action)
    app.mainloop()


if __name__ == "__main__":
    main()
