from __future__ import annotations

import os
import queue
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from app import APP_DIR, Settings, piper_command, rvc_command, validate
from piper_training import (
    PiperTrainPlan,
    export_command,
    latest_checkpoint,
    load_prompts,
    read_metadata,
    safe_voice_name,
    training_command,
    validate_trainer,
    write_metadata,
)

TRAINING_ROOT = APP_DIR / "training"
DEFAULT_PROMPTS = APP_DIR / "data" / "piper_training_prompts.txt"
DEFAULT_TRAINER_ROOT = APP_DIR / "tools" / "piper-trainer"
DEFAULT_TRAINER_SOURCE = DEFAULT_TRAINER_ROOT / "piper1-gpl"
DEFAULT_TRAINER_PYTHON = DEFAULT_TRAINER_ROOT / ".venv" / "Scripts" / "python.exe"


class PiperBuilder(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("RVC → Piper Model Builder")
        self.geometry("1080x790")
        self.minsize(940, 680)
        self.configure(bg="#0b1020")
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.process: subprocess.Popen[str] | None = None
        self.cancel_requested = False
        self._vars: dict[str, tk.Variable] = {}
        self._build_style()
        self._build_ui()
        self.after(100, self._poll)

    def _build_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        bg, card, field, text, muted, accent = "#0b1020", "#141b2d", "#0e1526", "#f3f6ff", "#8f9bb3", "#7c5cff"
        style.configure(".", background=bg, foreground=text, font=("Segoe UI", 10))
        style.configure("TFrame", background=bg)
        style.configure("Card.TFrame", background=card)
        style.configure("TLabel", background=bg, foreground=text)
        style.configure("Card.TLabel", background=card, foreground=text)
        style.configure("Title.TLabel", background=bg, foreground=text, font=("Segoe UI Semibold", 24))
        style.configure("Section.TLabel", background=card, foreground=text, font=("Segoe UI Semibold", 14))
        style.configure("Hint.TLabel", background=bg, foreground=muted)
        style.configure("CardHint.TLabel", background=card, foreground=muted)
        style.configure("Primary.TButton", background=accent, foreground="white", font=("Segoe UI Semibold", 10), padding=(14, 10), borderwidth=0)
        style.map("Primary.TButton", background=[("active", "#9178ff"), ("disabled", "#3a4055")])
        style.configure("Secondary.TButton", background="#222b44", foreground=text, padding=(12, 9), borderwidth=0)
        style.map("Secondary.TButton", background=[("active", "#303b5c")])
        style.configure("TEntry", fieldbackground=field, foreground=text, insertcolor=text, bordercolor="#303a58", padding=7)
        style.configure("TCombobox", fieldbackground=field, foreground=text, arrowcolor=text, padding=5)
        style.map("TCombobox", fieldbackground=[("readonly", field)], foreground=[("readonly", text)])
        style.configure("TProgressbar", background=accent, troughcolor="#202942", borderwidth=0)
        style.configure("TLabelframe", background=card, bordercolor="#2a3452", relief="solid")
        style.configure("TLabelframe.Label", background=card, foreground=text, font=("Segoe UI Semibold", 11))

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=22)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="RVC → Piper Model Builder", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            outer,
            text="Generate a text/audio dataset through your RVC voice, train Piper, then export a standalone ONNX voice.",
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(2, 14))

        card = ttk.Frame(outer, style="Card.TFrame", padding=18)
        card.pack(fill="x")
        top = ttk.Frame(card, style="Card.TFrame")
        top.pack(fill="x")
        self._entry(top, "Voice name", "voice_name", "en_GB-rvc-custom-medium", 0, 27)
        self._entry(top, "eSpeak voice", "espeak_voice", "en-gb", 1, 12)
        self._combo(top, "Device", "accelerator", ["auto", "gpu", "cpu"], "auto", 2)
        self._entry(top, "Batch size", "batch_size", "8", 3, 9)
        self._entry(top, "Max epochs", "max_epochs", "1000", 4, 10)

        self._path_row(card, "Training prompts", "prompts", str(DEFAULT_PROMPTS), [("Text files", "*.txt"), ("All files", "*.*")])
        count_row = ttk.Frame(card, style="Card.TFrame")
        count_row.pack(fill="x", pady=(8, 0))
        ttk.Label(count_row, text="Prompt limit", style="Card.TLabel", width=22).pack(side="left")
        count = tk.StringVar(value="120")
        self._vars["prompt_limit"] = count
        ttk.Entry(count_row, textvariable=count, width=12).pack(side="left")
        ttk.Label(count_row, text="Use 0 for every line. More clean lines usually improves the final voice.", style="CardHint.TLabel").pack(side="left", padx=10)

        self._path_row(card, "Piper trainer Python", "trainer_python", str(DEFAULT_TRAINER_PYTHON), [("Python", "python.exe")])
        self._path_row(card, "Piper trainer source", "trainer_source", str(DEFAULT_TRAINER_SOURCE), folder=True)
        self._path_row(card, "Warm-start checkpoint", "checkpoint", "", [("Piper checkpoint", "*.ckpt")], optional=True)

        use_model = tk.BooleanVar(value=True)
        self._vars["use_model"] = use_model
        ttk.Checkbutton(card, text="Use exported Piper model in Studio automatically", variable=use_model).pack(anchor="w", pady=(12, 0))

        actions = ttk.Frame(outer)
        actions.pack(fill="x", pady=(14, 8))
        self.setup_button = ttk.Button(actions, text="Install / repair trainer", style="Secondary.TButton", command=self.install_trainer)
        self.setup_button.pack(side="left")
        self.dataset_button = ttk.Button(actions, text="1. Build RVC dataset", style="Secondary.TButton", command=self.build_dataset)
        self.dataset_button.pack(side="left", padx=6)
        self.train_button = ttk.Button(actions, text="2. Train Piper", style="Secondary.TButton", command=self.train)
        self.train_button.pack(side="left", padx=6)
        self.export_button = ttk.Button(actions, text="3. Export ONNX", style="Secondary.TButton", command=self.export)
        self.export_button.pack(side="left", padx=6)
        self.all_button = ttk.Button(actions, text="Build everything", style="Primary.TButton", command=self.build_all)
        self.all_button.pack(side="left", padx=(10, 6))
        self.stop_button = ttk.Button(actions, text="Stop", style="Secondary.TButton", command=self.stop, state="disabled")
        self.stop_button.pack(side="left")

        self.progress = ttk.Progressbar(outer, mode="determinate", maximum=100)
        self.progress.pack(fill="x", pady=(6, 5))
        self.status = ttk.Label(outer, text="Ready. The current Studio model settings will be used for dataset generation.", style="Hint.TLabel")
        self.status.pack(anchor="w")

        log_card = ttk.Frame(outer, style="Card.TFrame", padding=12)
        log_card.pack(fill="both", expand=True, pady=(10, 0))
        ttk.Label(log_card, text="Build log", style="Section.TLabel").pack(anchor="w", pady=(0, 6))
        self.log = tk.Text(log_card, wrap="word", height=14, bg="#0e1526", fg="#dfe6ff", insertbackground="#f3f6ff", relief="flat", font=("Cascadia Mono", 9))
        self.log.pack(fill="both", expand=True)

    def _entry(self, parent, label, key, value, column, width=18):
        frame = ttk.Frame(parent, style="Card.TFrame")
        frame.grid(row=0, column=column, sticky="w", padx=(0, 12))
        ttk.Label(frame, text=label, style="Card.TLabel").pack(anchor="w")
        var = tk.StringVar(value=value)
        self._vars[key] = var
        ttk.Entry(frame, textvariable=var, width=width).pack(pady=(4, 0))

    def _combo(self, parent, label, key, values, value, column):
        frame = ttk.Frame(parent, style="Card.TFrame")
        frame.grid(row=0, column=column, sticky="w", padx=(0, 12))
        ttk.Label(frame, text=label, style="Card.TLabel").pack(anchor="w")
        var = tk.StringVar(value=value)
        self._vars[key] = var
        ttk.Combobox(frame, textvariable=var, values=values, state="readonly", width=10).pack(pady=(4, 0))

    def _path_row(self, parent, label, key, value, types=None, folder=False, optional=False):
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", pady=(8, 0))
        ttk.Label(row, text=label + (" (optional)" if optional else ""), style="Card.TLabel", width=22).pack(side="left")
        var = tk.StringVar(value=value)
        self._vars[key] = var
        ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True)

        def browse():
            chosen = filedialog.askdirectory() if folder else filedialog.askopenfilename(filetypes=types)
            if chosen:
                var.set(chosen)

        ttk.Button(row, text="Browse…", style="Secondary.TButton", command=browse).pack(side="left", padx=(6, 0))

    def _plan(self, require_dataset: bool = False) -> PiperTrainPlan:
        name = safe_voice_name(str(self._vars["voice_name"].get()))
        batch_size = int(str(self._vars["batch_size"].get()).strip())
        max_epochs = int(str(self._vars["max_epochs"].get()).strip())
        if batch_size < 1:
            raise ValueError("Batch size must be at least 1.")
        if max_epochs < 1:
            raise ValueError("Max epochs must be at least 1.")
        checkpoint_raw = str(self._vars["checkpoint"].get()).strip()
        project = TRAINING_ROOT / name
        plan = PiperTrainPlan(
            voice_name=name,
            trainer_python=Path(str(self._vars["trainer_python"].get()).strip()),
            trainer_source=Path(str(self._vars["trainer_source"].get()).strip()),
            dataset_dir=project / "dataset",
            output_dir=project / "piper",
            espeak_voice=str(self._vars["espeak_voice"].get()).strip() or "en-gb",
            batch_size=batch_size,
            max_epochs=max_epochs,
            accelerator=str(self._vars["accelerator"].get()).strip() or "auto",
            checkpoint=Path(checkpoint_raw) if checkpoint_raw else None,
        )
        if require_dataset:
            errors = validate_trainer(plan)
            if errors:
                raise ValueError("\n".join(errors))
        return plan

    def _prompt_limit(self) -> int | None:
        value = int(str(self._vars["prompt_limit"].get()).strip())
        if value < 0:
            raise ValueError("Prompt limit cannot be negative.")
        return value or None

    def _set_busy(self, busy: bool, status: str | None = None) -> None:
        state = "disabled" if busy else "normal"
        for button in (self.setup_button, self.dataset_button, self.train_button, self.export_button, self.all_button):
            button.configure(state=state)
        self.stop_button.configure(state="normal" if busy else "disabled")
        if status:
            self.status.configure(text=status)
        if busy:
            self.cancel_requested = False
        else:
            self.process = None

    def _start(self, target, status: str) -> None:
        if self.process is not None:
            return
        self._set_busy(True, status)
        threading.Thread(target=target, daemon=True).start()

    def install_trainer(self) -> None:
        script = APP_DIR / "setup-piper-training.ps1"
        if not script.is_file():
            messagebox.showerror("Missing setup script", f"Could not find {script}")
            return

        accelerator = str(self._vars.get("accelerator", tk.StringVar(value="auto")).get()).strip().lower()
        engine = "cpu" if accelerator == "cpu" else "cuda"

        def worker():
            try:
                command = [
                    "powershell.exe",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(script),
                    "-Engine",
                    engine,
                    "-InstallBuildTools",
                ]
                self.events.put(("log", f"Trainer engine selected: {engine}"))
                self.events.put(("log", "Missing Microsoft C++ Build Tools will be installed automatically with winget."))
                self._run_stream(command, APP_DIR)
                self.events.put(("complete", "Piper training environment is ready. Close/reopen Studio or press Refresh on tab 3."))
            except Exception as exc:
                self.events.put(("error", str(exc)))

        self._start(worker, "Installing Piper trainer and required Windows build tools…")

    def build_dataset(self) -> None:
        def worker():
            try:
                self._build_dataset_worker()
                self.events.put(("complete", "RVC training dataset is ready."))
            except Exception as exc:
                self.events.put(("error", str(exc)))

        self._start(worker, "Generating the RVC training dataset…")

    def _build_dataset_worker(self) -> None:
        settings = Settings.load()
        errors = validate(settings)
        if errors:
            raise RuntimeError("Studio voice setup is incomplete:\n" + "\n".join(errors))
        plan = self._plan()
        prompts = load_prompts(Path(str(self._vars["prompts"].get()).strip()), self._prompt_limit())
        plan.audio_dir.mkdir(parents=True, exist_ok=True)
        existing = read_metadata(plan.metadata_csv)
        rows: list[tuple[str, str]] = []
        total = len(prompts)
        for index, text in enumerate(prompts, start=1):
            if self.cancel_requested:
                raise RuntimeError("Dataset build stopped by user.")
            file_name = f"utt-{index:05d}.wav"
            final = plan.audio_dir / file_name
            if final.is_file() and final.stat().st_size > 0 and existing.get(file_name) == text:
                rows.append((file_name, text))
                self.events.put(("log", f"[{index}/{total}] keep {file_name}"))
                self.events.put(("progress", index * 100 / total))
                continue
            self.events.put(("status", f"Generating dataset line {index} of {total}…"))
            with tempfile.TemporaryDirectory(prefix="rvc-piper-dataset-") as raw:
                base = Path(raw) / "base.wav"
                self._run_capture(piper_command(settings, base), APP_DIR, text + "\n")
                self._run_capture(rvc_command(settings, base, final), Path(settings.rvc_root))
            if not final.is_file() or final.stat().st_size == 0:
                raise RuntimeError(f"RVC did not create {final.name}")
            rows.append((file_name, text))
            write_metadata(plan.metadata_csv, rows)
            self.events.put(("log", f"[{index}/{total}] created {file_name} | {text}"))
            self.events.put(("progress", index * 100 / total))
        write_metadata(plan.metadata_csv, rows)
        self.events.put(("log", f"Dataset complete: {len(rows)} utterances in {plan.dataset_dir}"))

    def train(self) -> None:
        def worker():
            try:
                self._train_worker()
                self.events.put(("complete", "Piper training finished. Export the newest checkpoint next."))
            except Exception as exc:
                self.events.put(("error", str(exc)))

        self._start(worker, "Training Piper…")

    def _train_worker(self) -> None:
        plan = self._plan(require_dataset=True)
        plan.output_dir.mkdir(parents=True, exist_ok=True)
        self.events.put(("progress", 0))
        self.events.put(("log", "Starting Piper trainer…"))
        self._run_stream(training_command(plan), plan.trainer_source)

    def export(self) -> None:
        def worker():
            try:
                model = self._export_worker()
                self.events.put(("complete", f"Exported Piper voice: {model}"))
            except Exception as exc:
                self.events.put(("error", str(exc)))

        self._start(worker, "Exporting the latest Piper checkpoint to ONNX…")

    def _export_worker(self) -> Path:
        plan = self._plan(require_dataset=True)
        checkpoint = latest_checkpoint(plan.lightning_dir)
        if checkpoint is None:
            raise RuntimeError(f"No .ckpt file was found under {plan.lightning_dir}")
        self.events.put(("log", f"Exporting checkpoint: {checkpoint}"))
        plan.output_dir.mkdir(parents=True, exist_ok=True)
        self._run_stream(export_command(plan, checkpoint), plan.trainer_source)
        if not plan.onnx_path.is_file() or plan.onnx_path.stat().st_size == 0:
            raise RuntimeError("Piper export finished without creating the ONNX model.")
        if not plan.config_path.is_file():
            raise RuntimeError("Piper export succeeded, but the matching ONNX JSON config is missing.")
        if bool(self._vars["use_model"].get()):
            settings = Settings.load()
            settings.piper_model = str(plan.onnx_path)
            settings.piper_config = str(plan.config_path)
            settings.save()
            self.events.put(("log", "Studio settings updated to use the new Piper voice."))
        self.events.put(("progress", 100))
        return plan.onnx_path

    def build_all(self) -> None:
        def worker():
            try:
                plan = self._plan()
                if not plan.trainer_python.is_file() or not (plan.trainer_source / "src" / "piper" / "train").is_dir():
                    raise RuntimeError("Piper trainer is not installed yet. Click 'Install / repair trainer' first.")
                self._build_dataset_worker()
                self.events.put(("status", "Dataset ready. Starting Piper training…"))
                self._train_worker()
                self.events.put(("status", "Training complete. Exporting ONNX…"))
                model = self._export_worker()
                self.events.put(("complete", f"Full Piper build complete: {model}"))
            except Exception as exc:
                self.events.put(("error", str(exc)))

        self._start(worker, "Building the complete Piper voice…")

    def stop(self) -> None:
        self.cancel_requested = True
        process = self.process
        if process and process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass
        self.status.configure(text="Stopping current task…")

    def _run_capture(self, command: list[str], cwd: Path, stdin: str | None = None) -> None:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            command,
            input=stdin,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=cwd,
            capture_output=True,
            creationflags=flags,
            env=env,
        )
        if result.returncode:
            detail = (result.stderr or result.stdout or "Unknown error").strip()
            raise RuntimeError(detail[-5000:])

    def _run_stream(self, command: list[str], cwd: Path) -> None:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        self.events.put(("log", "> " + subprocess.list2cmdline(command)))
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=flags,
            env=env,
        )
        self.process = process
        assert process.stdout is not None
        for line in process.stdout:
            self.events.put(("log", line.rstrip()))
            if self.cancel_requested and process.poll() is None:
                process.terminate()
                break
        code = process.wait()
        self.process = None
        if self.cancel_requested:
            raise RuntimeError("Task stopped by user.")
        if code:
            raise RuntimeError(f"Command failed with exit code {code}. See the build log above.")

    def _poll(self) -> None:
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "log":
                    self.log.insert("end", str(value) + "\n")
                    self.log.see("end")
                elif kind == "status":
                    self.status.configure(text=str(value))
                elif kind == "progress":
                    self.progress["value"] = float(value)
                elif kind == "complete":
                    self._set_busy(False, str(value))
                    self.progress["value"] = 100
                    self.log.insert("end", "\n✓ " + str(value) + "\n")
                    self.log.see("end")
                elif kind == "error":
                    self._set_busy(False, "Build failed")
                    self.log.insert("end", "\nERROR: " + str(value) + "\n")
                    self.log.see("end")
                    messagebox.showerror("Piper model build failed", str(value))
        except queue.Empty:
            pass
        self.after(100, self._poll)


if __name__ == "__main__":
    PiperBuilder().mainloop()
