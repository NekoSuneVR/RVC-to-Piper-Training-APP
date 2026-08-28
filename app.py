from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


APP_DIR = Path(__file__).resolve().parent
SETTINGS_FILE = APP_DIR / "data" / "settings.json"
OUTPUT_DIR = APP_DIR / "generated"
PITCH_TEST_TEXT = (
    "Pitch test. One, two, three. The quick brown fox jumps over the lazy dog. "
    "She sells seashells by the seashore. This sample tests low, middle, and high tones."
)


@dataclass
class Settings:
    piper_exe: str = str(APP_DIR / "tools" / "piper" / "piper" / "piper.exe")
    piper_model: str = str(APP_DIR / "models" / "piper" / "en_GB-alba-medium.onnx")
    piper_config: str = str(APP_DIR / "models" / "piper" / "en_GB-alba-medium.onnx.json")
    rvc_root: str = str(APP_DIR / "tools" / "rvc")
    rvc_python: str = str(APP_DIR / "tools" / "python" / "python.exe")
    rvc_model: str = ""
    rvc_index: str = ""
    pitch: int = 0
    index_rate: float = 0.75
    protect: float = 0.33
    f0_method: str = "rmvpe"

    @classmethod
    def load(cls) -> "Settings":
        if not SETTINGS_FILE.exists():
            return cls()
        try:
            raw = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            loaded = cls(**{k: v for k, v in raw.items() if k in cls.__annotations__})
            nested_piper = APP_DIR / "tools" / "piper" / "piper" / "piper.exe"
            if not Path(loaded.piper_exe).is_file() and nested_piper.is_file():
                loaded.piper_exe = str(nested_piper)
            if not loaded.piper_model:
                loaded.piper_model = str(APP_DIR / "models" / "piper" / "en_GB-alba-medium.onnx")
            if not loaded.piper_config:
                loaded.piper_config = str(APP_DIR / "models" / "piper" / "en_GB-alba-medium.onnx.json")
            portable_python = APP_DIR / "tools" / "python" / "python.exe"
            if not Path(loaded.rvc_python).is_file() and portable_python.is_file():
                loaded.rvc_python = str(portable_python)
            return loaded
        except (OSError, ValueError, TypeError):
            return cls()

    def save(self) -> None:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def piper_command(settings: Settings, output: Path) -> list[str]:
    command = [settings.piper_exe, "--model", settings.piper_model]
    if settings.piper_config:
        command += ["--config", settings.piper_config]
    return command + ["--output_file", str(output)]


def rvc_command(settings: Settings, source: Path, output: Path) -> list[str]:
    command = [
        settings.rvc_python,
        "-m", "infer.cli",
        "--model", settings.rvc_model,
        "--input", str(source),
        "--output", str(output),
        "--pitch", str(settings.pitch),
        "--f0-method", settings.f0_method,
        "--index-rate", str(settings.index_rate),
        "--protect", str(settings.protect),
        "--format", "wav",
        "--overwrite",
    ]
    if settings.rvc_index:
        command += ["--index", settings.rvc_index]
    return command


def validate(settings: Settings) -> list[str]:
    checks = [
        (settings.piper_exe, "Piper program"),
        (settings.piper_model, "Piper voice (.onnx)"),
        (settings.rvc_python, "RVC Python runtime"),
        (str(Path(settings.rvc_root) / "infer" / "cli.py"), "RVC offline inference"),
        (settings.rvc_model, "RVC model (.pth)"),
    ]
    errors = [f"{label} not found: {path or '(not selected)'}" for path, label in checks if not Path(path).is_file()]
    if settings.piper_model and Path(settings.piper_model).suffix.lower() != ".onnx":
        errors.append("Piper voice must be an .onnx file.")
    if settings.rvc_model and Path(settings.rvc_model).suffix.lower() != ".pth":
        errors.append("RVC model must be a .pth file.")
    if settings.rvc_index and not Path(settings.rvc_index).is_file():
        errors.append(f"RVC index not found: {settings.rvc_index}")
    return errors


class Studio(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("RVC + Piper Studio")
        self.geometry("1120x780")
        self.minsize(940, 680)
        self.configure(bg="#0b1020")
        self.settings = Settings.load()
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.last_output: Path | None = None
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
        style.configure("Title.TLabel", background=bg, foreground=text, font=("Segoe UI Semibold", 25))
        style.configure("Section.TLabel", background=card, foreground=text, font=("Segoe UI Semibold", 14))
        style.configure("Hint.TLabel", background=bg, foreground=muted)
        style.configure("CardHint.TLabel", background=card, foreground=muted)
        style.configure("Step.TLabel", background=card, foreground="#cbd3e6", font=("Segoe UI Semibold", 10))
        style.configure("Badge.TLabel", background="#252d46", foreground="#b9c2dc", padding=(8, 4))
        style.configure("Primary.TButton", background=accent, foreground="white", font=("Segoe UI Semibold", 11), padding=(16, 11), borderwidth=0)
        style.map("Primary.TButton", background=[("active", "#9178ff"), ("disabled", "#3a4055")])
        style.configure("Secondary.TButton", background="#222b44", foreground=text, padding=(12, 9), borderwidth=0)
        style.map("Secondary.TButton", background=[("active", "#303b5c")])
        style.configure("TEntry", fieldbackground=field, foreground=text, insertcolor=text, bordercolor="#303a58", padding=7)
        style.configure("TCombobox", fieldbackground=field, foreground=text, arrowcolor=text, padding=5)
        style.map("TCombobox", fieldbackground=[("readonly", field)], foreground=[("readonly", text)])
        style.configure("TNotebook", background=bg, borderwidth=0)
        style.configure("TNotebook.Tab", background="#11182a", foreground=muted, padding=(18, 10), borderwidth=0, font=("Segoe UI Semibold", 10))
        style.map("TNotebook.Tab", background=[("selected", card)], foreground=[("selected", text)])
        style.configure("TProgressbar", background=accent, troughcolor="#202942", borderwidth=0)
        style.configure("Horizontal.TScale", background=card, troughcolor="#2a3452")
        style.configure("TLabelframe", background=card, bordercolor="#2a3452", relief="solid")
        style.configure("TLabelframe.Label", background=card, foreground=text, font=("Segoe UI Semibold", 11))

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=24)
        outer.pack(fill="both", expand=True)
        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 18))
        titlebox = ttk.Frame(header)
        titlebox.pack(side="left")
        ttk.Label(titlebox, text="RVC + Piper Studio", style="Title.TLabel").pack(anchor="w")
        ttk.Label(titlebox, text="Create, test and tune AI voices locally — no uploads.", style="Hint.TLabel").pack(anchor="w", pady=(2, 0))
        ttk.Label(header, text="LOCAL  •  PRIVATE", style="Badge.TLabel").pack(side="right", anchor="n", pady=6)

        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)
        guide = ttk.Frame(body, style="Card.TFrame", padding=18, width=245)
        guide.pack(side="left", fill="y", padx=(0, 16))
        guide.pack_propagate(False)
        ttk.Label(guide, text="Quick start", style="Section.TLabel").pack(anchor="w", pady=(0, 14))
        self._guide_step(guide, "1", "Run Easy Setup", "Installs Piper and the RVC engine.")
        self._guide_step(guide, "2", "Add your models", "Choose a Piper .onnx and RVC .pth file.")
        self._guide_step(guide, "3", "Test and tune", "Move pitch, then generate a short test.")
        ttk.Separator(guide).pack(fill="x", pady=18)
        ttk.Label(guide, text="What you need", style="Step.TLabel").pack(anchor="w")
        ttk.Label(guide, text="• Piper voice (.onnx)\n• Matching Piper config (.json)\n• RVC voice (.pth)\n• RVC index (.index, optional)", style="CardHint.TLabel", justify="left", wraplength=205).pack(anchor="w", pady=(8, 0))
        ttk.Label(guide, text="Use only voices you have permission to use.", style="CardHint.TLabel", wraplength=205).pack(side="bottom", anchor="w")

        self.notebook = ttk.Notebook(body)
        self.notebook.pack(side="left", fill="both", expand=True)
        setup = ttk.Frame(self.notebook, style="Card.TFrame", padding=20)
        create = ttk.Frame(self.notebook, style="Card.TFrame", padding=20)
        self.notebook.add(setup, text="  1  MODELS & SETUP  ")
        self.notebook.add(create, text="  2  TUNE & CREATE  ")

        ttk.Label(create, text="Voice preview & pitch tuning", style="Section.TLabel").pack(anchor="w")
        ttk.Label(create, text="Tune the voice, run a short standard test, then create your own line.", style="CardHint.TLabel").pack(anchor="w", pady=(2, 14))

        tuner = ttk.LabelFrame(create, text=" Tuning controls ", padding=14)
        tuner.pack(fill="x")
        self._pitch_control(tuner)
        self._range_control(tuner, "Voice similarity", "index_rate", self.settings.index_rate, 0.0, 1.0, "How strongly RVC follows the model", row=1)
        self._range_control(tuner, "Sound protection", "protect", self.settings.protect, 0.0, 0.5, "Keeps S, T and breath sounds clear", row=2)
        method_row = ttk.Frame(tuner, style="Card.TFrame")
        method_row.grid(row=3, column=0, columnspan=4, sticky="w", pady=(12, 0))
        ttk.Label(method_row, text="Pitch detection", style="Card.TLabel", width=20).pack(side="left")
        method = tk.StringVar(value=self.settings.f0_method)
        self._vars["f0_method"] = method
        ttk.Combobox(method_row, textvariable=method, values=["rmvpe", "pm"], width=10, state="readonly").pack(side="left", padx=(8, 8))
        ttk.Label(method_row, text="RMVPE recommended", style="CardHint.TLabel").pack(side="left")

        preview = ttk.Frame(create, style="Card.TFrame")
        preview.pack(fill="x", pady=(14, 16))
        self.test_button = ttk.Button(preview, text="▶  Generate pitch test", style="Primary.TButton", command=lambda: self.generate(test=True))
        self.test_button.pack(side="left")
        ttk.Label(preview, text="Uses a fixed phrase so every pitch comparison is consistent.", style="CardHint.TLabel").pack(side="left", padx=12)

        ttk.Label(create, text="Your text", style="Card.TLabel", font=("Segoe UI Semibold", 11)).pack(anchor="w")
        self.text = tk.Text(create, height=7, wrap="word", font=("Segoe UI", 11), padx=12, pady=10, bg="#0e1526", fg="#f3f6ff", insertbackground="#f3f6ff", relief="flat", selectbackground="#6246d8")
        self.text.pack(fill="x", pady=(6, 12))
        self.text.insert("1.0", "Hello! This is Piper speaking through my RVC voice model.")
        buttons = ttk.Frame(create, style="Card.TFrame")
        buttons.pack(fill="x", pady=(0, 8))
        self.generate_button = ttk.Button(buttons, text="Generate my audio", style="Primary.TButton", command=self.generate)
        self.generate_button.pack(side="left")
        self.play_button = ttk.Button(buttons, text="Play last result", style="Secondary.TButton", command=self.play, state="disabled")
        self.play_button.pack(side="left", padx=8)
        ttk.Button(buttons, text="Open outputs", style="Secondary.TButton", command=lambda: self._open(OUTPUT_DIR)).pack(side="left")
        self.progress = ttk.Progressbar(create, mode="indeterminate")
        self.progress.pack(fill="x", pady=(6, 6))
        self.status = ttk.Label(create, text="Ready to create", style="CardHint.TLabel")
        self.status.pack(anchor="w")

        ttk.Label(setup, text="Connect your voice models", style="Section.TLabel").pack(anchor="w")
        ttk.Label(setup, text="Do this once. Your choices are remembered next time.", style="CardHint.TLabel").pack(anchor="w", pady=(2, 10))
        self._path_row(setup, "Piper program", "piper_exe", self.settings.piper_exe, [("Program", "*.exe")])
        self._path_row(setup, "Piper base voice", "piper_model", self.settings.piper_model, [("Piper model", "*.onnx")])
        self._path_row(setup, "Piper voice config", "piper_config", self.settings.piper_config, [("JSON config", "*.json")], optional=True)
        self._path_row(setup, "RVC installation folder", "rvc_root", self.settings.rvc_root, folder=True)
        self._path_row(setup, "RVC Python", "rvc_python", self.settings.rvc_python, [("Python", "python.exe")])
        self._path_row(setup, "RVC voice model", "rvc_model", self.settings.rvc_model, [("RVC model", "*.pth")])
        self._path_row(setup, "RVC feature index", "rvc_index", self.settings.rvc_index, [("RVC index", "*.index")], optional=True)
        actions = ttk.Frame(setup, style="Card.TFrame")
        actions.pack(fill="x", pady=(16, 5))
        ttk.Button(actions, text="Save and check setup", style="Primary.TButton", command=self.check_setup).pack(side="left")
        ttk.Label(actions, text="A neutral Piper voice usually gives cleaner RVC results.", style="CardHint.TLabel").pack(side="left", padx=12)

        if not validate(self.settings):
            self.notebook.select(create)

    def _guide_step(self, parent, number, title, detail):
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", pady=(0, 14))
        ttk.Label(row, text=number, style="Badge.TLabel").pack(side="left", anchor="n", padx=(0, 9))
        copy = ttk.Frame(row, style="Card.TFrame")
        copy.pack(side="left", fill="x", expand=True)
        ttk.Label(copy, text=title, style="Step.TLabel").pack(anchor="w")
        ttk.Label(copy, text=detail, style="CardHint.TLabel", wraplength=165).pack(anchor="w", pady=(2, 0))

    def _entry(self, parent, label, key, value, column, width=18):
        frame = ttk.Frame(parent)
        frame.grid(row=0, column=column, sticky="w", padx=(0, 18))
        ttk.Label(frame, text=label).pack(anchor="w")
        var = tk.StringVar(value=value)
        self._vars[key] = var
        ttk.Entry(frame, textvariable=var, width=width).pack(pady=(4, 0))

    def _combo(self, parent, label, key, values, value, column):
        frame = ttk.Frame(parent)
        frame.grid(row=0, column=column, sticky="w", padx=(0, 18))
        ttk.Label(frame, text=label).pack(anchor="w")
        var = tk.StringVar(value=value)
        self._vars[key] = var
        ttk.Combobox(frame, textvariable=var, values=values, width=13, state="readonly").pack(pady=(4, 0))

    def _pitch_control(self, parent):
        ttk.Label(parent, text="Pitch", width=20, style="Card.TLabel").grid(row=0, column=0, sticky="w")
        var = tk.DoubleVar(value=self.settings.pitch)
        self._vars["pitch"] = var
        value_label = ttk.Label(parent, width=15, anchor="center", style="Card.TLabel", font=("Segoe UI Semibold", 10))
        value_label.grid(row=0, column=2, padx=(8, 0))

        def update_label(*_):
            value = int(round(var.get()))
            if value == 0:
                value_label.configure(text="0 · Natural")
            else:
                value_label.configure(text=f"{value:+d} semitones")

        scale = ttk.Scale(parent, from_=-24, to=24, variable=var, command=lambda _: update_label())
        scale.grid(row=0, column=1, sticky="ew", padx=8)
        parent.columnconfigure(1, weight=1)
        update_label()
        presets = ttk.Frame(parent, style="Card.TFrame")
        presets.grid(row=0, column=3, padx=(12, 0))
        for caption, amount in (("−12 octave", -12), ("Reset", 0), ("+12 octave", 12)):
            ttk.Button(presets, text=caption, style="Secondary.TButton", command=lambda n=amount: (var.set(n), update_label())).pack(side="left", padx=2)

    def _range_control(self, parent, label, key, value, minimum, maximum, hint, row):
        ttk.Label(parent, text=label, width=20, style="Card.TLabel").grid(row=row, column=0, sticky="w", pady=(10, 0))
        var = tk.DoubleVar(value=value)
        self._vars[key] = var
        value_label = ttk.Label(parent, width=15, anchor="center", style="Card.TLabel")
        value_label.grid(row=row, column=2, padx=(8, 0), pady=(10, 0))

        def update_label(*_):
            value_label.configure(text=f"{round(var.get() * 100):d}%")

        ttk.Scale(parent, from_=minimum, to=maximum, variable=var, command=lambda _: update_label()).grid(row=row, column=1, sticky="ew", padx=8, pady=(10, 0))
        ttk.Label(parent, text=hint, style="CardHint.TLabel").grid(row=row, column=3, sticky="w", padx=(12, 0), pady=(10, 0))
        update_label()

    def _path_row(self, parent, label, key, value, types=None, folder=False, optional=False):
        ttk.Label(parent, text=label + (" (optional)" if optional else ""), style="Card.TLabel").pack(anchor="w", pady=(8, 0))
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", pady=(3, 0))
        var = tk.StringVar(value=value)
        self._vars[key] = var
        ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True)
        def browse():
            chosen = filedialog.askdirectory() if folder else filedialog.askopenfilename(filetypes=types)
            if chosen:
                var.set(chosen)
        ttk.Button(row, text="Browse…", style="Secondary.TButton", command=browse).pack(side="left", padx=(6, 0))

    def _collect(self) -> Settings:
        for key in ("piper_exe", "piper_model", "piper_config", "rvc_root", "rvc_python", "rvc_model", "rvc_index", "f0_method"):
            setattr(self.settings, key, str(self._vars[key].get()).strip())
        self.settings.pitch = int(round(float(self._vars["pitch"].get())))
        self.settings.index_rate = round(float(self._vars["index_rate"].get()), 2)
        self.settings.protect = round(float(self._vars["protect"].get()), 2)
        return self.settings

    def check_setup(self) -> bool:
        try:
            settings = self._collect()
        except ValueError:
            messagebox.showerror("Invalid setting", "Pitch and strength settings must be numbers.")
            return False
        errors = validate(settings)
        if errors:
            messagebox.showerror("Setup needs attention", "\n\n".join(errors))
            return False
        settings.save()
        messagebox.showinfo("Setup complete", "Piper and RVC are ready.")
        self.notebook.select(1)
        return True

    def generate(self, test: bool = False) -> None:
        text = PITCH_TEST_TEXT if test else self.text.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("No text", "Type something to speak first.")
            return
        try:
            settings = self._collect()
        except ValueError:
            messagebox.showerror("Invalid setting", "Pitch and strength settings must be numbers.")
            return
        errors = validate(settings)
        if errors:
            messagebox.showerror("Setup needs attention", "Open Models & setup, then fix:\n\n" + "\n".join(errors))
            return
        settings.save()
        self.generate_button.configure(state="disabled")
        self.test_button.configure(state="disabled")
        self.progress.start(10)
        self.status.configure(text="Creating a consistent pitch test…" if test else "Piper is generating the base speech…")
        threading.Thread(target=self._run_pipeline, args=(text, settings, test), daemon=True).start()

    def _run_pipeline(self, text: str, settings: Settings, test: bool = False) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        kind = f"pitch-test-{settings.pitch:+d}" if test else "rvc-piper"
        final = OUTPUT_DIR / f"{kind}-{stamp}.wav"
        try:
            with tempfile.TemporaryDirectory(prefix="rvc-piper-") as temp_dir:
                base = Path(temp_dir) / "piper.wav"
                self._execute(piper_command(settings, base), text + "\n", APP_DIR)
                self.events.put(("status", "RVC is applying the voice model…"))
                self._execute(rvc_command(settings, base, final), None, Path(settings.rvc_root))
            if not final.is_file() or final.stat().st_size == 0:
                raise RuntimeError("RVC finished but did not create the expected WAV file.")
            self.events.put(("done", final))
        except Exception as exc:
            self.events.put(("error", str(exc)))

    @staticmethod
    def _execute(command: list[str], stdin: str | None, cwd: Path) -> None:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            command, input=stdin, text=True, encoding="utf-8", errors="replace",
            cwd=cwd, capture_output=True, creationflags=flags, env=environment,
        )
        if result.returncode:
            detail = (result.stderr or result.stdout or "Unknown error").strip()
            raise RuntimeError(detail[-4000:])

    def _poll(self) -> None:
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "status":
                    self.status.configure(text=str(value))
                elif kind == "done":
                    self.last_output = Path(value)
                    self.progress.stop()
                    self.generate_button.configure(state="normal")
                    self.test_button.configure(state="normal")
                    self.play_button.configure(state="normal")
                    self.status.configure(text=f"Finished: {self.last_output.name}")
                    self.play()
                elif kind == "error":
                    self.progress.stop()
                    self.generate_button.configure(state="normal")
                    self.test_button.configure(state="normal")
                    self.status.configure(text="Generation failed")
                    messagebox.showerror("Could not generate audio", str(value))
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def play(self) -> None:
        if self.last_output and self.last_output.exists():
            self._open(self.last_output)

    @staticmethod
    def _open(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True) if path.suffix == "" else None
        os.startfile(str(path))


if __name__ == "__main__":
    Studio().mainloop()
