from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from app import APP_DIR, Settings, Studio as BaseStudio


TRAINING_ROOT = APP_DIR / "training"
TRAINER_PYTHON = APP_DIR / "tools" / "piper-trainer" / ".venv" / "Scripts" / "python.exe"
PROMPTS_FILE = APP_DIR / "data" / "piper_training_prompts.txt"
BUILDER_LAUNCHER = APP_DIR / "builder_launcher.py"


class Studio(BaseStudio):
    """Main Studio with the Piper model-building workflow exposed as tab 3."""

    def _build_ui(self) -> None:
        super()._build_ui()
        self._build_piper_model_tab()

    def _build_piper_model_tab(self) -> None:
        builder = ttk.Frame(self.notebook, style="Card.TFrame", padding=20)
        self.notebook.add(builder, text="  3  BUILD PIPER MODEL  ")
        self.builder_tab = builder

        ttk.Label(builder, text="Build a standalone Piper voice", style="Section.TLabel").pack(anchor="w")
        ttk.Label(
            builder,
            text=(
                "Turn your selected RVC voice into a Piper training dataset, train Piper, then export a native "
                ".onnx voice and matching .onnx.json config."
            ),
            style="CardHint.TLabel",
            wraplength=760,
            justify="left",
        ).pack(anchor="w", pady=(3, 14))

        readiness = ttk.LabelFrame(builder, text=" Builder readiness ", padding=14)
        readiness.pack(fill="x")
        self.builder_readiness = ttk.Label(
            readiness,
            text="Checking current setup…",
            style="Card.TLabel",
            justify="left",
            wraplength=760,
        )
        self.builder_readiness.pack(side="left", fill="x", expand=True, anchor="w")
        ttk.Button(
            readiness,
            text="Refresh",
            style="Secondary.TButton",
            command=self._refresh_builder_readiness,
        ).pack(side="right", padx=(12, 0))

        steps = ttk.Frame(builder, style="Card.TFrame")
        steps.pack(fill="x", pady=(14, 0))
        for column in range(3):
            steps.columnconfigure(column, weight=1, uniform="builder-step")
        self._builder_step_card(
            steps,
            0,
            "1 · Dataset",
            "Piper speaks each training phrase, then RVC converts it into your selected voice. Existing completed lines are reused.",
            "Build RVC dataset",
            "dataset",
        )
        self._builder_step_card(
            steps,
            1,
            "2 · Train",
            "Train a real Piper VITS voice from the generated WAV files and metadata. Training can take a long time.",
            "Train Piper",
            "train",
        )
        self._builder_step_card(
            steps,
            2,
            "3 · Export",
            "Find the newest Piper checkpoint and export the final .onnx model plus its matching JSON configuration.",
            "Export ONNX",
            "export",
        )

        primary = ttk.LabelFrame(builder, text=" One-click build ", padding=14)
        primary.pack(fill="x", pady=(14, 0))
        ttk.Button(
            primary,
            text="▶  BUILD EVERYTHING",
            style="Primary.TButton",
            command=lambda: self._launch_builder("all"),
        ).pack(side="left")
        ttk.Label(
            primary,
            text="Runs dataset generation → training → ONNX export in order. A builder window opens with live logs and Stop control.",
            style="CardHint.TLabel",
            wraplength=560,
            justify="left",
        ).pack(side="left", padx=12)

        tools = ttk.Frame(builder, style="Card.TFrame")
        tools.pack(fill="x", pady=(14, 0))
        ttk.Button(
            tools,
            text="Install / repair Piper trainer",
            style="Secondary.TButton",
            command=lambda: self._launch_builder("install"),
        ).pack(side="left")
        ttk.Button(
            tools,
            text="Open full builder settings",
            style="Secondary.TButton",
            command=lambda: self._launch_builder("open"),
        ).pack(side="left", padx=6)
        ttk.Button(
            tools,
            text="Open training folder",
            style="Secondary.TButton",
            command=lambda: self._open(TRAINING_ROOT),
        ).pack(side="left", padx=6)

        note = ttk.Frame(builder, style="Card.TFrame")
        note.pack(fill="x", pady=(16, 0))
        ttk.Label(note, text="Important", style="Step.TLabel").pack(anchor="w")
        ttk.Label(
            note,
            text=(
                "The current RVC model, RVC index, pitch, similarity and protection settings saved in Models & Setup are used when "
                "creating the training WAV files. Test the voice first, save your settings, then build the dataset."
            ),
            style="CardHint.TLabel",
            wraplength=780,
            justify="left",
        ).pack(anchor="w", pady=(4, 0))

        self._refresh_builder_readiness()
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed, add="+")

    def _builder_step_card(self, parent, column: int, title: str, detail: str, button: str, action: str) -> None:
        card = ttk.LabelFrame(parent, text=f" {title} ", padding=12)
        card.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 5, 0 if column == 2 else 5))
        ttk.Label(
            card,
            text=detail,
            style="CardHint.TLabel",
            wraplength=220,
            justify="left",
        ).pack(anchor="w", fill="x", expand=True)
        ttk.Button(
            card,
            text=button,
            style="Secondary.TButton",
            command=lambda name=action: self._launch_builder(name),
        ).pack(anchor="w", pady=(12, 0))

    def _on_tab_changed(self, _event=None) -> None:
        try:
            if self.notebook.select() == str(self.builder_tab):
                self._refresh_builder_readiness()
        except tk.TclError:
            pass

    def _refresh_builder_readiness(self) -> None:
        settings = Settings.load()
        rvc_model = Path(settings.rvc_model) if settings.rvc_model else None
        rvc_ready = bool(rvc_model and rvc_model.is_file())
        index_ready = bool(settings.rvc_index and Path(settings.rvc_index).is_file())
        trainer_ready = TRAINER_PYTHON.is_file()
        prompts_ready = PROMPTS_FILE.is_file()

        lines = [
            f"{'✓' if rvc_ready else '✗'} RVC voice model: {rvc_model if rvc_model else 'not selected'}",
            f"{'✓' if index_ready else '•'} RVC feature index: {settings.rvc_index if settings.rvc_index else 'optional / not selected'}",
            f"{'✓' if trainer_ready else '✗'} Piper training environment: {'installed' if trainer_ready else 'not installed yet'}",
            f"{'✓' if prompts_ready else '✗'} Training prompts: {PROMPTS_FILE}",
        ]
        if rvc_ready and trainer_ready and prompts_ready:
            lines.append("\nReady to build. You can use BUILD EVERYTHING.")
        elif rvc_ready and prompts_ready:
            lines.append("\nInstall the Piper trainer first, then build.")
        else:
            lines.append("\nFinish Models & Setup first, then return here.")
        self.builder_readiness.configure(text="\n".join(lines))

    def _launch_builder(self, action: str) -> None:
        if not BUILDER_LAUNCHER.is_file():
            messagebox.showerror("Builder missing", f"Could not find {BUILDER_LAUNCHER}")
            return

        if action in {"dataset", "train", "export", "all"}:
            try:
                self._collect().save()
            except (ValueError, tk.TclError) as exc:
                messagebox.showerror("Invalid Studio setting", str(exc))
                return

        try:
            subprocess.Popen(
                [sys.executable, str(BUILDER_LAUNCHER), "--action", action],
                cwd=str(APP_DIR),
            )
        except OSError as exc:
            messagebox.showerror("Could not open Piper builder", str(exc))
            return

        if action == "open":
            self.status.configure(text="Piper model builder opened.")
        elif action == "install":
            self.status.configure(text="Piper trainer setup opened. Follow the builder log.")
        else:
            self.status.configure(text=f"Piper builder started: {action}. Follow the builder window for progress.")


if __name__ == "__main__":
    Studio().mainloop()
