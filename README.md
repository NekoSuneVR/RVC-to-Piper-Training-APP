# RVC + Piper Studio

A local Windows app that turns text into speech with Piper, applies an RVC voice model, and can now build a native Piper voice from the RVC result.

New user? Start with **FIRST-VOICE-GUIDE.md** for the exact files to obtain and a pitch-tuning walkthrough.

## Quick start

1. Double-click **Start RVC Piper Studio.cmd**. On first launch it automatically installs the CPU-compatible runtime, Piper, RVC, and a British-English medium Piper voice. This is a large one-time download.
2. Open the **Models & setup** tab.
3. The included Piper voice is already selected. Select your RVC `.pth` model. An RVC `.index` is optional but usually improves voice similarity.
4. Click **Save and check setup**, return to **Create audio**, and generate.

The Create tab includes a large pitch slider from **-24 to +24 semitones**, quick octave buttons, voice-similarity strength, consonant protection, and pitch-detection selection. `0` keeps the original Piper pitch; negative values deepen the voice and positive values raise it.

Use **Generate pitch test** after each adjustment. It always speaks the same pronunciation-rich phrase and names files with the chosen pitch (for example, `pitch-test--4-...wav`), making comparisons easy.

For NVIDIA RVC inference setup, open PowerShell in this folder and run:

```powershell
.\setup.ps1 -Engine cuda
```

Generated WAV files are saved under `generated`. Audio, text, and models never leave the PC.

## Build a real Piper model from the RVC voice

The pitch test proves the Piper → RVC inference path works, but it is not a Piper model. **Build Piper Model.cmd** adds the missing training stage:

`training text → base Piper WAV → RVC-converted WAV dataset → Piper training → checkpoint → ONNX + ONNX JSON`

### 1. Install the Piper trainer

Run **Build Piper Model.cmd** and click **Install / repair trainer**. The training environment is isolated under `tools/piper-trainer` and uses the current Open Home Foundation Piper trainer.

Piper training needs the Microsoft C++ build tools because its monotonic alignment extension is compiled locally. If they are not installed, either install **Visual Studio 2022 Build Tools** with **Desktop development with C++**, or run:

```powershell
.\setup-piper-training.ps1 -InstallBuildTools
```

The trainer setup defaults to CUDA. For CPU-only training use:

```powershell
.\setup-piper-training.ps1 -Engine cpu
```

### 2. Build the RVC training dataset

The builder reads the Piper and RVC model choices already saved by Studio. It includes `data/piper_training_prompts.txt` as a starter prompt list.

Click **1. Build RVC dataset**. For every prompt the builder:

1. generates neutral base speech with Piper;
2. converts that WAV through the selected RVC `.pth` / `.index`;
3. stores the converted WAV under `training/<voice>/dataset/audio`;
4. writes Piper-compatible `metadata.csv` entries.

Dataset generation is resumable. Existing WAV files are reused when their matching text has already been generated.

The bundled prompt set is useful for testing the pipeline. For a better final model, provide a much larger clean text corpus with one sentence per line. More varied, correctly pronounced training audio generally gives Piper more useful material to learn from.

### 3. Train Piper

Set the voice name, eSpeak language, device, batch size, and maximum epochs, then click **2. Train Piper**.

A Piper medium warm-start `.ckpt` is optional in the UI but strongly recommended. Piper's own training documentation recommends fine-tuning from an existing checkpoint because it substantially speeds up training.

Training checkpoints and caches are kept under:

```text
training/<voice>/piper/
```

Use **Stop** if you need to end the current build. Checkpoints already written by Piper remain available for a later export/resume workflow.

### 4. Export the voice

Click **3. Export ONNX**. The builder finds the newest `.ckpt` and runs Piper's ONNX exporter.

The final files are:

```text
training/<voice>/piper/<voice>.onnx
training/<voice>/piper/<voice>.onnx.json
```

Keep **Use exported Piper model in Studio automatically** checked if you want those two files selected as Studio's active Piper voice after export.

You can also click **Build everything** to run dataset generation, training, and export in one sequence after the trainer is installed.

## What “conversion” means

RVC and Piper use different neural-network architectures. An RVC `.pth` checkpoint cannot simply be renamed or directly converted into a Piper `.onnx` file. A native Piper voice requires a Piper-compatible text/audio dataset and Piper training.

The project now supports both workflows:

```text
Fast preview:
text → Piper base speech → RVC voice conversion → WAV

Native Piper build:
text corpus → Piper base speech → RVC dataset → Piper training → .onnx + .onnx.json
```

## Requirements and notes

- Windows 10 or 11, 64-bit
- Several GB of free disk space; RVC, Piper training, PyTorch, datasets, caches, and checkpoints can become large
- NVIDIA GPU recommended for practical training speed; CPU training is supported by the setup script but can be very slow
- Piper's official docs report training success with GPUs around 8 GB VRAM, while larger GPUs give more headroom
- Only use voice models and voices you have permission to use
- `training/`, `models/`, `tools/`, generated audio, and local settings are ignored by Git
- The stable standalone Piper runtime remains isolated under `tools/piper`; model training uses the maintained `OHF-Voice/piper1-gpl` trainer in its own environment

If setup is interrupted, run **Easy Setup.cmd** or the relevant Piper trainer setup again. Existing downloads and generated dataset files are reused where possible.

## Tests

Run the command/helper tests with:

```powershell
.\tools\python\python.exe -m unittest discover -s tests -v
```
