# RVC + Piper Studio

A local Windows app that turns text into speech with Piper, then applies an RVC voice model.

New user? Start with **FIRST-VOICE-GUIDE.md** for the exact files to obtain and a pitch-tuning walkthrough.

## Quick start

1. Double-click **Start RVC Piper Studio.cmd**. On first launch it automatically installs the CPU-compatible runtime, Piper, RVC, and a British-English medium Piper voice. This is a large one-time download.
2. Open the **Models & setup** tab.
3. The included Piper voice is already selected. Select your RVC `.pth` model. An RVC `.index` is optional but usually improves voice similarity.
4. Click **Save and check setup**, return to **Create audio**, and generate.

The Create tab includes a large pitch slider from **-24 to +24 semitones**, quick octave buttons, voice-similarity strength, consonant protection, and pitch-detection selection. `0` keeps the original Piper pitch; negative values deepen the voice and positive values raise it.

Use **Generate pitch test** after each adjustment. It always speaks the same pronunciation-rich phrase and names files with the chosen pitch (for example, `pitch-test--4-...wav`), making comparisons easy.

For NVIDIA setup, open PowerShell in this folder and run:

```powershell
.\setup.ps1 -Engine cuda
```

Generated WAV files are saved under `generated`. Audio, text, and models never leave the PC.

## What “conversion” means

RVC and Piper use different neural-network architectures. An RVC `.pth` checkpoint cannot be changed into a native Piper `.onnx` voice without a proper Piper training dataset and retraining. This app instead provides the useful all-in-one workflow:

`text → Piper base speech → RVC voice conversion → WAV`

## Requirements and notes

- Windows 10 or 11, 64-bit
- Several GB of free disk space; RVC and PyTorch are large
- Only use voice models you have permission to use
- The original Piper repository is archived; this app uses its stable Windows runtime and keeps it isolated under `tools/piper`.

If setup is interrupted, run **Easy Setup.cmd** again. Existing downloads are reused where possible.
