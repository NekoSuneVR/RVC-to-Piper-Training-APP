# Google Colab support

The Colab path runs the RVC → Piper model builder without Tkinter and can use either a **Colab GPU or CPU**.

Open `RVC_to_Piper_Training_Colab.ipynb` in Google Colab and choose one of the notebook compute modes:

- **Auto** — use GPU when Colab has one attached, otherwise fall back to CPU.
- **GPU** — require CUDA and stop with a clear error if no GPU runtime is attached.
- **CPU** — force CPU-only PyTorch even when a GPU is available.

GPU is strongly recommended for practical Piper training speed, but it is no longer required.

The notebook normally keeps persistent data in Google Drive. You can disable Drive mounting and use `/content` paths instead, but `/content` is temporary and is lost when the Colab runtime is deleted.

Recommended Drive layout:

```text
MyDrive/RVC-Piper-Colab/
├── models/
│   ├── my-voice.pth
│   └── my-voice.index        # optional
├── _checkpoints/
│   └── en_US-lessac-medium.ckpt
└── <voice-name>/
    ├── dataset/
    │   ├── metadata.csv
    │   └── audio/
    └── piper/
        ├── training/
        ├── <voice-name>.onnx
        └── <voice-name>.onnx.json
```

## What the notebook does

1. Resolves Auto/GPU/CPU before installing the runtime.
2. Creates separate Python 3.12 RVC and Piper virtual environments on Colab's local disk.
3. Installs CUDA PyTorch for GPU mode or CPU-only PyTorch for CPU mode.
4. Rebuilds the managed environments automatically if you switch CPU ↔ GPU in the same Colab VM.
5. Downloads HuBERT, RMVPE and the default Alba Piper base voice; Amy, other Piper voice keys, and custom ONNX + JSON base voices are selectable in the notebook.
6. Loads the RVC model and converts the generated dataset at the selected pitch.
7. For high shifts such as `+12`, runs the same high-pitch mastering used by the Windows/Docker builder before Piper caches the audio.
8. Fine-tunes Piper from the official medium warm-start checkpoint instead of starting from random weights.
9. Writes reduced-frequency resumable checkpoints to the selected persistent project folder.
10. Resumes from the newest `last.ckpt` after a reconnect when available.
11. Exports the best `val_mel` checkpoint from the newest run to ONNX + JSON.
12. Provides a final pure-Piper test cell with an inline audio player.

## CPU notes

CPU mode runs the whole path on CPU: Piper base synthesis, RVC/RMVPE conversion, Piper training and ONNX export. It is useful when no Colab GPU is available, but training can be dramatically slower.

Before starting a long CPU job, test with something small such as:

```text
prompt_limit = 10
max_epochs = 1
batch_size = 2
```

Then return to your real prompt/epoch settings once the complete pipeline reaches ONNX export.

## Important notes

- Google Colab controls GPU availability and session limits; the notebook cannot guarantee a specific GPU or uninterrupted runtime.
- Persistent Drive storage survives runtime resets; `/content` storage does not.
- The temporary Python/RVC/Piper environments under `/content` must be rebuilt after a full Colab runtime reset; rerunning the setup cell does that automatically.
- If no RVC `.index` is used, the notebook forces `index_rate` to `0` for that build.
- If CUDA runs out of memory, lower the Piper batch size and rerun. Existing persistent dataset/checkpoints are reused where possible.
