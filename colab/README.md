# Google Colab support

The Colab path runs the RVC → Piper model builder without Tkinter and keeps persistent data in Google Drive.

Open `RVC_to_Piper_Training_Colab.ipynb` in Google Colab, select a GPU runtime, mount Drive, and run the notebook from top to bottom.

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

1. Creates separate RVC and Piper virtual environments on Colab's local disk.
2. Downloads HuBERT, RMVPE and the neutral British-English Piper base voice.
3. Loads the RVC model once and converts the generated dataset at the selected pitch.
4. For high shifts such as `+12`, runs the same high-pitch mastering used by the Windows builder before Piper caches the audio.
5. Fine-tunes Piper from the official medium warm-start checkpoint instead of starting from random weights.
6. Writes reduced-frequency resumable checkpoints to Google Drive.
7. Resumes from the newest `last.ckpt` after a Colab reconnect when available.
8. Exports the best `val_mel` checkpoint from the newest run to ONNX + JSON.
9. Provides a final pure-Piper test cell with an inline audio player.

## Important notes

- Google Colab controls GPU availability and session limits; the notebook cannot guarantee a specific GPU or uninterrupted runtime.
- Checkpoints and final models are stored in Drive so a runtime reset does not lose them.
- The temporary Python/RVC/Piper environments under `/content` must be rebuilt after a full Colab runtime reset; rerunning the setup cell does that automatically.
- If no RVC `.index` is used, set `index_rate` to `0` in the notebook.
- If CUDA runs out of memory, lower the Piper batch size and rerun. The Drive dataset/checkpoints remain available.
