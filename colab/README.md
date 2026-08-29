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
        │   └── checkpoints/
        │       ├── last.ckpt
        │       ├── resume-latest.ckpt
        │       ├── training-progress.json
        │       └── epoch=...-val_mel=....ckpt
        ├── <voice-name>.onnx
        └── <voice-name>.onnx.json
```

## What the notebook does

1. Resolves Auto/GPU/CPU before installing the runtime.
2. Creates separate Python 3.12 RVC and Piper virtual environments on Colab's local disk.
3. Installs CUDA PyTorch for GPU mode or CPU-only PyTorch for CPU mode.
4. Rebuilds the managed environments automatically if you switch CPU ↔ GPU.
5. Downloads HuBERT, RMVPE and the default Alba Piper base voice; Amy, other Piper voice keys, and custom ONNX + JSON base voices are selectable.
6. Loads the RVC model and converts the generated dataset at the selected pitch.
7. For high shifts such as `+12`, runs the same high-pitch mastering used by the Windows/Docker builder before Piper caches the audio.
8. Fine-tunes Piper from the official medium warm-start checkpoint instead of starting from random weights.
9. Refreshes a **full-state resumable checkpoint every completed epoch**.
10. Prints live epoch/batch/global-step/loss/validation/checkpoint progress to the Colab console.
11. Writes `training-progress.json` beside the checkpoint so you can see the last saved epoch/global step after reconnecting.
12. Resumes from the newest `last.ckpt` when `resume_if_possible = True`.
13. Exports the best `val_mel` checkpoint from the newest run to ONNX + JSON.
14. Provides a final pure-Piper test cell with an inline audio player.

## GPU → CPU continuation

This is supported intentionally. A Piper Lightning checkpoint contains the model parameters, optimizer state, epoch and global step, so the same run can continue on a different device.

Typical workflow:

```text
1. Start Colab in GPU mode.
2. Train normally.
3. After each completed epoch, Drive receives a new last.ckpt.
4. GPU quota/session ends.
5. Open/reconnect Colab with a CPU runtime.
6. Select CPU in the first notebook cell.
7. Rerun repo/setup/settings/base-voice cells.
8. Leave resume_if_possible = True.
9. Run Build everything again.
10. The console should print RESUME CONFIRMED with the restored epoch/global step.
```

A hard Colab termination can happen during an epoch. In that case, training resumes from the most recently **completed and saved epoch**, so only the unfinished epoch can be lost. Saving a full multi-hundred-MB checkpoint after every training batch would be impractical for Google Drive, so epoch-level resume is the default balance between safety, storage and speed.

The notebook's `checkpoint_every` setting controls how often the separate best-`val_mel` checkpoints are considered. It does **not** reduce the once-per-epoch `last.ckpt` resume safety.

## Console progress

The notebook exposes:

```text
console_batch_every = 2
```

Set it to `1` for a line after every training batch. During a run you will see messages similar to:

```text
[EPOCH 14/1000] START global_step=156 device=cuda:0
[train] epoch=14/1000 batch=2/12 step=158 train_mel=... loss_g=... loss_d=... epoch_time=...
[validation] epoch=14/1000 step=168 val_mel=... val_loss=...
[EPOCH 14/1000] COMPLETE global_step=168 epoch_time=... total_time=...
[checkpoint] SAVED resume-latest.ckpt (...) epoch=14 step=168
[checkpoint] SAVED last.ckpt (...) epoch=14 step=168
```

The guarded build runner also writes the overall child-process log to:

```text
/content/rvc-piper-runtime/build.log
```

`training-progress.json` is persistent when your project root is in Drive and includes the most recent epoch, global step, device and selected metrics.

## CPU notes

CPU mode runs the whole path on CPU: Piper base synthesis, RVC/RMVPE conversion, Piper training and ONNX export. It is useful when no Colab GPU is available, but training can be dramatically slower.

For a **fresh** CPU test, start small:

```text
prompt_limit = 10
max_epochs = 1
batch_size = 2
```

If you are **resuming an existing GPU checkpoint**, keep the same voice name, dataset, model settings and maximum epoch target. Do not set `resume_if_possible = False`, because that deliberately starts a fresh warm-start run instead.

## Important notes

- Google Colab controls GPU availability and session limits; the notebook cannot guarantee a specific GPU or uninterrupted runtime.
- Persistent Drive storage survives runtime resets; `/content` storage does not.
- The temporary Python/RVC/Piper environments under `/content` must be rebuilt after a full Colab runtime reset; rerunning the setup cell does that automatically.
- If no RVC `.index` is used, the notebook forces `index_rate` to `0` for that build.
- If CUDA runs out of memory, lower the Piper batch size and rerun. Existing persistent dataset/checkpoints are reused where possible.
- Switching GPU → CPU changes the compute device, not the persistent training project or checkpoint history.
