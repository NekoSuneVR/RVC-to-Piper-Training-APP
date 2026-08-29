# RVC → Piper Docker Trainer

This Docker workflow runs the headless RVC → Piper builder without Google Colab or Google Drive.

Two images are provided:

- `Dockerfile.cpu` + `docker-compose.cpu.yml` for Linux servers with no GPU.
- `Dockerfile.gpu` + `docker-compose.gpu.yml` for NVIDIA GPU hosts.

The container contains RVC, RMVPE/HuBERT, Piper and the Studio training code. Large/persistent files live on the host under `docker-data/` and are mounted at `/data`.

## Persistent layout

```text
docker-data/
├── models/
│   ├── Neeko.pth
│   └── Neeko.index
├── cache/
│   └── base-voices/
├── work/
│   ├── _checkpoints/
│   └── <voice-name>/
│       ├── dataset/
│       └── piper/
│           └── training/
└── output/
    └── <voice-name>/
        ├── <voice-name>.onnx
        └── <voice-name>.onnx.json
```

Deleting or rebuilding the Docker image does not delete `docker-data/`.

## Common setup

From the repository root:

```bash
cp docker.env.example docker.env
mkdir -p docker-data/models
```

Copy the RVC model and optional index into `docker-data/models/`, then edit `docker.env`:

```env
VOICE_NAME=en_US-nekoai-medium
RVC_MODEL=models/Neeko.pth
RVC_INDEX=models/Neeko.index
BASE_PIPER_VOICE=en_US-amy-medium
PITCH=12
INDEX_RATE=0.75
PROTECT=0.33
F0_METHOD=rmvpe
BATCH_SIZE=8
MAX_EPOCHS=1000
RUN_PREFLIGHT=1
NO_RESUME=0
```

## CPU-only Linux server

No NVIDIA driver or NVIDIA Container Toolkit is required.

Build:

```bash
docker compose -f docker-compose.cpu.yml build
```

Run the one-sentence preflight only:

```bash
docker compose -f docker-compose.cpu.yml run --rm trainer preflight
```

Run the full pipeline:

```bash
docker compose -f docker-compose.cpu.yml run --rm trainer
```

The CPU image uses CPU-only PyTorch. Both RVC dataset conversion and Piper training run on the CPU.

CPU training is supported but can be extremely slow. The persistent checkpoint layout is important: if the process is stopped after `last.ckpt` exists, run the same command again with `NO_RESUME=0` to resume.

For a quick validation before committing to a very long CPU job, temporarily use something such as:

```env
PROMPT_LIMIT=10
MAX_EPOCHS=1
BATCH_SIZE=2
```

Once that completes successfully, restore the real prompt/epoch values. Do not reuse the tiny validation dataset for the final model unless that is intentional; either remove the test voice folder under `docker-data/work/` or use a different `VOICE_NAME` for the test.

## NVIDIA GPU server

Requirements:

- Docker Engine / Compose v2
- NVIDIA driver
- NVIDIA Container Toolkit

Check GPU passthrough:

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

Build and run:

```bash
docker compose -f docker-compose.gpu.yml build
docker compose -f docker-compose.gpu.yml run --rm trainer
```

## What the container does

The build performs:

```text
selected Piper base voice
→ base speech
→ RVC conversion
→ pitch (+12 is supported)
→ persistent dataset
→ high-pitch cleanup
→ Piper warm-start
→ CPU or GPU training
→ best val_mel checkpoint
→ ONNX export
```

Final files are copied to:

```text
docker-data/output/<voice-name>/<voice-name>.onnx
docker-data/output/<voice-name>/<voice-name>.onnx.json
```

## Resume

Keep:

```env
NO_RESUME=0
```

and rerun the same Compose command. The builder searches the persistent training folder for the newest `last.ckpt`.

Dataset WAVs are also persistent, so completed RVC conversions do not need to be regenerated when their metadata still matches.

## Storage

CPU mode removes the large CUDA base image, but RVC, Piper, PyTorch, the warm-start checkpoint, datasets and training checkpoints still use several GB. Keep plenty of free host disk space and mount `docker-data` on a larger disk if necessary.
