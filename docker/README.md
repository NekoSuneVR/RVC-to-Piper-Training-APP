# RVC → Piper Docker GPU Trainer

This Docker workflow runs the headless RVC → Piper builder without Google Colab or Google Drive.

The container contains CUDA/PyTorch, RVC, RMVPE/HuBERT, Piper and the Studio training code. Large and persistent files live on the host under `docker-data/` and are mounted into the container at `/data`.

## Persistent layout

```text
docker-data/
├── models/
│   ├── Neeko.pth
│   └── Neeko.index
├── cache/
│   └── base-voices/          # downloaded Piper base voices
├── work/
│   ├── _checkpoints/         # warm-start checkpoint
│   └── <voice-name>/
│       ├── dataset/
│       └── piper/
│           └── training/     # Lightning checkpoints
└── output/
    └── <voice-name>/
        ├── <voice-name>.onnx
        └── <voice-name>.onnx.json
```

Deleting/rebuilding the Docker image does not delete `docker-data/`.

## Requirements

- Docker Engine or Docker Desktop with Compose v2
- An NVIDIA GPU exposed to Docker
- NVIDIA Container Toolkit on Linux, or a Docker Desktop configuration that supports NVIDIA GPU passthrough
- Plenty of local disk space. The image plus CUDA/PyTorch/RVC/Piper assets is large, and training checkpoints/datasets add more space. Around 30 GB free is a practical minimum; more is better for long training runs.

Check GPU access first:

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

If this fails, fix Docker GPU access before building the trainer.

## Quick start

From the repository root:

```bash
cp docker.env.example docker.env
mkdir -p docker-data/models
```

Copy your RVC model and optional index into `docker-data/models/`, for example:

```text
docker-data/models/Neeko.pth
docker-data/models/Neeko.index
```

Edit `docker.env` as needed. The important defaults are:

```env
VOICE_NAME=en_US-nekoai-medium
RVC_MODEL=models/Neeko.pth
RVC_INDEX=models/Neeko.index
BASE_PIPER_VOICE=en_GB-alba-medium
PITCH=12
BATCH_SIZE=8
MAX_EPOCHS=1000
```

Then build the image:

```bash
docker compose -f docker-compose.gpu.yml build
```

Run the complete pipeline:

```bash
docker compose -f docker-compose.gpu.yml run --rm trainer
```

The container first checks CUDA, runs a one-sentence Piper + RVC preflight, then generates/resumes the dataset, cleans high-pitch audio, fine-tunes Piper, exports the best checkpoint and copies the final ONNX pair to `docker-data/output/<voice-name>/`.

## Windows PowerShell

```powershell
Copy-Item docker.env.example docker.env
New-Item -ItemType Directory -Force docker-data\models
Copy-Item C:\Path\To\Neeko.pth docker-data\models\Neeko.pth
Copy-Item C:\Path\To\Neeko.index docker-data\models\Neeko.index

docker compose -f docker-compose.gpu.yml build
docker compose -f docker-compose.gpu.yml run --rm trainer
```

## Change the base Piper voice

Alba is the default:

```env
BASE_PIPER_VOICE=en_GB-alba-medium
```

Amy:

```env
BASE_PIPER_VOICE=en_US-amy-medium
```

Short aliases also work:

```env
BASE_PIPER_VOICE=alba
BASE_PIPER_VOICE=amy
```

Any key present in Piper's official `voices.json` can be used. The matching ONNX and JSON are downloaded once into `docker-data/cache/base-voices/` and reused on later runs.

For your own base Piper model, place the ONNX/JSON under `docker-data/models/` and set both:

```env
BASE_PIPER_MODEL=models/custom-base.onnx
BASE_PIPER_CONFIG=models/custom-base.onnx.json
```

## Resume after stopping or restarting

Training state is persisted under `docker-data/work/`.

With the default:

```env
NO_RESUME=0
```

running the same Compose command again automatically looks for the newest `last.ckpt` and resumes when possible:

```bash
docker compose -f docker-compose.gpu.yml run --rm trainer
```

To force a new warm-start training run instead:

```env
NO_RESUME=1
```

## Reuse an existing generated dataset

After the dataset exists under `docker-data/work/<voice-name>/dataset`, set:

```env
SKIP_DATASET=1
```

This skips Piper-base synthesis and RVC conversion and goes directly to Piper training/resume.

## Run only the preflight

```bash
docker compose -f docker-compose.gpu.yml run --rm trainer preflight
```

This tests GPU access, the selected base Piper voice, your RVC `.pth`/`.index`, RMVPE and the +12 conversion without starting the full training job.

## Open a shell inside the image

```bash
docker compose -f docker-compose.gpu.yml run --rm trainer shell
```

## Output

After a successful build:

```text
docker-data/output/<voice-name>/<voice-name>.onnx
docker-data/output/<voice-name>/<voice-name>.onnx.json
```

These are ordinary host files and can be copied directly to the Windows Studio or another Piper runtime.
