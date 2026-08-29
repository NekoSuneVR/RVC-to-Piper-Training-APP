#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="${1:-/content/RVC-to-Piper-Training-APP}"
RUNTIME_ROOT="${RVC_PIPER_COLAB_RUNTIME:-/content/rvc-piper-runtime}"
RVC_ROOT="${RUNTIME_ROOT}/rvc"
PIPER_ROOT="${RUNTIME_ROOT}/piper1-gpl"
RVC_VENV="${RUNTIME_ROOT}/rvc-venv"
PIPER_VENV="${RUNTIME_ROOT}/piper-venv"
BASE_DIR="${RUNTIME_ROOT}/base-voice"

echo "=== RVC + Piper Studio: Google Colab setup ==="
echo "App:     ${APP_ROOT}"
echo "Runtime: ${RUNTIME_ROOT}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: No NVIDIA GPU runtime detected. In Colab choose Runtime > Change runtime type > GPU." >&2
  exit 2
fi

nvidia-smi || true
python3 - <<'PY'
import torch
print("python torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access the Colab GPU.")
print("gpu:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
PY

export DEBIAN_FRONTEND=noninteractive
apt-get -qq update
apt-get -qq install -y \
  git build-essential python3-venv cmake ninja-build ffmpeg libsndfile1 >/dev/null

mkdir -p "${RUNTIME_ROOT}" "${BASE_DIR}"

if [ ! -d "${RVC_ROOT}/.git" ]; then
  git clone --depth 1 https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git "${RVC_ROOT}"
else
  git -C "${RVC_ROOT}" pull --ff-only
fi

if [ ! -d "${PIPER_ROOT}/.git" ]; then
  git clone --depth 1 https://github.com/OHF-Voice/piper1-gpl.git "${PIPER_ROOT}"
else
  git -C "${PIPER_ROOT}" pull --ff-only
fi

if [ ! -x "${RVC_VENV}/bin/python" ]; then
  python3 -m venv --system-site-packages "${RVC_VENV}"
fi
if [ ! -x "${PIPER_VENV}/bin/python" ]; then
  python3 -m venv --system-site-packages "${PIPER_VENV}"
fi

RVC_PY="${RVC_VENV}/bin/python"
PIPER_PY="${PIPER_VENV}/bin/python"

echo
echo "=== Installing RVC inference environment ==="
"${RVC_PY}" -m pip install -q --upgrade pip 'setuptools<81' wheel

PY_MINOR="$("${RVC_PY}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [ "${PY_MINOR}" = "3.12" ] && [ -f "${RVC_ROOT}/requirments_cu128_py312.txt" ]; then
  "${RVC_PY}" -m pip install -q -r "${RVC_ROOT}/requirments_cu128_py312.txt"
else
  echo "Colab Python is ${PY_MINOR}; installing the headless RVC inference dependency set."
  "${RVC_PY}" -m pip install -q \
    'numpy>=1.26.4,<2' 'scipy>=1.13.1,<2' 'librosa>=0.10.2,<0.11' \
    'soundfile>=0.13,<1' 'faiss-cpu>=1.9,<2' 'praat-parselmouth>=0.4.5,<1' \
    'PyYAML>=6' 'scikit-learn>=1.6,<2' 'torchfcpe>=0.0.4,<0.1' \
    'transformers>=4.49,<4.50' 'ffmpeg-python>=0.2,<1' 'av>=14,<16' \
    'einops>=0.8,<1' 'local-attention>=1.11,<2' 'huggingface_hub>=0.24'
fi

mkdir -p "${RVC_ROOT}/assets/rmvpe"
if [ ! -s "${RVC_ROOT}/assets/rmvpe/rmvpe.pt" ]; then
  "${RVC_PY}" - <<PY
import urllib.request
url = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt?download=true"
dst = r"${RVC_ROOT}/assets/rmvpe/rmvpe.pt"
print("Downloading RMVPE...")
urllib.request.urlretrieve(url, dst)
PY
fi

if [ ! -f "${RVC_ROOT}/assets/hubert_base/config.json" ]; then
  "${RVC_PY}" - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="lj1995/VoiceConversionWebUI",
    allow_patterns=["hubert_base/*"],
    local_dir=r"${RVC_ROOT}/assets",
)
PY
fi

echo
echo "=== Installing Piper training/inference environment ==="
"${PIPER_PY}" -m pip install -q --upgrade pip 'setuptools<82' wheel scikit-build 'cmake>=3.26,<4' ninja 'cython>=3,<4'
"${PIPER_PY}" -m pip install -q -e "${PIPER_ROOT}[train]"

echo "Building Piper eSpeak bridge..."
(
  cd "${PIPER_ROOT}"
  "${PIPER_PY}" setup.py build_ext --inplace
)

echo "Building Piper monotonic alignment extension..."
ALIGN_DIR="${PIPER_ROOT}/src/piper/train/vits/monotonic_align"
(
  cd "${ALIGN_DIR}"
  mkdir -p monotonic_align
  rm -f core.c core*.so monotonic_align/core*.so
  "${PIPER_VENV}/bin/cythonize" -i core.pyx
  mv core*.so monotonic_align/
)

echo
echo "=== Downloading base Piper voice ==="
"${PIPER_PY}" - <<PY
from pathlib import Path
import urllib.request

root = Path(r"${BASE_DIR}")
root.mkdir(parents=True, exist_ok=True)
files = {
    "en_GB-alba-medium.onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alba/medium/en_GB-alba-medium.onnx?download=true",
    "en_GB-alba-medium.onnx.json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alba/medium/en_GB-alba-medium.onnx.json?download=true",
}
for name, url in files.items():
    path = root / name
    if not path.is_file() or path.stat().st_size == 0:
        print("Downloading", name)
        urllib.request.urlretrieve(url, path)
PY

echo
echo "=== Verification ==="
"${RVC_PY}" -W ignore - <<PY
import torch
print("RVC torch:", torch.__version__)
print("RVC CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
import librosa, soundfile, faiss
print("RVC imports: OK")
PY

"${PIPER_PY}" -W ignore - <<PY
import torch
import piper.train
from piper.phonemize_espeak import EspeakPhonemizer
p = EspeakPhonemizer()
assert p.phonemize("en-GB-x-rp", "Colab Piper verification.")
from piper.train.vits.monotonic_align.monotonic_align.core import maximum_path_c
print("Piper torch:", torch.__version__)
print("Piper CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("eSpeak bridge: OK")
print("monotonic alignment: OK")
PY

cat > "${RUNTIME_ROOT}/paths.env" <<EOF
APP_ROOT=${APP_ROOT}
RUNTIME_ROOT=${RUNTIME_ROOT}
RVC_ROOT=${RVC_ROOT}
RVC_PY=${RVC_PY}
PIPER_ROOT=${PIPER_ROOT}
PIPER_PY=${PIPER_PY}
BASE_MODEL=${BASE_DIR}/en_GB-alba-medium.onnx
BASE_CONFIG=${BASE_DIR}/en_GB-alba-medium.onnx.json
EOF

echo
echo "Colab runtime is ready."
echo "RVC Python:   ${RVC_PY}"
echo "Piper Python: ${PIPER_PY}"
echo "Base voice:   ${BASE_DIR}/en_GB-alba-medium.onnx"
