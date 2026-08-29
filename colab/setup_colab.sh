#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="${1:-/content/RVC-to-Piper-Training-APP}"
RUNTIME_ROOT="${RVC_PIPER_COLAB_RUNTIME:-/content/rvc-piper-runtime}"
RVC_ROOT="${RUNTIME_ROOT}/rvc"
PIPER_ROOT="${RUNTIME_ROOT}/piper1-gpl"
RVC_VENV="${RUNTIME_ROOT}/rvc-venv"
PIPER_VENV="${RUNTIME_ROOT}/piper-venv"
BASE_DIR="${RUNTIME_ROOT}/base-voice"
LOG_FILE="${RUNTIME_ROOT}/setup.log"
MODE_FILE="${RUNTIME_ROOT}/compute.mode"
TARGET_PYTHON="3.12"
TORCH_VERSION="2.7.1"
TORCH_CUDA="${RVC_PIPER_TORCH_CUDA:-cu126}"
REQUESTED_COMPUTE="${RVC_PIPER_COMPUTE:-auto}"

mkdir -p "${RUNTIME_ROOT}"
exec > >(tee -a "${LOG_FILE}") 2>&1

setup_failed() {
  code=$?
  if [ "${code}" -ne 0 ]; then
    echo
    echo "============================================================"
    echo "COLAB SETUP FAILED (exit ${code})"
    echo "Full log: ${LOG_FILE}"
    echo "The failing command is immediately above this message."
    echo "============================================================"
  fi
  exit "${code}"
}
trap setup_failed EXIT

case "${REQUESTED_COMPUTE}" in
  auto|gpu|cpu) ;;
  *)
    echo "ERROR: RVC_PIPER_COMPUTE must be auto, gpu, or cpu (got: ${REQUESTED_COMPUTE})." >&2
    exit 2
    ;;
esac

HAS_NVIDIA=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  HAS_NVIDIA=1
fi

if [ "${REQUESTED_COMPUTE}" = "auto" ]; then
  if [ "${HAS_NVIDIA}" = "1" ]; then
    COMPUTE="gpu"
  else
    COMPUTE="cpu"
  fi
else
  COMPUTE="${REQUESTED_COMPUTE}"
fi

if [ "${COMPUTE}" = "gpu" ] && [ "${HAS_NVIDIA}" != "1" ]; then
  echo "ERROR: GPU mode was selected but no NVIDIA GPU is attached to this Colab runtime." >&2
  echo "Choose Runtime > Change runtime type > GPU, or select CPU/Auto in the notebook." >&2
  exit 2
fi

if [ "${COMPUTE}" = "gpu" ]; then
  TORCH_FLAVOR="${TORCH_CUDA}"
  TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_CUDA}"
  TORCH_SPEC="${TORCH_VERSION}+${TORCH_CUDA}"
else
  TORCH_FLAVOR="cpu"
  TORCH_INDEX="https://download.pytorch.org/whl/cpu"
  TORCH_SPEC="${TORCH_VERSION}+cpu"
fi

echo "=== RVC + Piper Studio: Google Colab setup ==="
echo "App:              ${APP_ROOT}"
echo "Runtime:          ${RUNTIME_ROOT}"
echo "Colab Python:     $(python3 --version 2>&1)"
echo "Managed Python:   ${TARGET_PYTHON}"
echo "Requested mode:   ${REQUESTED_COMPUTE}"
echo "Resolved mode:    ${COMPUTE}"
echo "PyTorch wheel:    ${TORCH_SPEC}"
echo

if [ "${COMPUTE}" = "gpu" ]; then
  nvidia-smi || true
else
  echo "CPU mode selected; NVIDIA GPU is not required."
fi

python3 - <<'PY'
try:
    import torch
    print("Colab base torch:", torch.__version__)
    print("Colab base CUDA:", torch.version.cuda)
    print("Colab base CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Colab base GPU:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("Colab base torch probe unavailable:", exc)
PY

export DEBIAN_FRONTEND=noninteractive
apt-get -qq update
apt-get -qq install -y \
  git build-essential cmake ninja-build ffmpeg libsndfile1 pkg-config >/dev/null

# Colab's system Python changes independently of RVC/Piper. Keep a managed
# Python 3.12 runtime so the pinned NumPy/SciPy/RVC stack stays reproducible.
echo
echo "=== Preparing managed Python ${TARGET_PYTHON} ==="
python3 -m pip install -q --upgrade uv
UV_BIN="$(command -v uv || true)"
if [ -z "${UV_BIN}" ]; then
  echo "ERROR: uv was installed but its executable is not on PATH." >&2
  exit 3
fi
export UV_PYTHON_INSTALL_DIR="${RUNTIME_ROOT}/uv-python"
"${UV_BIN}" python install "${TARGET_PYTHON}"

# Switching CPU <-> GPU in the same Colab VM must not reuse an environment
# containing the other PyTorch wheel.
if [ -f "${MODE_FILE}" ]; then
  PREVIOUS_MODE="$(cat "${MODE_FILE}" 2>/dev/null || true)"
  if [ -n "${PREVIOUS_MODE}" ] && [ "${PREVIOUS_MODE}" != "${COMPUTE}" ]; then
    echo "Compute mode changed ${PREVIOUS_MODE} -> ${COMPUTE}; rebuilding RVC/Piper virtual environments."
    rm -rf "${RVC_VENV}" "${PIPER_VENV}"
  fi
fi

ensure_py312_venv() {
  local venv="$1"
  if [ -x "${venv}/bin/python" ]; then
    local version
    version="$("${venv}/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
    if [ "${version}" != "${TARGET_PYTHON}" ]; then
      echo "Replacing old ${venv} (Python ${version:-unknown}) with Python ${TARGET_PYTHON}."
      rm -rf "${venv}"
    fi
  fi
  if [ ! -x "${venv}/bin/python" ]; then
    "${UV_BIN}" venv --python "${TARGET_PYTHON}" --seed "${venv}"
  fi
}

ensure_py312_venv "${RVC_VENV}"
ensure_py312_venv "${PIPER_VENV}"

RVC_PY="${RVC_VENV}/bin/python"
PIPER_PY="${PIPER_VENV}/bin/python"

echo "RVC Python:   $("${RVC_PY}" --version)"
echo "Piper Python: $("${PIPER_PY}" --version)"

mkdir -p "${BASE_DIR}"

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

install_torch() {
  local python="$1"
  local label="$2"
  echo "Installing ${label} PyTorch ${TORCH_SPEC} (${COMPUTE})..."
  "${python}" -m pip install -q --upgrade pip wheel
  "${python}" -m pip install -q --upgrade --index-url "${TORCH_INDEX}" \
    "torch==${TORCH_SPEC}" \
    "torchaudio==${TORCH_SPEC}"

  if [ "${COMPUTE}" = "gpu" ]; then
    "${python}" -W ignore - <<PY
import torch
print("${label} torch:", torch.__version__)
print("${label} CUDA runtime:", torch.version.cuda)
print("${label} CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("${label} PyTorch cannot access the Colab GPU")
x = torch.arange(8, device="cuda", dtype=torch.float32).sum()
print("${label} CUDA kernel test:", float(x.item()))
print("${label} GPU:", torch.cuda.get_device_name(0))
PY
  else
    "${python}" -W ignore - <<PY
import torch
print("${label} torch:", torch.__version__)
print("${label} CUDA available:", torch.cuda.is_available())
x = torch.arange(8, dtype=torch.float32).sum()
print("${label} CPU kernel test:", float(x.item()))
PY
  fi
}

echo
echo "=== Installing RVC inference environment ==="
"${RVC_PY}" -m pip install -q --upgrade 'setuptools<81' wheel
install_torch "${RVC_PY}" "RVC"

# Headless dependencies only. Avoid the old Gradio/UI dependency tree because
# Colab only needs infer.cli for dataset generation.
"${RVC_PY}" -m pip install -q \
  'numpy==1.26.4' 'scipy==1.13.1' 'librosa>=0.10.2,<0.11' \
  'soundfile>=0.13,<1' 'faiss-cpu>=1.9,<2' 'praat-parselmouth>=0.4.5,<1' \
  'PyYAML>=6' 'scikit-learn>=1.6,<2' 'torchfcpe>=0.0.4,<0.1' \
  'transformers>=4.49,<4.50' 'ffmpeg-python>=0.2,<1' 'av>=14,<16' \
  'einops>=0.8,<1' 'local-attention>=1.11,<2' 'huggingface_hub>=0.24' \
  'packaging>=24,<26'

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
print("Downloading RVC HuBERT/ContentVec files...")
snapshot_download(
    repo_id="lj1995/VoiceConversionWebUI",
    allow_patterns=["hubert_base/*"],
    local_dir=r"${RVC_ROOT}/assets",
)
PY
fi

echo
echo "=== Installing Piper training/inference environment ==="
"${PIPER_PY}" -m pip install -q --upgrade 'setuptools<82' wheel scikit-build 'cmake>=3.26,<4' ninja 'cython>=3,<4'
install_torch "${PIPER_PY}" "Piper"
"${PIPER_PY}" -m pip install -q \
  'numpy==1.26.4' 'scipy==1.13.1' 'ml-dtypes>=0.5,<0.6' 'onnx>=1.17,<2'
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
echo "=== Downloading default base Piper voice ==="
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
import numpy, scipy, librosa, soundfile, faiss
print("RVC Python:  ", __import__('sys').version.split()[0])
print("RVC torch:   ", torch.__version__)
print("RVC NumPy:   ", numpy.__version__)
print("RVC SciPy:   ", scipy.__version__)
print("RVC CUDA:    ", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("RVC compute: ", "gpu" if torch.cuda.is_available() else "cpu")
print("RVC imports: OK")
PY

"${PIPER_PY}" -W ignore - <<PY
import torch
import numpy, scipy
import piper.train
from piper.phonemize_espeak import EspeakPhonemizer
p = EspeakPhonemizer()
assert p.phonemize("en-GB-x-rp", "Colab Piper verification.")
from piper.train.vits.monotonic_align.monotonic_align.core import maximum_path_c
print("Piper Python:", __import__('sys').version.split()[0])
print("Piper torch: ", torch.__version__)
print("Piper NumPy: ", numpy.__version__)
print("Piper SciPy: ", scipy.__version__)
print("Piper CUDA:  ", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("Piper compute:", "gpu" if torch.cuda.is_available() else "cpu")
print("eSpeak bridge: OK")
print("monotonic alignment: OK")
PY

printf '%s\n' "${COMPUTE}" > "${MODE_FILE}"

cat > "${RUNTIME_ROOT}/paths.env" <<EOF
APP_ROOT=${APP_ROOT}
RUNTIME_ROOT=${RUNTIME_ROOT}
RVC_ROOT=${RVC_ROOT}
RVC_PY=${RVC_PY}
PIPER_ROOT=${PIPER_ROOT}
PIPER_PY=${PIPER_PY}
COMPUTE=${COMPUTE}
TORCH_FLAVOR=${TORCH_FLAVOR}
BASE_MODEL=${BASE_DIR}/en_GB-alba-medium.onnx
BASE_CONFIG=${BASE_DIR}/en_GB-alba-medium.onnx.json
EOF

echo
echo "Colab runtime is ready."
echo "Compute mode: ${COMPUTE}"
echo "RVC Python:   ${RVC_PY}"
echo "Piper Python: ${PIPER_PY}"
echo "Base voice:   ${BASE_DIR}/en_GB-alba-medium.onnx"
trap - EXIT
