#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TORCH_SPEC="${TORCH_SPEC:-torch}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. Install an NVIDIA driver before running GPU training." >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip

# Install the CUDA-compatible PyTorch wheel first so pip does not resolve to a
# newer CUDA build than the host driver supports.
python -m pip install --force-reinstall --index-url "$TORCH_INDEX_URL" "$TORCH_SPEC"

python -m pip install -e ".[dev]"
python -m pip install "fsspec[http]<=2026.2.0,>=2023.1.0"

python -m pip check

python - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"device_count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"device_name={torch.cuda.get_device_name(0)}")
    print(f"bf16_supported={torch.cuda.is_bf16_supported()}")
else:
    raise SystemExit("PyTorch cannot use CUDA in this environment.")
PY

echo
echo "Environment ready. Run:"
echo "  source $VENV_DIR/bin/activate"
echo "  ./scripts/run_minimum_comparison.sh"
echo "Logs are written under logs/ by the run scripts."
