#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLRL_DIR="${TOOLRL_DIR:-$SCRIPT_DIR/vendor/ToolRL}"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -f "$TOOLRL_DIR/pyproject.toml" ]]; then
  echo "Vendored ToolRL source not found at $TOOLRL_DIR" >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install -e "$TOOLRL_DIR"

python - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device={torch.cuda.get_device_name(0)}")
PY

python -m pip install flash-attn --no-build-isolation

cat <<EOF

Setup complete.

Activate the environment:
  source "$VENV_DIR/bin/activate"

Run the paper-style VPO test:
  EXPERIMENT_NAME=toolrl-vpo-adamw OBJECTIVE=vpo OPTIMIZER=adamw ./run_grpo_smoke.sh
EOF
