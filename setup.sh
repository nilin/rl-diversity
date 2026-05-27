#!/usr/bin/env bash
set -euo pipefail

TOOLRL_DIR="${TOOLRL_DIR:-vendor/ToolRL}"

if [[ ! -f "$TOOLRL_DIR/pyproject.toml" ]]; then
  echo "Vendored ToolRL source not found at $TOOLRL_DIR" >&2
  exit 1
fi

cat <<EOF
Vendored ToolRL source is ready at $TOOLRL_DIR

Next setup inside the target Python/CUDA environment:
  cd $TOOLRL_DIR
  pip install -e .

Install torch, vLLM, Ray, and flash-attn according to the target CUDA image.
EOF

