#!/usr/bin/env bash
set -euo pipefail

TOOLRL_DIR="${TOOLRL_DIR:-toolrl/vendor/ToolRL}"

if [[ ! -f "$TOOLRL_DIR/pyproject.toml" ]]; then
  echo "Vendored ToolRL checkout not found at $TOOLRL_DIR" >&2
  exit 1
fi

cat <<EOF
Vendored ToolRL workspace is ready at $TOOLRL_DIR

Next setup inside the ToolRL environment:
  cd $TOOLRL_DIR
  pip install -e .

Install torch, vLLM, Ray, and flash-attn according to the target CUDA image.
EOF
