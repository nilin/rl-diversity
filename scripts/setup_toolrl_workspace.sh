#!/usr/bin/env bash
set -euo pipefail

TOOLRL_REPO="${TOOLRL_REPO:-https://github.com/qiancheng0/ToolRL.git}"
TOOLRL_COMMIT="${TOOLRL_COMMIT:-8cee13ec0ca72f0461da372a93a6fd8140dbb840}"
TOOLRL_DIR="${TOOLRL_DIR:-external/ToolRL}"

if [[ -d "$TOOLRL_DIR/.git" ]]; then
  git -C "$TOOLRL_DIR" fetch origin "$TOOLRL_COMMIT"
else
  mkdir -p "$(dirname "$TOOLRL_DIR")"
  git clone "$TOOLRL_REPO" "$TOOLRL_DIR"
fi

git -C "$TOOLRL_DIR" checkout "$TOOLRL_COMMIT"

cat <<EOF
ToolRL workspace ready at $TOOLRL_DIR

Next setup inside the ToolRL environment:
  cd $TOOLRL_DIR
  pip install -e .

Install torch, vLLM, Ray, and flash-attn according to the target CUDA image.
EOF
