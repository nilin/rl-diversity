#!/usr/bin/env bash
set -euo pipefail

if [[ "${RUN_LOGGING_ACTIVE:-0}" != "1" ]]; then
  LOG_DIR="${LOG_DIR:-logs}"
  mkdir -p "$LOG_DIR"
  RUN_LOG_FILE="${RUN_LOG_FILE:-$LOG_DIR/$(basename "$0" .sh)_$(date +%Y%m%d_%H%M%S).log}"
  export RUN_LOGGING_ACTIVE=1
  exec > >(tee -a "$RUN_LOG_FILE") 2>&1
  echo "Writing log to $RUN_LOG_FILE"
fi

NUM_PROCESSES="${NUM_PROCESSES:-}"
SEED="${SEED:-0}"
RUN_VPO="${RUN_VPO:-1}"
RUN_MUON="${RUN_MUON:-0}"
RUN_SOFT_MUON="${RUN_SOFT_MUON:-1}"
EVAL_PROMPTS="${EVAL_PROMPTS:-16}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-10}"
MAZE_SIZE="${MAZE_SIZE:-7}"
CHECKPOINT="${CHECKPOINT:-checkpoint-40}"
ADAM_MULTI_OUTPUT="${ADAM_MULTI_OUTPUT:-outputs/qwen17b_7x7_adam_multirlvr}"
MUON_MULTI_OUTPUT="${MUON_MULTI_OUTPUT:-outputs/qwen17b_7x7_muon_multirlvr}"
SOFT_MUON_MULTI_OUTPUT="${SOFT_MUON_MULTI_OUTPUT:-outputs/qwen17b_7x7_soft_muon_p05_multirlvr}"
ADAM_VPO_OUTPUT="${ADAM_VPO_OUTPUT:-outputs/qwen17b_7x7_adam_vpo}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-outputs/qwen17b_7x7_eval}"

mkdir -p "$EVAL_OUTPUT_DIR"

accelerate_launch() {
  local args=(accelerate launch)
  if [[ -n "$NUM_PROCESSES" ]]; then
    args+=(--num_processes "$NUM_PROCESSES")
  fi
  "${args[@]}" -m diversity_muon.train_grpo "$@"
}

echo "Running AdamW Multi-RLVR minimum run"
accelerate_launch \
  --config configs/qwen17b_7x7_adam_multirlvr.yaml \
  --seed "$SEED"

if [[ "$RUN_SOFT_MUON" == "1" ]]; then
  echo "Running Soft-Muon Multi-RLVR minimum run"
  accelerate_launch \
    --config configs/qwen17b_7x7_soft_muon_p05_multirlvr.yaml \
    --seed "$SEED"
fi

if [[ "$RUN_VPO" == "1" ]]; then
  echo "Running AdamW VPO minimum positive control"
  accelerate_launch \
    --config configs/qwen17b_7x7_adam_vpo.yaml \
    --seed "$SEED"
fi

if [[ "$RUN_MUON" == "1" ]]; then
  echo "Running Muon Multi-RLVR minimum run"
  accelerate_launch \
    --config configs/qwen17b_7x7_muon_multirlvr.yaml \
    --seed "$SEED"
fi

echo "Evaluating AdamW Multi-RLVR"
python -m diversity_muon.eval_diversity \
  --model "$ADAM_MULTI_OUTPUT/$CHECKPOINT" \
  --maze-size "$MAZE_SIZE" \
  --num-prompts "$EVAL_PROMPTS" \
  --samples-per-prompt "$SAMPLES_PER_PROMPT" \
  --seed "$SEED" \
  --output "$EVAL_OUTPUT_DIR/adam_multirlvr.json"

if [[ "$RUN_MUON" == "1" ]]; then
  echo "Evaluating Muon Multi-RLVR"
  python -m diversity_muon.eval_diversity \
    --model "$MUON_MULTI_OUTPUT/$CHECKPOINT" \
    --maze-size "$MAZE_SIZE" \
    --num-prompts "$EVAL_PROMPTS" \
    --samples-per-prompt "$SAMPLES_PER_PROMPT" \
    --seed "$SEED" \
    --output "$EVAL_OUTPUT_DIR/muon_multirlvr.json"
fi

if [[ "$RUN_SOFT_MUON" == "1" ]]; then
  echo "Evaluating Soft-Muon Multi-RLVR"
  python -m diversity_muon.eval_diversity \
    --model "$SOFT_MUON_MULTI_OUTPUT/$CHECKPOINT" \
    --maze-size "$MAZE_SIZE" \
    --num-prompts "$EVAL_PROMPTS" \
    --samples-per-prompt "$SAMPLES_PER_PROMPT" \
    --seed "$SEED" \
    --output "$EVAL_OUTPUT_DIR/soft_muon_p05_multirlvr.json"
fi

if [[ "$RUN_VPO" == "1" ]]; then
  echo "Evaluating AdamW VPO"
  python -m diversity_muon.eval_diversity \
    --model "$ADAM_VPO_OUTPUT/$CHECKPOINT" \
    --maze-size "$MAZE_SIZE" \
    --num-prompts "$EVAL_PROMPTS" \
    --samples-per-prompt "$SAMPLES_PER_PROMPT" \
    --seed "$SEED" \
    --output "$EVAL_OUTPUT_DIR/adam_vpo.json"
fi

echo "Summary"
python - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

import os

eval_dir = Path(os.environ.get("EVAL_OUTPUT_DIR", "outputs/qwen17b_7x7_eval"))
for path in sorted(eval_dir.glob("*.json")):
    data = json.loads(path.read_text())
    print(
        f"{path.stem}: "
        f"diversity={data['mean_diversity']:.4f}, "
        f"best@30={data['mean_best_at_30']:.4f}, "
        f"best@10={data['mean_best_at_10']:.4f}"
    )
PY
