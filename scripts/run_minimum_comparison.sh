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
SEED_LABEL="${SEED_LABEL:-seed${SEED}}"
EVAL_START="${EVAL_START:-0}"
RUN_VPO="${RUN_VPO:-1}"
RUN_MUON="${RUN_MUON:-0}"
RUN_SOFT_MUON_P04="${RUN_SOFT_MUON_P04:-1}"
EVAL_PROMPTS="${EVAL_PROMPTS:-16}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-3}"
MAZE_SIZE="${MAZE_SIZE:-7}"
CHECKPOINT="${CHECKPOINT:-checkpoint-50}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-1.7B}"
EVAL_DEVICE="${EVAL_DEVICE:-auto}"
ADAM_MULTI_RUN_NAME="${ADAM_MULTI_RUN_NAME:-qwen17b_7x7_adam_multirlvr_${SEED_LABEL}}"
MUON_MULTI_RUN_NAME="${MUON_MULTI_RUN_NAME:-qwen17b_7x7_muon_multirlvr_${SEED_LABEL}}"
SOFT_MUON_P04_RUN_NAME="${SOFT_MUON_P04_RUN_NAME:-qwen17b_7x7_soft_muon_p04_fixed_coeffs_multirlvr_${SEED_LABEL}}"
ADAM_VPO_RUN_NAME="${ADAM_VPO_RUN_NAME:-qwen17b_7x7_adam_vpo_${SEED_LABEL}}"
ADAM_MULTI_OUTPUT="${ADAM_MULTI_OUTPUT:-outputs/$ADAM_MULTI_RUN_NAME}"
MUON_MULTI_OUTPUT="${MUON_MULTI_OUTPUT:-outputs/$MUON_MULTI_RUN_NAME}"
SOFT_MUON_P04_OUTPUT="${SOFT_MUON_P04_OUTPUT:-outputs/$SOFT_MUON_P04_RUN_NAME}"
ADAM_VPO_OUTPUT="${ADAM_VPO_OUTPUT:-outputs/$ADAM_VPO_RUN_NAME}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-outputs/qwen17b_7x7_eval_${SEED_LABEL}}"
INITIAL_EVAL_OUTPUT="${INITIAL_EVAL_OUTPUT:-$EVAL_OUTPUT_DIR/initial.json}"

mkdir -p "$EVAL_OUTPUT_DIR"
export EVAL_OUTPUT_DIR

cat <<EOF
Benchmark run config:
  seed: $SEED
  seed_label: $SEED_LABEL
  num_processes: ${NUM_PROCESSES:-default}
  checkpoint: $CHECKPOINT
  maze_size: $MAZE_SIZE
  eval_start: $EVAL_START
  base_model: $BASE_MODEL
  eval_prompts: $EVAL_PROMPTS
  samples_per_prompt: $SAMPLES_PER_PROMPT
  eval_device: $EVAL_DEVICE
  run_vpo: $RUN_VPO
  run_muon: $RUN_MUON
  run_soft_muon_p04: $RUN_SOFT_MUON_P04
  adam_multi_run_name: $ADAM_MULTI_RUN_NAME
  muon_multi_run_name: $MUON_MULTI_RUN_NAME
  soft_muon_p04_run_name: $SOFT_MUON_P04_RUN_NAME
  adam_vpo_run_name: $ADAM_VPO_RUN_NAME
  adam_multi_output: $ADAM_MULTI_OUTPUT
  muon_multi_output: $MUON_MULTI_OUTPUT
  soft_muon_p04_output: $SOFT_MUON_P04_OUTPUT
  adam_vpo_output: $ADAM_VPO_OUTPUT
  eval_output_dir: $EVAL_OUTPUT_DIR
  initial_eval_output: $INITIAL_EVAL_OUTPUT
EOF

accelerate_launch() {
  local args=(accelerate launch)
  if [[ -n "$NUM_PROCESSES" ]]; then
    args+=(--num_processes "$NUM_PROCESSES")
  fi
  "${args[@]}" -m diversity_muon.train_grpo "$@"
}

if [[ "$EVAL_START" == "1" ]]; then
  if [[ -f "$INITIAL_EVAL_OUTPUT" ]]; then
    echo "Reusing initial eval at $INITIAL_EVAL_OUTPUT"
  else
    echo "Evaluating shared initial model"
    python -m diversity_muon.eval_diversity \
      --model "$BASE_MODEL" \
      --maze-size "$MAZE_SIZE" \
      --num-prompts "$EVAL_PROMPTS" \
      --samples-per-prompt "$SAMPLES_PER_PROMPT" \
      --device "$EVAL_DEVICE" \
      --seed "$SEED" \
      --output "$INITIAL_EVAL_OUTPUT"
  fi
fi

echo "Running AdamW Multi-RLVR minimum run"
accelerate_launch \
  --config configs/qwen17b_7x7_adam_multirlvr.yaml \
  --seed "$SEED" \
  --run-name "$ADAM_MULTI_RUN_NAME" \
  --output-dir "$ADAM_MULTI_OUTPUT"

if [[ "$RUN_SOFT_MUON_P04" == "1" ]]; then
  echo "Running fixed Soft-Muon p=0.4 Multi-RLVR minimum run"
  accelerate_launch \
    --config configs/qwen17b_7x7_soft_muon_p04_fixed_coeffs_multirlvr.yaml \
    --seed "$SEED" \
    --run-name "$SOFT_MUON_P04_RUN_NAME" \
    --output-dir "$SOFT_MUON_P04_OUTPUT"
fi

if [[ "$RUN_VPO" == "1" ]]; then
  echo "Running AdamW VPO minimum positive control"
  accelerate_launch \
    --config configs/qwen17b_7x7_adam_vpo.yaml \
    --seed "$SEED" \
    --run-name "$ADAM_VPO_RUN_NAME" \
    --output-dir "$ADAM_VPO_OUTPUT"
fi

if [[ "$RUN_MUON" == "1" ]]; then
  echo "Running Muon Multi-RLVR minimum run"
  accelerate_launch \
    --config configs/qwen17b_7x7_muon_multirlvr.yaml \
    --seed "$SEED" \
    --run-name "$MUON_MULTI_RUN_NAME" \
    --output-dir "$MUON_MULTI_OUTPUT"
fi

echo "Evaluating AdamW Multi-RLVR"
python -m diversity_muon.eval_diversity \
  --model "$ADAM_MULTI_OUTPUT/$CHECKPOINT" \
  --maze-size "$MAZE_SIZE" \
  --num-prompts "$EVAL_PROMPTS" \
  --samples-per-prompt "$SAMPLES_PER_PROMPT" \
  --device "$EVAL_DEVICE" \
  --seed "$SEED" \
  --output "$EVAL_OUTPUT_DIR/adam_multirlvr.json"

if [[ "$RUN_MUON" == "1" ]]; then
  echo "Evaluating Muon Multi-RLVR"
  python -m diversity_muon.eval_diversity \
    --model "$MUON_MULTI_OUTPUT/$CHECKPOINT" \
    --maze-size "$MAZE_SIZE" \
    --num-prompts "$EVAL_PROMPTS" \
    --samples-per-prompt "$SAMPLES_PER_PROMPT" \
    --device "$EVAL_DEVICE" \
    --seed "$SEED" \
    --output "$EVAL_OUTPUT_DIR/muon_multirlvr.json"
fi

if [[ "$RUN_SOFT_MUON_P04" == "1" ]]; then
  echo "Evaluating fixed Soft-Muon p=0.4 Multi-RLVR"
  python -m diversity_muon.eval_diversity \
    --model "$SOFT_MUON_P04_OUTPUT/$CHECKPOINT" \
    --maze-size "$MAZE_SIZE" \
    --num-prompts "$EVAL_PROMPTS" \
    --samples-per-prompt "$SAMPLES_PER_PROMPT" \
    --device "$EVAL_DEVICE" \
    --seed "$SEED" \
    --output "$EVAL_OUTPUT_DIR/soft_muon_p04_fixed_coeffs_multirlvr.json"
fi

if [[ "$RUN_VPO" == "1" ]]; then
  echo "Evaluating AdamW VPO"
  python -m diversity_muon.eval_diversity \
    --model "$ADAM_VPO_OUTPUT/$CHECKPOINT" \
    --maze-size "$MAZE_SIZE" \
    --num-prompts "$EVAL_PROMPTS" \
    --samples-per-prompt "$SAMPLES_PER_PROMPT" \
    --device "$EVAL_DEVICE" \
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
    best_keys = sorted(
        [key for key in data if key.startswith("mean_best_at_")],
        key=lambda key: int(key.rsplit("_", 1)[-1]),
    )
    best_summary = ", ".join(
        f"{key.removeprefix('mean_')}={data[key]:.4f}" for key in best_keys
    )
    print(
        f"{path.stem}: "
        f"diversity={data['mean_diversity']:.4f}, "
        f"{best_summary}"
    )
PY
