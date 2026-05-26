#!/usr/bin/env bash
set -euo pipefail

STEPS="${STEPS:-10}"
EVAL_PROMPTS="${EVAL_PROMPTS:-16}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-5}"
MAZE_SIZE="${MAZE_SIZE:-5}"

case "$STEPS" in
  10)
    CONFIG="${CONFIG:-configs/min_soft_muon_p05_multirlvr.yaml}"
    OUTPUT_DIR="${OUTPUT_DIR:-outputs/min5_soft_muon_p05_multirlvr}"
    EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-outputs/min5_eval}"
    CHECKPOINT="checkpoint-10"
    ;;
  30)
    CONFIG="${CONFIG:-configs/min30_soft_muon_p05_multirlvr.yaml}"
    OUTPUT_DIR="${OUTPUT_DIR:-outputs/min5_30_soft_muon_p05_multirlvr}"
    EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-outputs/min5_30_eval}"
    CHECKPOINT="checkpoint-30"
    ;;
  *)
    echo "Unsupported STEPS=$STEPS. Use STEPS=10 or STEPS=30." >&2
    exit 2
    ;;
esac

mkdir -p "$EVAL_OUTPUT_DIR"

echo "Running Soft-Muon p05 Multi-RLVR ($STEPS steps)"
accelerate launch -m diversity_muon.train_grpo --config "$CONFIG"

echo "Evaluating Soft-Muon p05 Multi-RLVR"
python -m diversity_muon.eval_diversity \
  --model "$OUTPUT_DIR/$CHECKPOINT" \
  --maze-size "$MAZE_SIZE" \
  --num-prompts "$EVAL_PROMPTS" \
  --samples-per-prompt "$SAMPLES_PER_PROMPT" \
  --output "$EVAL_OUTPUT_DIR/soft_muon_p05_multirlvr.json"

echo "Summary"
python - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

eval_dir = Path(os.environ["EVAL_OUTPUT_DIR"])
path = eval_dir / "soft_muon_p05_multirlvr.json"
data = json.loads(path.read_text())
print(
    f"{path}: "
    f"diversity={data['mean_diversity']:.4f}, "
    f"best@30={data['mean_best_at_30']:.4f}, "
    f"best@10={data['mean_best_at_10']:.4f}"
)
PY
