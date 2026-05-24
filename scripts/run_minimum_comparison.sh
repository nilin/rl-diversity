#!/usr/bin/env bash
set -euo pipefail

RUN_VPO="${RUN_VPO:-1}"
EVAL_PROMPTS="${EVAL_PROMPTS:-16}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-5}"

mkdir -p outputs/min_eval

echo "Running AdamW Multi-RLVR minimum run"
accelerate launch -m diversity_muon.train_grpo \
  --config configs/min_adam_multirlvr.yaml

echo "Running Muon Multi-RLVR minimum run"
accelerate launch -m diversity_muon.train_grpo \
  --config configs/min_muon_multirlvr.yaml

if [[ "$RUN_VPO" == "1" ]]; then
  echo "Running AdamW VPO minimum positive control"
  accelerate launch -m diversity_muon.train_grpo \
    --config configs/min_adam_vpo.yaml
fi

echo "Evaluating AdamW Multi-RLVR"
python -m diversity_muon.eval_diversity \
  --model outputs/min_adam_multirlvr/checkpoint-10 \
  --num-prompts "$EVAL_PROMPTS" \
  --samples-per-prompt "$SAMPLES_PER_PROMPT" \
  --output outputs/min_eval/adam_multirlvr.json

echo "Evaluating Muon Multi-RLVR"
python -m diversity_muon.eval_diversity \
  --model outputs/min_muon_multirlvr/checkpoint-10 \
  --num-prompts "$EVAL_PROMPTS" \
  --samples-per-prompt "$SAMPLES_PER_PROMPT" \
  --output outputs/min_eval/muon_multirlvr.json

if [[ "$RUN_VPO" == "1" ]]; then
  echo "Evaluating AdamW VPO"
  python -m diversity_muon.eval_diversity \
    --model outputs/min_adam_vpo/checkpoint-10 \
    --num-prompts "$EVAL_PROMPTS" \
    --samples-per-prompt "$SAMPLES_PER_PROMPT" \
    --output outputs/min_eval/adam_vpo.json
fi

echo "Summary"
python - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

for path in sorted(Path("outputs/min_eval").glob("*.json")):
    data = json.loads(path.read_text())
    print(
        f"{path.stem}: "
        f"diversity={data['mean_diversity']:.4f}, "
        f"best@30={data['mean_best_at_30']:.4f}, "
        f"best@10={data['mean_best_at_10']:.4f}"
    )
PY
