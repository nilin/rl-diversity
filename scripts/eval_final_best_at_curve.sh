#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${CHECKPOINT:-checkpoint-50}"
SEED="${SEED:-0}"
SEED_LABEL="${SEED_LABEL:-seed${SEED}}"
EVAL_PROMPTS="${EVAL_PROMPTS:-16}"
MAZE_SIZE="${MAZE_SIZE:-7}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-3}"
BEST_AT_KS="${BEST_AT_KS:-1,3,6,9}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen17b_7x7_best_at_curve_${SEED_LABEL}}"
RUN_MUON="${RUN_MUON:-0}"
DEVICE="${DEVICE:-auto}"
ADAM_MULTI_MODEL_DIR="${ADAM_MULTI_MODEL_DIR:-outputs/qwen17b_7x7_adam_multirlvr_${SEED_LABEL}}"
SOFT_MUON_P04_MODEL_DIR="${SOFT_MUON_P04_MODEL_DIR:-outputs/qwen17b_7x7_soft_muon_p04_fixed_coeffs_multirlvr_${SEED_LABEL}}"
ADAM_VPO_MODEL_DIR="${ADAM_VPO_MODEL_DIR:-outputs/qwen17b_7x7_adam_vpo_${SEED_LABEL}}"
MUON_MULTI_MODEL_DIR="${MUON_MULTI_MODEL_DIR:-outputs/qwen17b_7x7_muon_multirlvr_${SEED_LABEL}}"

mkdir -p "$OUTPUT_DIR"
export OUTPUT_DIR

cat <<EOF
Final eval run config:
  seed: $SEED
  seed_label: $SEED_LABEL
  checkpoint: $CHECKPOINT
  eval_prompts: $EVAL_PROMPTS
  maze_size: $MAZE_SIZE
  samples_per_prompt: $SAMPLES_PER_PROMPT
  best_at_ks: $BEST_AT_KS
  device: $DEVICE
  run_muon: $RUN_MUON
  output_dir: $OUTPUT_DIR
  adam_multi_model_dir: $ADAM_MULTI_MODEL_DIR
  soft_muon_p04_model_dir: $SOFT_MUON_P04_MODEL_DIR
  adam_vpo_model_dir: $ADAM_VPO_MODEL_DIR
  muon_multi_model_dir: $MUON_MULTI_MODEL_DIR
EOF

eval_run() {
  local name="$1"
  local model_dir="$2"

  echo "Evaluating $name from $model_dir/$CHECKPOINT"
  python -m diversity_muon.eval_diversity \
    --model "$model_dir/$CHECKPOINT" \
    --maze-size "$MAZE_SIZE" \
    --num-prompts "$EVAL_PROMPTS" \
    --samples-per-prompt "$SAMPLES_PER_PROMPT" \
    --best-at-ks "$BEST_AT_KS" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --output "$OUTPUT_DIR/$name.json"
}

eval_run adam_multirlvr "$ADAM_MULTI_MODEL_DIR"
eval_run soft_muon_p04_fixed_coeffs_multirlvr "$SOFT_MUON_P04_MODEL_DIR"
eval_run adam_vpo "$ADAM_VPO_MODEL_DIR"
if [[ "$RUN_MUON" == "1" ]]; then
  eval_run muon_multirlvr "$MUON_MULTI_MODEL_DIR"
fi

python - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

output_dir = Path(os.environ.get("OUTPUT_DIR", "outputs/qwen17b_7x7_best_at_curve"))
rows = []
for path in sorted(output_dir.glob("*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    row = {"run": path.stem, "diversity": data["mean_diversity"]}
    for key, value in data.items():
        if key.startswith("mean_best_at_"):
            row[key.removeprefix("mean_")] = value
    rows.append(row)

best_keys = sorted(
    {key for row in rows for key in row if key.startswith("best_at_")},
    key=lambda key: int(key.rsplit("_", 1)[-1]),
)
print("run,diversity," + ",".join(best_keys))
for row in rows:
    values = [row["run"], f"{row['diversity']:.6f}"]
    values.extend(f"{row[key]:.6f}" for key in best_keys)
    print(",".join(values))
PY
