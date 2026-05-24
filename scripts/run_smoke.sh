#!/usr/bin/env bash
set -euo pipefail

python -m diversity_muon.train_grpo --config configs/qwen06b_adam_multirlvr.yaml
