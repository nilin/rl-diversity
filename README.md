# Diversity in RL Finetuning through Muon Variants

Small experiments for testing whether Muon or Soft-Muon reduces scalar-RL diversity collapse on the VPO paper's Maze-style setting.

This repo is a fast prototype, not an exact reproduction of the paper. The paper does not release its Maze generator or training code. The implementation here keeps the important mechanism fixed:

- multi-answer prompts with `m=3` route candidates
- reward vector `[completion, gold, diamond, avoid_lava]`
- scalar Multi-RLVR with a fixed uniform scalar
- VPO-style set reward from Dirichlet-sampled scalarizations
- AdamW vs Soft-Muon/Muon optimizer swap
- reward-space diversity evaluation

## Install

For a fresh NVIDIA GPU machine, run the setup script:

```bash
./scripts/setup_minimum_env.sh
```

It creates `.venv`, installs the CUDA 12.8 PyTorch wheel, installs this
package in editable mode, and verifies CUDA/bf16 availability.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For GPU training with vLLM generation:

```bash
pip install -e ".[dev,vllm]"
```

## Smoke Runs

AdamW Multi-RLVR, Qwen3-1.7B on 7x7 mazes:

```bash
accelerate launch -m diversity_muon.train_grpo \
  --config configs/qwen17b_7x7_adam_multirlvr.yaml
```

Fixed Soft-Muon p=0.4 Multi-RLVR, Qwen3-1.7B on 7x7 mazes:

```bash
accelerate launch -m diversity_muon.train_grpo \
  --config configs/qwen17b_7x7_soft_muon_p04_fixed_coeffs_multirlvr.yaml
```

Optional normal Muon Multi-RLVR, Qwen3-1.7B on 7x7 mazes:

```bash
accelerate launch -m diversity_muon.train_grpo \
  --config configs/qwen17b_7x7_muon_multirlvr.yaml
```

AdamW VPO positive control, Qwen3-1.7B on 7x7 mazes:

```bash
accelerate launch -m diversity_muon.train_grpo \
  --config configs/qwen17b_7x7_adam_vpo.yaml
```

Offline diversity eval for a checkpoint:

```bash
python -m diversity_muon.eval_diversity \
  --model outputs/qwen17b_7x7_soft_muon_p04_fixed_coeffs_multirlvr_seed0/checkpoint-50 \
  --maze-size 7 \
  --num-prompts 32 \
  --samples-per-prompt 3
```

To run the default 7x7 comparison end-to-end:

```bash
./scripts/run_minimum_comparison.sh
```

By default this runs seed 0 for AdamW Multi-RLVR, fixed Soft-Muon p=0.4
Multi-RLVR, and AdamW VPO. To run seeds 0, 1, and 2 in sequence:

```bash
./scripts/run_minimum_comparison.sh 3
```

Set `SEED` to change the starting seed. For example, `SEED=4
./scripts/run_minimum_comparison.sh 2` runs seeds 4 and 5. Add normal Muon
after VPO with:

```bash
RUN_MUON=1 ./scripts/run_minimum_comparison.sh
```

The 1.7B/7x7 configs also run a small in-training diversity evaluation every
10 steps, with paper-style route-level best@1/3/6/9 from 3 multi-answer
chains per prompt, and append JSONL records to
`outputs/<run>/diversity_eval.jsonl`.
Plot those Figure-6-style curves with:

```bash
python scripts/plot_training_diversity.py
```

## Evaluation Metric Convention

Each sampled completion is a multi-answer rollout containing `m=3` route
candidates. `mean_best_at_k` follows the VPO paper's candidate-pool convention,
not the older "first k flattened routes" shortcut:

- `best@1`: score of the first route in one rollout, averaged across rollouts.
- `best@3`: best route inside one rollout, averaged across rollouts.
- `best@6`: best route inside every pair of rollouts, averaged over all rollout pairs.
- `best@9`: best route inside all three rollouts.

For non-multiples of 3, the evaluator averages over ordered rollout selections
and takes the first `k` routes after concatenating those rollouts in draw order.
Both in-training eval and final eval use the same shared implementation. The
JSON outputs also keep raw `rollouts`, `completion_route_scores`, `route_scores`,
and `best_at_values`, so alternative post-hoc metrics can be recomputed without
rerunning generation.

## Learning-Rate Matching

The AdamW, Muon, and Soft-Muon comparison configs use the same nominal
`learning_rate: 1.0e-6`. That makes the configs easy to compare, but the
optimizer update rules are not identical.

For Muon-style 2D matrix parameters, the code applies Muon's
`match_rms_adamw` learning-rate adjustment. In practice this multiplies the
configured LR by:

```text
0.2 * sqrt(max(fan_out, fan_in))
```

Soft-Muon uses the same shape-dependent adjustment after normalizing its soft
update to the same Frobenius-scale target as the Muon zeropower update. Normal
Muon uses PyTorch's `adjust_lr_fn="match_rms_adamw"` path.

Auxiliary parameters still use AdamW at the configured LR. This includes
embeddings, `lm_head`, norms, biases, and non-2D tensors. So the LR comparison
is best read as: same nominal LR, Muon/Soft-Muon matrix updates scaled by the
standard RMS-AdamW matching heuristic, not strict step-for-step equivalence with
AdamW.
