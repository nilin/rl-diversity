# diversity

Small experiments for testing whether Muon reduces scalar-RL diversity collapse on the VPO paper's Maze-style setting.

This repo is a fast prototype, not a byte-exact reproduction of the paper. The paper does not release its Maze generator or training code. The implementation here keeps the important mechanism fixed:

- multi-answer prompts with `m=3` route candidates
- reward vector `[completion, gold, diamond, avoid_lava]`
- scalar Multi-RLVR with a fixed uniform scalar
- VPO-style set reward from Dirichlet-sampled scalarizations
- AdamW vs Muon optimizer swap
- reward-space diversity evaluation

## Install

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

AdamW Multi-RLVR:

```bash
accelerate launch -m diversity_muon.train_grpo \
  --config configs/qwen06b_adam_multirlvr.yaml
```

Muon Multi-RLVR:

```bash
accelerate launch -m diversity_muon.train_grpo \
  --config configs/qwen06b_muon_multirlvr.yaml
```

AdamW VPO positive control:

```bash
accelerate launch -m diversity_muon.train_grpo \
  --config configs/qwen06b_adam_vpo.yaml
```

Offline diversity eval for a checkpoint:

```bash
python -m diversity_muon.eval_diversity \
  --model outputs/qwen06b_muon_multirlvr/checkpoint-40 \
  --num-prompts 32 \
  --samples-per-prompt 10
```

## Interpretation

The first decision point is early training, not final Table 2 reproduction:

- if Muon Multi-RLVR tracks AdamW Multi-RLVR on reward-space diversity, stop
- if Muon lifts diversity by steps 20-60, run the same test with Qwen3-4B
- if VPO does not lift diversity in this scaffold, treat the scaffold as suspect before interpreting Muon

The paper-comparable run would use Qwen3-4B Maze and the authors' full veRL-like setup. This repo is for the cheap triage experiment.
