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

Muon Multi-RLVR, Qwen3-1.7B on 7x7 mazes:

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
  --model outputs/qwen17b_7x7_muon_multirlvr/checkpoint-40 \
  --maze-size 7 \
  --num-prompts 32 \
  --samples-per-prompt 10
```

The 1.7B/7x7 configs also run a small in-training diversity evaluation every
10 steps, with real best@3/6/12 from 4 multi-answer chains per prompt, and append
JSONL records to `outputs/<run>/diversity_eval.jsonl`.
Plot those Figure-6-style curves with:

```bash
python scripts/plot_training_diversity.py
```

