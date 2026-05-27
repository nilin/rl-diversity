# ToolRL Worktree

This branch keeps ToolRL work in a separate worktree while leaving the original Maze checkout
available at:

```text
/Users/nilinabra/external/diversity
```

The ToolRL branch/worktree is:

```text
/Users/nilinabra/external/diversity-toolrl
```

The vendored ToolRL source lives in `toolrl/vendor/ToolRL`. Verify it with:

```bash
./toolrl/setup.sh
```

The top-level ToolRL commands live in this `toolrl/` directory so the important experiment entry
points are visible without digging through the upstream checkout.

## Smoke Baseline

Run a small GRPO baseline from the worktree root:

```bash
BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct ./toolrl/run_grpo_smoke.sh
```

Useful overrides:

```bash
N_GPUS=2 BASE_MODEL=Qwen/Qwen2.5-3B-Instruct ./toolrl/run_grpo_smoke.sh
TRAIN_BATCH_SIZE=64 VAL_BATCH_SIZE=32 ./toolrl/run_grpo_smoke.sh
ROLLOUT_N=8 ./toolrl/run_grpo_smoke.sh
```

## Next Patch Points

For VPO, patch the reward and advantage path in `toolrl/vendor/ToolRL`:

```text
toolrl/vendor/ToolRL/verl/utils/reward_score/rlla.py
toolrl/vendor/ToolRL/verl/trainer/main_ppo.py
toolrl/vendor/ToolRL/verl/trainer/ppo/ray_trainer.py
toolrl/vendor/ToolRL/verl/trainer/ppo/core_algos.py
```

For Muon, patch the actor optimizer construction:

```text
toolrl/vendor/ToolRL/verl/workers/fsdp_workers.py
```

The intended VPO reward vector for ToolRL is:

```text
[format, tool_name_f1, arg_key_f1, arg_value_f1]
```

