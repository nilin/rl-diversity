# ToolRL Integration Notes

This clean branch intentionally contains only ToolRL-related files, not the Maze scaffold,
Maze outputs, plots, or run logs.

The upstream ToolRL source is vendored in:

```text
vendor/ToolRL
```

The processed RL data is included:

```text
vendor/ToolRL/dataset/rlla_4k/train.parquet
vendor/ToolRL/dataset/rlla_4k/test.parquet
```

## Baseline

The main runnable target is the paper-style ToolRL recipe:

```bash
OBJECTIVE=grpo OPTIMIZER=adamw ./run_grpo_smoke.sh
```

The same wrapper can select patched variants:

```bash
OBJECTIVE=vpo ./run_grpo_smoke.sh
OPTIMIZER=muon ./run_grpo_smoke.sh
OPTIMIZER=soft_muon SOFT_MUON_P=0.5 ./run_grpo_smoke.sh
OBJECTIVE=vpo OPTIMIZER=muon ./run_grpo_smoke.sh
OBJECTIVE=vpo OPTIMIZER=soft_muon SOFT_MUON_P=0.5 ./run_grpo_smoke.sh
```

By default the wrapper uses Qwen3-1.7B, `m=3` attempts per completion, `n=8`
rollouts per prompt, train batch 128, mini-batch 64, micro-batch 8, loss-side
KL, no entropy bonus, training temperature 1.0, and the ToolRL vector reward
described in the VPO paper. Set `TOTAL_TRAINING_STEPS=2 TRAIN_BATCH_SIZE=4
VAL_BATCH_SIZE=4 ROLLOUT_N=2 MULTI_ANSWER_COUNT=1` for a quick mechanical test.

## VPO Patch Points

ToolRL currently computes scalar reward in:

```text
vendor/ToolRL/verl/utils/reward_score/rlla.py
```

For VPO-paper faithfulness, expose a reward vector closer to:

```text
[format, tool_name_f1, arg_key_f1, arg_value_f1]
```

This branch carries per-attempt reward vectors through the trainer and enables
the set-level advantage estimator with:

```text
algorithm.adv_estimator=vpo
```

Then patch:

```text
vendor/ToolRL/verl/trainer/main_ppo.py
vendor/ToolRL/verl/trainer/ppo/ray_trainer.py
vendor/ToolRL/verl/trainer/ppo/core_algos.py
```

The existing GRPO advantage function is:

```text
compute_grpo_outcome_advantage(...)
```

VPO samples Dirichlet weights and scores each generated completion as the mean
best attempt under those scalarizations before GRPO-normalizing across the
rollout group.

## Muon Patch Point

ToolRL's FSDP actor optimizer is currently hard-coded to AdamW in:

```text
vendor/ToolRL/verl/workers/fsdp_workers.py
```

The immediate Muon hook is the actor branch in `_build_model_optimizer`, where it constructs:

```python
actor_optimizer = optim.AdamW(actor_module_fsdp.parameters(), ...)
```

This branch enables a first-pass FSDP Muon split with:

```text
actor_rollout_ref.actor.optim.optimizer=muon
```

It also enables Soft-Muon p=0.5 with:

```text
actor_rollout_ref.actor.optim.optimizer=soft_muon
actor_rollout_ref.actor.optim.soft_muon_p=0.5
actor_rollout_ref.actor.optim.muon_ns_coefficients=[2.0,-1.5,0.5]
actor_rollout_ref.actor.optim.muon_ns_steps=12
```

The Soft-Muon path uses the PR291 Newton-Schulz polynomial and a fixed convex
combination of the 0th through 12th iterates fitted to approximate the
singular-value power map `s^0.5`.
