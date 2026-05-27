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

The first runnable target is the GRPO smoke baseline:

```bash
BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct ./run_grpo_smoke.sh
```

This verifies ToolRL's veRL trainer, vLLM rollout, parquet data, reward function, and GRPO update
before longer VPO/Muon runs.

The same wrapper can select patched variants:

```bash
OBJECTIVE=vpo ./run_grpo_smoke.sh
OPTIMIZER=muon ./run_grpo_smoke.sh
OBJECTIVE=vpo OPTIMIZER=muon ./run_grpo_smoke.sh
```

## VPO Patch Points

ToolRL currently computes scalar reward in:

```text
vendor/ToolRL/verl/utils/reward_score/rlla.py
```

For VPO-paper faithfulness, expose a reward vector closer to:

```text
[format, tool_name_f1, arg_key_f1, arg_value_f1]
```

This branch carries that vector through as token-level tensors and enables the set-level
advantage estimator with:

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

VPO should add a set-level objective over each prompt group by sampling Dirichlet weights and
taking max over candidates under each sampled scalarization.

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
