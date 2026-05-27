# ToolRL VPO/Muon

Clean ToolRL-only workspace for testing VPO-style reward-vector training and Muon optimizer swaps.

Vendored upstream:

```text
repo: https://github.com/qiancheng0/ToolRL.git
commit: 8cee13ec0ca72f0461da372a93a6fd8140dbb840
path: vendor/ToolRL
```

Verify the vendored source:

```bash
./setup.sh
```

Run a small GRPO baseline smoke test:

```bash
BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct ./run_grpo_smoke.sh
```

Run patched variants from the same entry point:

```bash
OBJECTIVE=vpo BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct ./run_grpo_smoke.sh
OPTIMIZER=muon BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct ./run_grpo_smoke.sh
OBJECTIVE=vpo OPTIMIZER=muon BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct ./run_grpo_smoke.sh
```

Useful overrides:

```bash
N_GPUS=2 BASE_MODEL=Qwen/Qwen2.5-3B-Instruct ./run_grpo_smoke.sh
TRAIN_BATCH_SIZE=64 VAL_BATCH_SIZE=32 ./run_grpo_smoke.sh
ROLLOUT_N=8 ./run_grpo_smoke.sh
```

Patch points:

```text
vendor/ToolRL/verl/utils/reward_score/rlla.py
vendor/ToolRL/verl/trainer/main_ppo.py
vendor/ToolRL/verl/trainer/ppo/ray_trainer.py
vendor/ToolRL/verl/trainer/ppo/core_algos.py
vendor/ToolRL/verl/workers/fsdp_workers.py
```

VPO uses:

```text
algorithm.adv_estimator=vpo
reward vector: [format, tool_name_f1, arg_key_f1, arg_value_f1]
```

Muon uses:

```text
actor_rollout_ref.actor.optim.optimizer=muon
```
