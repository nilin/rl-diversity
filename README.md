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

Run the paper-style ToolRL GRPO baseline:

```bash
OBJECTIVE=grpo OPTIMIZER=adamw ./run_grpo_smoke.sh
```

Compare VPO and VPO + Soft-Muon:

```bash
EXPERIMENT_NAME=toolrl-vpo-adamw OBJECTIVE=vpo OPTIMIZER=adamw ./run_grpo_smoke.sh
EXPERIMENT_NAME=toolrl-vpo-soft-muon-p05 OBJECTIVE=vpo OPTIMIZER=soft_muon SOFT_MUON_P=0.5 ./run_grpo_smoke.sh
```

The wrapper defaults to the paper-style ToolRL recipe where possible: Qwen3-1.7B,
3 attempts per completion, 8 rollouts per prompt, train batch 128, mini-batch 64,
micro-batch 8, loss-side KL, no entropy bonus, and ToolRL vector reward
`[format, tool_name_f1, arg_key_f1, arg_value_f1]`.

Useful overrides for shorter tests:

```bash
TOTAL_TRAINING_STEPS=2 TRAIN_BATCH_SIZE=4 VAL_BATCH_SIZE=4 ROLLOUT_N=2 MULTI_ANSWER_COUNT=1 ./run_grpo_smoke.sh
N_GPUS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_grpo_smoke.sh
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

Soft-Muon p=0.5 uses:

```text
actor_rollout_ref.actor.optim.optimizer=soft_muon
actor_rollout_ref.actor.optim.soft_muon_p=0.5
actor_rollout_ref.actor.optim.muon_ns_coefficients=[2.0,-1.5,0.5]
actor_rollout_ref.actor.optim.muon_ns_steps=12
```
