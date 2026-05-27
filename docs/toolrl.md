# ToolRL Integration Notes

ToolRL is the next benchmark target for the VPO/Muon test because it already ships a veRL-based
RL stack, processed RL data, and a tool-call reward implementation.

Inspected upstream:

```text
repo: https://github.com/qiancheng0/ToolRL.git
commit: 8cee13ec0ca72f0461da372a93a6fd8140dbb840
```

Verify the vendored source:

```bash
./scripts/setup_toolrl_workspace.sh
```

By default this checks `toolrl/vendor/ToolRL`. Override with:

```bash
TOOLRL_DIR=/path/to/ToolRL ./scripts/setup_toolrl_workspace.sh
```

## Baseline Path

ToolRL's baseline GRPO entry point is:

```bash
toolrl/vendor/ToolRL/train_grpo.sh
```

That script wraps:

```bash
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$DATA_DIR/train.parquet \
  data.val_files=$DATA_DIR/test.parquet \
  actor_rollout_ref.rollout.n=4
```

The shipped dataset is:

```text
toolrl/vendor/ToolRL/dataset/rlla_4k/train.parquet
toolrl/vendor/ToolRL/dataset/rlla_4k/test.parquet
```

## VPO Hook Points

ToolRL currently computes scalar reward in:

```text
toolrl/vendor/ToolRL/verl/utils/reward_score/rlla.py
```

The scalar comes from:

```text
format_score + correctness_score + length_score
```

For VPO-paper faithfulness, we should expose a vector reward closer to:

```text
[format, tool_name_f1, arg_key_f1, arg_value_f1]
```

Then the advantage path to patch is:

```text
toolrl/vendor/ToolRL/verl/trainer/main_ppo.py
toolrl/vendor/ToolRL/verl/trainer/ppo/ray_trainer.py
toolrl/vendor/ToolRL/verl/trainer/ppo/core_algos.py
```

The existing GRPO advantage function is:

```text
compute_grpo_outcome_advantage(...)
```

VPO should add a set-level objective over each prompt group by sampling Dirichlet weights and
taking max over candidates under each sampled scalarization.

## Muon Hook Point

ToolRL's FSDP actor optimizer is currently hard-coded to AdamW in:

```text
toolrl/vendor/ToolRL/verl/workers/fsdp_workers.py
```

The immediate Muon hook is the actor branch in `_build_model_optimizer`, where it constructs:

```python
actor_optimizer = optim.AdamW(actor_module_fsdp.parameters(), ...)
```

As with the Maze veRL path, ToolRL's FSDP optimizer path only sees parameters from the wrapped
module. A first pass can split by tensor rank; a name-aware split would require constructing the
optimizer before names are lost or carrying names through the FSDP wrapping path.

## First Runnable Target

Start with the top-level smoke wrapper before patching VPO/Muon:

```bash
BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct ./toolrl/run_grpo_smoke.sh
```

The equivalent raw ToolRL command is:

```bash
cd toolrl/vendor/ToolRL
export CUDA_VISIBLE_DEVICES=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export DATA_DIR=./dataset/rlla_4k
export BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct
export EXPERIMENT_NAME=toolrl-grpo-smoke

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$DATA_DIR/train.parquet \
  data.val_files=$DATA_DIR/test.parquet \
  data.train_batch_size=32 \
  data.val_batch_size=16 \
  data.max_prompt_length=2048 \
  data.max_response_length=1024 \
  actor_rollout_ref.model.path=$BASE_MODEL \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.grad_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.critic_warmup=0 \
  trainer.logger='["console"]' \
  trainer.project_name=toolrl-vpo-muon \
  trainer.experiment_name=$EXPERIMENT_NAME \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.save_freq=5 \
  trainer.test_freq=5
```

If this smoke run is stable, patch reward-vector logging and VPO advantage next.
