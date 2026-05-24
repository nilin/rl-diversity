#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-0.6B}"
DATA_DIR="${DATA_DIR:-data/maze_verl}"
TRAIN_SIZE="${TRAIN_SIZE:-128}"
VAL_SIZE="${VAL_SIZE:-32}"
RUN_MUON="${RUN_MUON:-1}"
RUN_VPO="${RUN_VPO:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REWARD_PATH="$REPO_ROOT/src/diversity_muon/verl_reward.py"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

python -m diversity_muon.export_verl_maze \
  --output-dir "$DATA_DIR" \
  --train-size "$TRAIN_SIZE" \
  --val-size "$VAL_SIZE"

run_verl() {
  local name="$1"
  local objective="$2"
  local optimizer="$3"

  export DIVERSITY_OBJECTIVE="$objective"
  export DIVERSITY_SEED=0

  local optim_overrides=(
    actor_rollout_ref.actor.optim.optimizer=AdamW
    actor_rollout_ref.actor.optim.optimizer_impl=torch.optim
  )
  if [[ "$optimizer" == "muon" ]]; then
    optim_overrides=(
      actor_rollout_ref.actor.optim.optimizer=MuonNdMatWithAdamW
      actor_rollout_ref.actor.optim.optimizer_impl=diversity_muon.verl_optim
    )
  fi

  python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size=16 \
    data.max_prompt_length=1024 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    reward.custom_reward_function.path="$REWARD_PATH" \
    reward.custom_reward_function.name=compute_score \
    trainer.project_name=diversity \
    trainer.experiment_name="$name" \
    trainer.logger='["console"]' \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE:-1}" \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=10 \
    trainer.save_freq=10 \
    trainer.test_freq=-1 \
    "${optim_overrides[@]}"
}

run_verl min_verl_adam_multirlvr multirlvr adamw

if [[ "$RUN_MUON" == "1" ]]; then
  run_verl min_verl_muon_multirlvr multirlvr muon
fi

if [[ "$RUN_VPO" == "1" ]]; then
  run_verl min_verl_adam_vpo vpo adamw
fi
