#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLRL_DIR="${TOOLRL_DIR:-$SCRIPT_DIR/vendor/ToolRL}"

if [[ ! -f "$TOOLRL_DIR/pyproject.toml" ]]; then
  echo "Vendored ToolRL source not found at $TOOLRL_DIR" >&2
  exit 1
fi

cd "$TOOLRL_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export DATA_DIR="${DATA_DIR:-./dataset/rlla_4k}"
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-toolrl-grpo-smoke}"

N_GPUS="${N_GPUS:-1}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
ROLLOUT_N="${ROLLOUT_N:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
SAVE_FREQ="${SAVE_FREQ:-5}"
TEST_FREQ="${TEST_FREQ:-5}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"
OBJECTIVE="${OBJECTIVE:-grpo}"
OPTIMIZER="${OPTIMIZER:-adamw}"
SOFT_MUON_P="${SOFT_MUON_P:-0.5}"
VPO_WEIGHT_SAMPLES="${VPO_WEIGHT_SAMPLES:-16}"
VPO_DIRICHLET_ALPHA="${VPO_DIRICHLET_ALPHA:-1.0}"
VPO_SEED="${VPO_SEED:-0}"
if [[ -z "${USE_KL_LOSS:-}" ]]; then
  if [[ "$OBJECTIVE" == "vpo" ]]; then
    USE_KL_LOSS=true
  else
    USE_KL_LOSS=false
  fi
fi

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator="$OBJECTIVE" \
  algorithm.vpo_weight_samples="$VPO_WEIGHT_SAMPLES" \
  algorithm.vpo_dirichlet_alpha="$VPO_DIRICHLET_ALPHA" \
  algorithm.vpo_seed="$VPO_SEED" \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/test.parquet" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.val_batch_size="$VAL_BATCH_SIZE" \
  data.max_prompt_length=2048 \
  data.max_response_length=1024 \
  actor_rollout_ref.model.path="$BASE_MODEL" \
  actor_rollout_ref.actor.optim.optimizer="$OPTIMIZER" \
  actor_rollout_ref.actor.optim.soft_muon_p="$SOFT_MUON_P" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.use_kl_loss="$USE_KL_LOSS" \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.grad_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP_SIZE" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.critic_warmup=0 \
  trainer.logger='["console"]' \
  trainer.project_name=toolrl-vpo-muon \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes=1 \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ"
