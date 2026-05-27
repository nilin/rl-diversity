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
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-1.7B}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-toolrl-qwen3-paper}"

ROLLOUT_N_SET="${ROLLOUT_N+x}"
TRAIN_BATCH_SIZE_SET="${TRAIN_BATCH_SIZE+x}"
VAL_BATCH_SIZE_SET="${VAL_BATCH_SIZE+x}"
PPO_MINI_BATCH_SIZE_SET="${PPO_MINI_BATCH_SIZE+x}"
PPO_MICRO_BATCH_SIZE_SET="${PPO_MICRO_BATCH_SIZE+x}"
MAX_RESPONSE_LENGTH_SET="${MAX_RESPONSE_LENGTH+x}"

N_GPUS="${N_GPUS:-1}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
ROLLOUT_N="${ROLLOUT_N:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-80}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-8}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-2}"
SAVE_FREQ="${SAVE_FREQ:--1}"
TEST_FREQ="${TEST_FREQ:--1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
VAL_AFTER_TRAIN="${VAL_AFTER_TRAIN:-false}"
MULTI_ANSWER_COUNT="${MULTI_ANSWER_COUNT:-3}"
PAPER_TOOLRL_REWARD="${PAPER_TOOLRL_REWARD:-1}"
OBJECTIVE="${OBJECTIVE:-grpo}"
OPTIMIZER="${OPTIMIZER:-adamw}"
SOFT_MUON_P="${SOFT_MUON_P:-0.5}"
MUON_NS_COEFFICIENTS="${MUON_NS_COEFFICIENTS:-[2.0,-1.5,0.5]}"
MUON_NS_STEPS="${MUON_NS_STEPS:-12}"
VPO_WEIGHT_SAMPLES="${VPO_WEIGHT_SAMPLES:-16}"
VPO_DIRICHLET_ALPHA="${VPO_DIRICHLET_ALPHA:-1.0}"
VPO_SEED="${VPO_SEED:-0}"
USE_KL_LOSS="${USE_KL_LOSS:-true}"
export MULTI_ANSWER_COUNT
export PAPER_TOOLRL_REWARD

ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-vllm}"
ROLLOUT_MICRO_BATCH_SIZE="${ROLLOUT_MICRO_BATCH_SIZE:-1}"
if [[ "$BASE_MODEL" == Qwen/Qwen3-* && "$ROLLOUT_BACKEND" == "vllm" ]]; then
  # ToolRL's vendored vLLM integration is pinned to vLLM 0.6.3, which does not
  # support Qwen3. Use HF rollout for Qwen3 unless explicitly overridden.
  ROLLOUT_BACKEND=hf
fi
if [[ "$BASE_MODEL" == Qwen/Qwen3-* && "$ROLLOUT_BACKEND" == "hf" ]]; then
  # Keep the no-override path usable as a smoke test. The paper-sized defaults
  # remain available by setting these environment variables explicitly.
  [[ -z "$ROLLOUT_N_SET" ]] && ROLLOUT_N=2
  [[ -z "$TRAIN_BATCH_SIZE_SET" ]] && TRAIN_BATCH_SIZE=4
  [[ -z "$VAL_BATCH_SIZE_SET" ]] && VAL_BATCH_SIZE=4
  [[ -z "$PPO_MINI_BATCH_SIZE_SET" ]] && PPO_MINI_BATCH_SIZE="$TRAIN_BATCH_SIZE"
  [[ -z "$PPO_MICRO_BATCH_SIZE_SET" ]] && PPO_MICRO_BATCH_SIZE="$PPO_MINI_BATCH_SIZE"
  [[ -z "$MAX_RESPONSE_LENGTH_SET" ]] && MAX_RESPONSE_LENGTH=256
fi

TRAINING_STEP_ARG=()
if [[ -n "$TOTAL_TRAINING_STEPS" ]]; then
  TRAINING_STEP_ARG=(trainer.total_training_steps="$TOTAL_TRAINING_STEPS")
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
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.multi_answer_count="$MULTI_ANSWER_COUNT" \
  actor_rollout_ref.model.path="$BASE_MODEL" \
  actor_rollout_ref.actor.optim.optimizer="$OPTIMIZER" \
  actor_rollout_ref.actor.optim.soft_muon_p="$SOFT_MUON_P" \
  actor_rollout_ref.actor.optim.muon_ns_coefficients="$MUON_NS_COEFFICIENTS" \
  actor_rollout_ref.actor.optim.muon_ns_steps="$MUON_NS_STEPS" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size="$PPO_MICRO_BATCH_SIZE" \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.use_kl_loss="$USE_KL_LOSS" \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.grad_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP_SIZE" \
  actor_rollout_ref.rollout.name="$ROLLOUT_BACKEND" \
  actor_rollout_ref.rollout.micro_batch_size="$ROLLOUT_MICRO_BATCH_SIZE" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
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
  trainer.test_freq="$TEST_FREQ" \
  trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  trainer.val_after_train="$VAL_AFTER_TRAIN" \
  "${TRAINING_STEP_ARG[@]}"
