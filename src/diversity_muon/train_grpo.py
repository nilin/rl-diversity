from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import GRPOConfig, GRPOTrainer

from diversity_muon.config import load_config, resolve_batch_config
from diversity_muon.maze import build_dataset
from diversity_muon.objectives import MazeReward, RewardConfig
from diversity_muon.optim import build_optimizer, build_scheduler
from diversity_muon.prompting import format_chat_prompt
from diversity_muon.training_eval import DiversityEvalCallback, RolloutTraceCallback


def _resolve_bf16(requested: bool) -> bool:
    if not requested:
        return False
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return True
    print("bf16 requested in config, but this environment does not support it; using fp32.")
    return False


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE") or os.environ.get("ACCELERATE_NUM_PROCESSES") or "1")


def _is_main_process() -> bool:
    return int(os.environ.get("RANK") or "0") == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    parser.add_argument("--seed", type=int, default=None, help="Override the config seed.")
    parser.add_argument("--output-dir", default=None, help="Override the config output_dir.")
    parser.add_argument("--run-name", default=None, help="Override the config run_name.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)
    if args.output_dir is not None:
        cfg = replace(cfg, output_dir=args.output_dir)
    if args.run_name is not None:
        cfg = replace(cfg, run_name=args.run_name)
    world_size = _world_size()
    cfg = resolve_batch_config(cfg, world_size=world_size)
    if _is_main_process():
        effective_batch = (
            world_size * cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
        )
        print(
            "Resolved train batch: "
            f"world_size={world_size} "
            f"global_train_batch_size={cfg.global_train_batch_size} "
            f"per_device_train_batch_size={cfg.per_device_train_batch_size} "
            f"gradient_accumulation_steps={cfg.gradient_accumulation_steps} "
            f"num_generations={cfg.num_generations} "
            f"effective_batch={effective_batch}",
            flush=True,
        )
    set_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model, padding_side="left", trust_remote_code=True, fix_mistral_regex=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model,
        torch_dtype="auto",
        trust_remote_code=True,
    )

    train_dataset = build_dataset(
        start_seed=cfg.train_seed_start, count=cfg.train_size, size=cfg.maze_size
    )
    eval_dataset = build_dataset(
        start_seed=cfg.eval_seed_start, count=cfg.eval_size, size=cfg.maze_size
    )
    train_dataset = train_dataset.map(
        lambda row: {"prompt": format_chat_prompt(tokenizer, row["prompt"])}
    )
    eval_dataset = eval_dataset.map(
        lambda row: {"prompt": format_chat_prompt(tokenizer, row["prompt"])}
    )

    reward = MazeReward(
        RewardConfig(
            objective=cfg.objective,  # type: ignore[arg-type]
            vpo_weight_samples=cfg.vpo_weight_samples,
            seed=cfg.seed,
        )
    )

    training_args = GRPOConfig(
        output_dir=cfg.output_dir,
        run_name=cfg.run_name,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_generations=cfg.num_generations,
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_completion_length,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        beta=cfg.beta,
        epsilon=cfg.epsilon,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        use_vllm=cfg.use_vllm,
        bf16=_resolve_bf16(cfg.bf16),
        report_to=cfg.report_to,
        seed=cfg.seed,
        disable_tqdm=not sys.stderr.isatty(),
        save_only_model=True,
        remove_unused_columns=False,
    )

    optimizer = build_optimizer(
        model,
        optimizer_name=cfg.optimizer,  # type: ignore[arg-type]
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        muon_momentum=cfg.muon_momentum,
        soft_muon_power=cfg.soft_muon_power,
        soft_muon_mix=cfg.soft_muon_mix,
        soft_muon_ns_iterations=cfg.soft_muon_ns_iterations,
        soft_muon_ns_coefficients=cfg.soft_muon_ns_coefficients,
        soft_muon_coefficients=cfg.soft_muon_coefficients,
        soft_muon_tail_coefficient=cfg.soft_muon_tail_coefficient,
    )
    scheduler = build_scheduler(
        optimizer, scheduler_type="constant", num_training_steps=cfg.max_steps
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        reward_funcs=reward,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizers=(optimizer, scheduler),
    )
    eval_row_count = min(
        max(cfg.diversity_eval_prompts, cfg.trace_rollout_examples), len(eval_dataset)
    )
    eval_rows = [dict(row) for row in eval_dataset.select(range(eval_row_count))]
    if cfg.trace_rollout_examples > 0:
        trainer.add_callback(
            RolloutTraceCallback(
                tokenizer=tokenizer,
                rows=eval_rows[: cfg.trace_rollout_examples],
                output_dir=cfg.output_dir,
                max_new_tokens=cfg.max_completion_length,
                temperature=cfg.diversity_eval_temperature,
                top_p=cfg.diversity_eval_top_p,
                objective=cfg.objective,
                vpo_weight_samples=cfg.vpo_weight_samples,
                seed=cfg.seed,
            )
        )
    if cfg.diversity_eval_steps > 0:
        trainer.add_callback(
            DiversityEvalCallback(
                tokenizer=tokenizer,
                eval_rows=eval_rows[: cfg.diversity_eval_prompts],
                output_dir=cfg.output_dir,
                eval_steps=cfg.diversity_eval_steps,
                samples_per_prompt=cfg.diversity_eval_samples_per_prompt,
                max_new_tokens=cfg.max_completion_length,
                temperature=cfg.diversity_eval_temperature,
                top_p=cfg.diversity_eval_top_p,
            )
        )
    trainer.train()


if __name__ == "__main__":
    main()
