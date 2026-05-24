from __future__ import annotations

import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from diversity_muon.config import load_config
from diversity_muon.maze import build_dataset
from diversity_muon.objectives import MazeReward, RewardConfig
from diversity_muon.optim import build_optimizer, build_scheduler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    args = parser.parse_args()
    cfg = load_config(args.config)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model,
        torch_dtype="auto",
        trust_remote_code=True,
    )

    train_dataset = build_dataset(start_seed=cfg.train_seed_start, count=cfg.train_size)
    eval_dataset = build_dataset(start_seed=cfg.eval_seed_start, count=cfg.eval_size)

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
        bf16=cfg.bf16,
        report_to=cfg.report_to,
        seed=cfg.seed,
        remove_unused_columns=False,
    )

    optimizer = build_optimizer(
        model,
        optimizer_name=cfg.optimizer,  # type: ignore[arg-type]
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
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
    trainer.train()


if __name__ == "__main__":
    main()
