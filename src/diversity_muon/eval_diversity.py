from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from diversity_muon.maze import build_dataset
from diversity_muon.maze_reward import pairwise_l1_diversity, score_completion_routes
from diversity_muon.prompting import format_chat_prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--samples-per-prompt", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=4242)
    parser.add_argument("--maze-size", type=int, default=9)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, padding_side="left", trust_remote_code=True, fix_mistral_regex=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    model.eval()

    dataset = build_dataset(start_seed=args.seed_start, count=args.num_prompts, size=args.maze_size)
    weights = np.full((4,), 0.25, dtype=np.float32)
    prompt_metrics = []

    for row in dataset:
        prompt = format_chat_prompt(tokenizer, row["prompt"])
        encoded = tokenizer([prompt], return_tensors="pt").to(model.device)
        all_vectors = []
        for _ in range(args.samples_per_prompt):
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            completion_ids = generated[0, encoded["input_ids"].shape[1] :]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
            all_vectors.extend(
                score_completion_routes(
                    completion,
                    grid=row["grid"],
                    start=row["start"],
                    end=row["end"],
                    step_budget=row["step_budget"],
                    gold_total=row["gold_total"],
                    diamond_total=row["diamond_total"],
                    lava_total=row["lava_total"],
                    expected_routes=3,
                )
            )

        scalar_scores = [float(np.dot(weights, vec)) for vec in all_vectors]
        prompt_metrics.append(
            {
                "seed": row["seed"],
                "diversity": pairwise_l1_diversity(all_vectors),
                "best_at_3": best_at_k(scalar_scores, 3),
                "best_at_5": best_at_k(scalar_scores, 5),
                "best_at_10": best_at_k(scalar_scores, 10),
                "best_at_30": best_at_k(scalar_scores, 30),
            }
        )

    summary = {
        "model": args.model,
        "num_prompts": args.num_prompts,
        "samples_per_prompt": args.samples_per_prompt,
        "mean_diversity": mean_metric(prompt_metrics, "diversity"),
        "mean_best_at_3": mean_metric(prompt_metrics, "best_at_3"),
        "mean_best_at_5": mean_metric(prompt_metrics, "best_at_5"),
        "mean_best_at_10": mean_metric(prompt_metrics, "best_at_10"),
        "mean_best_at_30": mean_metric(prompt_metrics, "best_at_30"),
        "prompts": prompt_metrics,
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def best_at_k(scores: list[float], k: int) -> float:
    if not scores:
        return 0.0
    return float(max(scores[: min(k, len(scores))]))


def mean_metric(rows: list[dict[str, float]], key: str) -> float:
    return float(np.mean([row[key] for row in rows])) if rows else 0.0


if __name__ == "__main__":
    main()
