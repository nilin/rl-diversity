from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

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
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampled generation.")
    parser.add_argument("--maze-size", type=int, default=9)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help=(
            "Device for evaluation. Use auto to shard over all visible GPUs, cuda for "
            "single-GPU inference, or cpu for CPU."
        ),
    )
    parser.add_argument(
        "--best-at-ks",
        default=",".join(str(value) for value in BEST_AT_KS),
        help=(
            "Comma-separated k values for paper-style best@k over candidate routes, "
            "averaged over rollout-chain pools."
        ),
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    best_at_ks = parse_best_at_ks(args.best_at_ks)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, padding_side="left", trust_remote_code=True, fix_mistral_regex=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict[str, Any] = {"torch_dtype": "auto", "trust_remote_code": True}
    if args.device == "auto":
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if args.device != "auto":
        model = model.to(args.device)
    model.eval()

    dataset = build_dataset(start_seed=args.seed_start, count=args.num_prompts, size=args.maze_size)
    weights = np.full((4,), 0.25, dtype=np.float32)
    prompt_metrics = []

    for row in dataset:
        prompt = format_chat_prompt(tokenizer, row["prompt"])
        encoded = tokenizer([prompt], return_tensors="pt").to(_model_device(model))
        all_vectors = []
        completions = []
        completion_route_scores = []
        completion_route_vectors = []
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=args.samples_per_prompt,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for generated_ids in generated:
            completion_ids = generated_ids[encoded["input_ids"].shape[1] :]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
            completions.append(completion)
            route_vectors = score_completion_routes(
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
            all_vectors.extend(route_vectors)
            completion_route_scores.append([float(np.dot(weights, vec)) for vec in route_vectors])
            completion_route_vectors.append(vector_lists(route_vectors))

        prompt_metrics.append(
            build_prompt_metrics(
                seed=row["seed"],
                all_vectors=all_vectors,
                completions=completions,
                completion_route_scores=completion_route_scores,
                completion_route_vectors=completion_route_vectors,
                best_at_ks=best_at_ks,
            )
        )

    summary = {
        "model": args.model,
        "num_prompts": args.num_prompts,
        "samples_per_prompt": args.samples_per_prompt,
        "mean_diversity": mean_metric(prompt_metrics, "diversity"),
        **{
            f"mean_best_at_{k}": mean_metric(prompt_metrics, f"best_at_{k}")
            for k in best_at_ks
        },
        **{
            f"mean_route_best_at_{k}": mean_metric(prompt_metrics, f"route_best_at_{k}")
            for k in best_at_ks
        },
        **{
            f"mean_rollout_best_at_{k}": mean_metric(prompt_metrics, f"rollout_best_at_{k}")
            for k in best_at_ks
        },
        "mean_completion_positive_rate": mean_metric(prompt_metrics, "completion_positive_rate"),
        "mean_route_positive_rate": mean_metric(prompt_metrics, "route_positive_rate"),
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


def paper_best_at_k_values(completion_route_scores: list[list[float]], k: int) -> list[float]:
    """Return per-pool paper-style best@k values for multi-answer rollouts.

    Each rollout is a chain containing m route candidates. For k <= m, the pools are
    the first k candidates of each rollout. For larger k, pools are formed from
    whole rollout chains; if k is not a multiple of m, the final chain contributes
    only its first remaining candidates, so ordered chain selections are averaged.
    """
    rollouts = [list(route_scores) for route_scores in completion_route_scores if route_scores]
    if not rollouts:
        return []

    flat_scores = [score for route_scores in rollouts for score in route_scores]
    if k >= len(flat_scores):
        return [float(max(flat_scores))]

    routes_per_rollout = max(len(route_scores) for route_scores in rollouts)
    if k <= routes_per_rollout:
        return [
            float(max(route_scores[: min(k, len(route_scores))]))
            for route_scores in rollouts
            if route_scores[: min(k, len(route_scores))]
        ]

    rollout_pool_size = (k + routes_per_rollout - 1) // routes_per_rollout
    full_rollout_count, partial_route_count = divmod(k, routes_per_rollout)
    rollout_indices = range(len(rollouts))

    if partial_route_count == 0:
        return [
            float(max(score for index in indices for score in rollouts[index]))
            for indices in itertools.combinations(rollout_indices, full_rollout_count)
        ]

    values: list[float] = []
    for indices in itertools.permutations(rollout_indices, rollout_pool_size):
        selected_scores = [
            score for index in indices[:full_rollout_count] for score in rollouts[index]
        ]
        selected_scores.extend(rollouts[indices[-1]][:partial_route_count])
        values.append(float(max(selected_scores)))
    return values


def build_prompt_metrics(
    *,
    seed: int,
    all_vectors: list[Any],
    completions: list[str] | None = None,
    completion_route_scores: list[list[float]],
    completion_route_vectors: list[list[list[float]]] | None = None,
    best_at_ks: tuple[int, ...],
) -> dict[str, Any]:
    completion_scores = [
        max(route_scores) if route_scores else 0.0 for route_scores in completion_route_scores
    ]
    route_scores = [score for route_scores in completion_route_scores for score in route_scores]
    best_at_values = {
        str(k): paper_best_at_k_values(completion_route_scores, k) for k in best_at_ks
    }
    rollout_count = len(completion_route_scores)
    completions = completions if completions is not None else [""] * rollout_count
    completion_route_vectors = (
        completion_route_vectors
        if completion_route_vectors is not None
        else [[] for _ in range(rollout_count)]
    )
    return {
        "seed": seed,
        "diversity": pairwise_l1_diversity(all_vectors),
        "rollout_scores": completion_scores,
        "completion_scores": completion_scores,
        "completion_route_scores": completion_route_scores,
        "completion_route_vectors": completion_route_vectors,
        "rollouts": [
            {
                "rollout_index": index,
                "completion": completions[index] if index < len(completions) else "",
                "score": completion_scores[index] if index < len(completion_scores) else 0.0,
                "route_scores": completion_route_scores[index],
                "route_vectors": completion_route_vectors[index],
            }
            for index in range(rollout_count)
        ],
        "route_scores": route_scores,
        "best_at_values": best_at_values,
        "completion_positive_rate": positive_rate(completion_scores),
        "route_positive_rate": positive_rate(route_scores),
        **{f"best_at_{k}": mean_values(best_at_values[str(k)]) for k in best_at_ks},
        **{f"route_best_at_{k}": mean_values(best_at_values[str(k)]) for k in best_at_ks},
        **{f"rollout_best_at_{k}": best_at_k(completion_scores, k) for k in best_at_ks},
    }


def positive_rate(scores: list[float]) -> float:
    return float(np.mean([score > 0.0 for score in scores])) if scores else 0.0


def mean_values(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def vector_lists(vectors: list[Any]) -> list[list[float]]:
    return [[float(value) for value in vector] for vector in vectors]


BEST_AT_KS = (1, 3, 6, 9)


def parse_best_at_ks(value: str) -> tuple[int, ...]:
    ks = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not ks:
        raise ValueError("--best-at-ks must contain at least one integer")
    if any(k < 1 for k in ks):
        raise ValueError("--best-at-ks values must be >= 1")
    return ks


def mean_metric(rows: list[dict[str, float]], key: str) -> float:
    return float(np.mean([row[key] for row in rows])) if rows else 0.0


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    main()
