from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import TrainerCallback

from diversity_muon.eval_diversity import build_prompt_metrics, vector_lists
from diversity_muon.maze_reward import pairwise_l1_diversity, score_completion_routes


class DiversityEvalCallback(TrainerCallback):
    """Periodically sample fixed prompts and log Figure-6-style diversity metrics."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        eval_rows: list[dict[str, Any]],
        output_dir: str,
        eval_steps: int,
        samples_per_prompt: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        expected_routes: int = 3,
    ) -> None:
        self.tokenizer = tokenizer
        self.eval_rows = eval_rows
        self.output_path = Path(output_dir) / "diversity_eval.jsonl"
        self.eval_steps = eval_steps
        self.samples_per_prompt = samples_per_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.expected_routes = expected_routes
        self._last_step = -1

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
        if self.eval_steps <= 0 or state.global_step <= 0:
            return control
        if state.global_step == self._last_step or state.global_step % self.eval_steps != 0:
            return control
        if not getattr(state, "is_world_process_zero", True):
            return control

        model = kwargs.get("model")
        if model is None:
            return control

        self._last_step = state.global_step
        metrics = evaluate_diversity(
            model=model,
            tokenizer=self.tokenizer,
            rows=self.eval_rows,
            samples_per_prompt=self.samples_per_prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            expected_routes=self.expected_routes,
        )
        metrics = {"step": int(state.global_step), **metrics}
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")
        print(
            "diversity_eval "
            f"step={metrics['step']} "
            f"mean_chain_diversity={metrics['train_eval/mean_chain_diversity']:.4f} "
            f"mean_best_at_3={metrics['train_eval/mean_best_at_3']:.4f} "
            f"mean_best_at_9={metrics['train_eval/mean_best_at_9']:.4f}",
            flush=True,
        )
        return control


def evaluate_diversity(
    *,
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    samples_per_prompt: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    expected_routes: int = 3,
) -> dict[str, Any]:
    weights = np.full((4,), 0.25, dtype=np.float32)
    was_training = model.training
    model.eval()

    prompt_metrics: list[dict[str, float]] = []
    try:
        for row in rows:
            encoded = tokenizer([row["prompt"]], return_tensors="pt").to(_model_device(model))
            chain_diversities: list[float] = []
            chain_best_uniforms: list[float] = []
            all_vectors = []
            completions: list[str] = []
            completion_route_scores: list[list[float]] = []
            completion_route_vectors: list[list[list[float]]] = []
            for _ in range(samples_per_prompt):
                with torch.no_grad():
                    generated = model.generate(
                        **encoded,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                completion_ids = generated[0, encoded["input_ids"].shape[1] :]
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
                    expected_routes=expected_routes,
                )
                route_scores = [float(np.dot(weights, vec)) for vec in route_vectors]
                chain_diversities.append(pairwise_l1_diversity(route_vectors))
                chain_best_uniforms.append(max(route_scores) if route_scores else 0.0)
                completion_route_scores.append(route_scores)
                completion_route_vectors.append(vector_lists(route_vectors))
                all_vectors.extend(route_vectors)

            prompt_metric = build_prompt_metrics(
                seed=int(row["seed"]),
                all_vectors=all_vectors,
                completions=completions,
                completion_route_scores=completion_route_scores,
                completion_route_vectors=completion_route_vectors,
                best_at_ks=TRAIN_BEST_AT_KS,
            )
            prompt_metric.update(
                {
                    "chain_diversity": float(np.mean(chain_diversities)),
                    "pooled_diversity": pairwise_l1_diversity(all_vectors),
                    "best_uniform_in_chain": float(np.mean(chain_best_uniforms)),
                }
            )
            prompt_metrics.append(prompt_metric)
    finally:
        if was_training:
            model.train()

    return {
        "train_eval/mean_chain_diversity": mean_metric(prompt_metrics, "chain_diversity"),
        "train_eval/mean_pooled_diversity": mean_metric(prompt_metrics, "pooled_diversity"),
        "train_eval/mean_best_uniform_in_chain": mean_metric(
            prompt_metrics, "best_uniform_in_chain"
        ),
        **{
            f"train_eval/mean_best_at_{k}": mean_metric(prompt_metrics, f"best_at_{k}")
            for k in TRAIN_BEST_AT_KS
        },
        "train_eval/prompts": float(len(rows)),
        "train_eval/samples_per_prompt": float(samples_per_prompt),
        "train_eval/expected_routes": float(expected_routes),
        "train_eval/prompt_metrics": prompt_metrics,
    }


def best_at_k(scores: list[float], k: int) -> float:
    if not scores:
        return 0.0
    return float(max(scores[: min(k, len(scores))]))


TRAIN_BEST_AT_KS = (3, 6, 9)


def mean_metric(rows: list[dict[str, float]], key: str) -> float:
    return float(np.mean([row[key] for row in rows])) if rows else 0.0


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
