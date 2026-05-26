from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from diversity_muon.maze_reward import pairwise_l1_diversity, score_completion_routes

ObjectiveName = Literal["multirlvr", "vpo"]


@dataclass(frozen=True)
class RewardConfig:
    objective: ObjectiveName
    routes: int = 3
    scalar_weights: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)
    vpo_weight_samples: int = 16
    dirichlet_alpha: float = 1.0
    seed: int = 0


class MazeReward:
    def __init__(self, config: RewardConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.__name__ = f"maze_{config.objective}_reward"

    def __call__(
        self,
        completions: list[str],
        grid: list[list[str]],
        start: list[list[int]],
        end: list[list[int]],
        step_budget: list[int],
        gold_total: list[int],
        diamond_total: list[int],
        lava_total: list[int],
        log_metric=None,
        **_: object,
    ) -> list[float]:
        rewards: list[float] = []
        diversities: list[float] = []
        best_uniforms: list[float] = []
        for i, completion in enumerate(completions):
            completion_text = completion_to_text(completion)
            route_vectors = score_completion_routes(
                completion_text,
                grid=grid[i],
                start=start[i],
                end=end[i],
                step_budget=int(step_budget[i]),
                gold_total=int(gold_total[i]),
                diamond_total=int(diamond_total[i]),
                lava_total=int(lava_total[i]),
                expected_routes=self.config.routes,
            )
            reward = self.score_set(route_vectors)
            rewards.append(reward)
            diversities.append(pairwise_l1_diversity(route_vectors))
            best_uniforms.append(float(max(np.dot(self.weights, vec) for vec in route_vectors)))

        if log_metric:
            log_metric("maze_reward_space_diversity", float(np.mean(diversities)))
            log_metric("maze_best_uniform_in_chain", float(np.mean(best_uniforms)))
        return rewards

    @property
    def weights(self) -> np.ndarray:
        weights = np.asarray(self.config.scalar_weights, dtype=np.float32)
        return weights / weights.sum()

    def score_set(self, route_vectors: list[np.ndarray]) -> float:
        if self.config.objective == "multirlvr":
            return float(max(np.dot(self.weights, vec) for vec in route_vectors))
        if self.config.objective == "vpo":
            alpha = np.full((4,), self.config.dirichlet_alpha, dtype=np.float32)
            sampled_weights = self.rng.dirichlet(alpha, size=self.config.vpo_weight_samples)
            values = []
            for weights in sampled_weights:
                values.append(max(float(np.dot(weights, vec)) for vec in route_vectors))
            return float(np.mean(values))
        raise ValueError(f"Unsupported objective: {self.config.objective}")


def completion_to_text(completion: object) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        if completion and isinstance(completion[-1], dict):
            return str(completion[-1].get("content", ""))
        return " ".join(str(item) for item in completion)
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion or "")
