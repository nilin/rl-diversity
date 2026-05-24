from __future__ import annotations

import json
import os

import numpy as np

from diversity_muon.maze_reward import pairwise_l1_diversity, score_completion_routes
from diversity_muon.objectives import MazeReward, RewardConfig


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth,
    extra_info: dict | None = None,
    **_: object,
):
    if data_source != "maze":
        return {"score": 0.0, "maze_reward_space_diversity": 0.0}

    maze = _load_ground_truth(ground_truth, extra_info)
    objective = os.environ.get("DIVERSITY_OBJECTIVE", "multirlvr")
    reward = MazeReward(
        RewardConfig(
            objective=objective,  # type: ignore[arg-type]
            vpo_weight_samples=int(os.environ.get("DIVERSITY_VPO_WEIGHT_SAMPLES", "16")),
            seed=int(os.environ.get("DIVERSITY_SEED", "0")),
        )
    )
    route_vectors = score_completion_routes(
        solution_str,
        grid=maze["grid"],
        start=maze["start"],
        end=maze["end"],
        step_budget=int(maze["step_budget"]),
        gold_total=int(maze["gold_total"]),
        diamond_total=int(maze["diamond_total"]),
        lava_total=int(maze["lava_total"]),
        expected_routes=3,
    )
    uniform = np.full((4,), 0.25, dtype=np.float32)
    return {
        "score": reward.score_set(route_vectors),
        "maze_reward_space_diversity": pairwise_l1_diversity(route_vectors),
        "maze_best_uniform_in_chain": float(max(np.dot(uniform, vec) for vec in route_vectors)),
    }


def _load_ground_truth(ground_truth, extra_info: dict | None) -> dict:
    if isinstance(ground_truth, dict):
        return ground_truth
    if isinstance(ground_truth, str):
        return json.loads(ground_truth)
    if extra_info and "ground_truth" in extra_info:
        value = extra_info["ground_truth"]
        return json.loads(value) if isinstance(value, str) else value
    raise ValueError("Maze reward requires JSON ground_truth metadata.")
