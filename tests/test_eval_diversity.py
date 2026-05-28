from __future__ import annotations

import numpy as np

from diversity_muon.eval_diversity import build_prompt_metrics


def test_prompt_metrics_use_route_level_best_at_k() -> None:
    vectors = [
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([0.4, 0.4, 0.4, 0.4]),
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0]),
    ]
    metrics = build_prompt_metrics(
        seed=123,
        all_vectors=vectors,
        completion_route_scores=[
            [0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0],
        ],
        best_at_ks=(1, 2, 3, 6),
    )

    assert metrics["completion_scores"] == [0.0, 0.4]
    assert metrics["rollout_scores"] == [0.0, 0.4]
    assert metrics["route_scores"] == [0.0, 0.0, 0.0, 0.4, 0.0, 0.0]
    assert metrics["best_at_1"] == 0.0
    assert metrics["best_at_2"] == 0.0
    assert metrics["best_at_3"] == 0.0
    assert metrics["best_at_6"] == 0.4
    assert metrics["route_best_at_3"] == 0.0
    assert metrics["route_best_at_6"] == 0.4
    assert metrics["rollout_best_at_2"] == 0.4
    assert metrics["completion_positive_rate"] == 0.5
    assert metrics["route_positive_rate"] == 1 / 6
