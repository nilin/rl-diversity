from __future__ import annotations

import numpy as np

from diversity_muon.eval_diversity import build_prompt_metrics, paper_best_at_k_values


def test_prompt_metrics_use_combinatorial_route_level_best_at_k() -> None:
    vectors = [
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([0.4, 0.4, 0.4, 0.4]),
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0]),
    ]
    completion_route_scores = [
        [0.1, 0.2, 0.0],
        [0.4, 0.0, 0.3],
        [0.0, 0.5, 0.2],
    ]
    metrics = build_prompt_metrics(
        seed=123,
        all_vectors=vectors,
        completion_route_scores=completion_route_scores,
        best_at_ks=(1, 3, 4, 6, 9),
    )

    assert metrics["completion_scores"] == [0.2, 0.4, 0.5]
    assert metrics["rollout_scores"] == [0.2, 0.4, 0.5]
    assert metrics["route_scores"] == [0.1, 0.2, 0.0, 0.4, 0.0, 0.3, 0.0, 0.5, 0.2]
    assert metrics["best_at_values"]["1"] == [0.1, 0.4, 0.0]
    assert metrics["best_at_values"]["3"] == [0.2, 0.4, 0.5]
    assert metrics["best_at_values"]["4"] == [0.4, 0.2, 0.4, 0.4, 0.5, 0.5]
    assert metrics["best_at_values"]["6"] == [0.4, 0.5, 0.5]
    assert metrics["best_at_values"]["9"] == [0.5]
    assert np.isclose(metrics["best_at_1"], np.mean([0.1, 0.4, 0.0]))
    assert np.isclose(metrics["best_at_3"], np.mean([0.2, 0.4, 0.5]))
    assert np.isclose(metrics["best_at_4"], np.mean([0.4, 0.2, 0.4, 0.4, 0.5, 0.5]))
    assert np.isclose(metrics["best_at_6"], np.mean([0.4, 0.5, 0.5]))
    assert metrics["best_at_9"] == 0.5
    assert metrics["route_best_at_6"] == metrics["best_at_6"]
    assert metrics["rollout_best_at_3"] == 0.5
    assert metrics["completion_positive_rate"] == 1.0
    assert metrics["route_positive_rate"] == 6 / 9


def test_best_at_k_clamps_to_available_routes() -> None:
    assert paper_best_at_k_values([[0.1, 0.2, 0.0], [0.4, 0.0, 0.3]], 30) == [0.4]
