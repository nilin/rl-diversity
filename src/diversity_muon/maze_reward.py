from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

import numpy as np

MOVE_DELTAS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}

ROUTE_RE = re.compile(r"<route_(\d+)>(.*?)</route_\1>", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class RouteScore:
    completion: float
    gold: float
    diamond: float
    avoid_lava: float

    @property
    def vector(self) -> np.ndarray:
        return np.array(
            [self.completion, self.gold, self.diamond, self.avoid_lava], dtype=np.float32
        )


def parse_routes(completion: str, *, expected_routes: int = 3) -> list[list[str]]:
    matches = ROUTE_RE.findall(completion or "")
    by_index: dict[int, list[str]] = {}
    for raw_idx, body in matches:
        idx = int(raw_idx)
        moves = [token.upper() for token in re.findall(r"[A-Za-z]+", body)]
        by_index[idx] = [move for move in moves if move in MOVE_DELTAS]
    return [by_index.get(i, []) for i in range(1, expected_routes + 1)]


def score_route(
    moves: list[str],
    *,
    grid: list[str],
    start: list[int] | tuple[int, int],
    end: list[int] | tuple[int, int],
    step_budget: int,
    gold_total: int,
    diamond_total: int,
    lava_total: int,
) -> RouteScore:
    cells = [row.split() for row in grid]
    size = len(cells)
    r, c = int(start[0]), int(start[1])
    end_coord = (int(end[0]), int(end[1]))
    gold_seen: set[tuple[int, int]] = set()
    diamond_seen: set[tuple[int, int]] = set()
    lava_seen: set[tuple[int, int]] = set()

    for move in moves[:step_budget]:
        dr, dc = MOVE_DELTAS[move]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= size or nc < 0 or nc >= len(cells[nr]):
            return RouteScore(0.0, 0.0, 0.0, 0.0)
        if cells[nr][nc] == "#":
            return RouteScore(0.0, 0.0, 0.0, 0.0)
        r, c = nr, nc
        tile = cells[r][c]
        if tile == "G":
            gold_seen.add((r, c))
        elif tile == "D":
            diamond_seen.add((r, c))
        elif tile == "L":
            lava_seen.add((r, c))
        if (r, c) == end_coord:
            break

    if (r, c) != end_coord:
        return RouteScore(0.0, 0.0, 0.0, 0.0)

    gold = len(gold_seen) / max(1, gold_total)
    diamond = len(diamond_seen) / max(1, diamond_total)
    avoid_lava = 1.0 - (len(lava_seen) / max(1, lava_total))
    return RouteScore(1.0, min(1.0, gold), min(1.0, diamond), max(0.0, avoid_lava))


def score_completion_routes(
    completion: str,
    *,
    grid: list[str],
    start: list[int] | tuple[int, int],
    end: list[int] | tuple[int, int],
    step_budget: int,
    gold_total: int,
    diamond_total: int,
    lava_total: int,
    expected_routes: int = 3,
) -> list[np.ndarray]:
    routes = parse_routes(completion, expected_routes=expected_routes)
    return [
        score_route(
            route,
            grid=grid,
            start=start,
            end=end,
            step_budget=step_budget,
            gold_total=gold_total,
            diamond_total=diamond_total,
            lava_total=lava_total,
        ).vector
        for route in routes
    ]


def pairwise_l1_diversity(vectors: list[np.ndarray]) -> float:
    nonzero = [np.asarray(v, dtype=np.float32) for v in vectors if np.asarray(v).shape == (4,)]
    if len(nonzero) < 2:
        return 0.0
    distances = [float(np.abs(a - b).sum()) for a, b in itertools.combinations(nonzero, 2)]
    return float(np.mean(distances)) if distances else 0.0
