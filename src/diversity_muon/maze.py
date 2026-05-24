from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Iterable


Coord = tuple[int, int]


@dataclass(frozen=True)
class MazeExample:
    seed: int
    size: int
    grid: tuple[str, ...]
    start: Coord
    end: Coord
    step_budget: int
    gold_total: int
    diamond_total: int
    lava_total: int
    prompt: str

    def to_row(self) -> dict[str, object]:
        row = asdict(self)
        row["grid"] = list(self.grid)
        row["start"] = list(self.start)
        row["end"] = list(self.end)
        return row


def make_maze(seed: int, *, size: int = 9, step_budget: int = 20) -> MazeExample:
    """Create a deterministic Maze-style prompt.

    The VPO paper does not publish its exact generator. This generator deliberately
    keeps the same reward structure while remaining simple and deterministic.
    """

    rng = random.Random(seed)
    start = (0, 0)
    end = (size - 1, size - 1)
    grid = [["." for _ in range(size)] for _ in range(size)]
    grid[start[0]][start[1]] = "S"
    grid[end[0]][end[1]] = "E"

    protected = {start, end}
    _place_many(grid, rng, "G", 5, protected)
    _place_many(grid, rng, "D", 4, protected)
    _place_many(grid, rng, "L", 3, protected)
    _place_many(grid, rng, "B", 1, protected)

    rows = tuple(" ".join(row) for row in grid)
    prompt = format_prompt(
        rows,
        step_budget=step_budget,
        gold_total=5,
        diamond_total=4,
        lava_total=3,
        bonus_total=1,
    )
    return MazeExample(
        seed=seed,
        size=size,
        grid=rows,
        start=start,
        end=end,
        step_budget=step_budget,
        gold_total=5,
        diamond_total=4,
        lava_total=3,
        prompt=prompt,
    )


def _place_many(
    grid: list[list[str]], rng: random.Random, tile: str, count: int, protected: set[Coord]
) -> None:
    size = len(grid)
    placed = 0
    while placed < count:
        coord = (rng.randrange(size), rng.randrange(size))
        if coord in protected:
            continue
        r, c = coord
        if grid[r][c] != ".":
            continue
        grid[r][c] = tile
        placed += 1


def format_prompt(
    rows: Iterable[str],
    *,
    step_budget: int,
    gold_total: int,
    diamond_total: int,
    lava_total: int,
    bonus_total: int,
    routes: int = 3,
) -> str:
    rows = tuple(rows)
    grid_text = "\n".join(rows)
    return f"""You are navigating a {len(rows)}x9 grid maze.
S is the start. E is the exit. G is Gold. D is Diamond. L is Lava. B is Bonus.
Move with UP, DOWN, LEFT, RIGHT.

Grid:
{grid_text}

You have {step_budget} steps per route.
This maze has {gold_total} Gold, {diamond_total} Diamond, {lava_total} Lava, and {bonus_total} Bonus tiles.
Reason briefly about the maze, then provide {routes} genuinely different routes from S to E.
Each route is a sequence of UP/DOWN/LEFT/RIGHT moves (space-separated).
Wrap each route in numbered tags (<route_1>...</route_1>, <route_2>...</route_2>, <route_3>...</route_3>).
Inside each tag put ONLY moves (no arrows, no coordinates, no prose); any reasoning goes outside the tags.
Each route has its own {step_budget}-step budget and must reach E. A route scores zero if it does not reach E.
"""


def build_dataset(*, start_seed: int, count: int):
    from datasets import Dataset

    return Dataset.from_list([make_maze(start_seed + i).to_row() for i in range(count)])
