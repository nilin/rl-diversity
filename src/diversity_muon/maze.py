from __future__ import annotations

import random
from collections import deque
from collections.abc import Iterable
from dataclasses import asdict, dataclass

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


def make_maze(seed: int, *, size: int = 9, step_budget: int | None = None) -> MazeExample:
    """Create a deterministic Maze-style prompt.

    This follows the VPO appendix setup: a wall maze with extra cycles, endpoints in
    opposite corners, resource clusters near the other corners, and a budget that
    permits either the gold or diamond detour but not both.
    """

    if size < 5 or size % 2 == 0:
        raise ValueError("Maze size must be an odd integer >= 5.")
    rng = random.Random(seed)
    for _ in range(10_000):
        grid = _carve_wall_maze(rng, size=size)
        _add_cycles(grid, rng, count=rng.randint(18, 28))

        if rng.random() < 0.5:
            start, end = (0, 0), (size - 1, size - 1)
            resource_corners = [(0, size - 1), (size - 1, 0)]
        else:
            start, end = (0, size - 1), (size - 1, 0)
            resource_corners = [(0, 0), (size - 1, size - 1)]
        rng.shuffle(resource_corners)
        gold_corner, diamond_corner = resource_corners

        center = (size // 2, size // 2)
        for coord in [start, end, gold_corner, diamond_corner, center]:
            grid[coord[0]][coord[1]] = "."

        via_gold = _distance(grid, start, gold_corner) + _distance(grid, gold_corner, end)
        via_diamond = _distance(grid, start, diamond_corner) + _distance(grid, diamond_corner, end)
        via_both = (
            _distance(grid, start, gold_corner)
            + _distance(grid, gold_corner, diamond_corner)
            + _distance(grid, diamond_corner, end)
        )
        if _unreachable(via_gold, via_diamond, via_both):
            continue

        budget = step_budget if step_budget is not None else max(via_gold, via_diamond) + 7
        if via_both <= budget:
            continue

        protected = {start, end, center}
        gold_total = rng.randint(3, 5)
        diamond_total = rng.randint(3, 5)
        lava_total = rng.randint(3, 5)
        if not _place_in_radius(grid, rng, "G", gold_total, gold_corner, protected):
            continue
        protected |= _tile_coords(grid, "G")
        if not _place_in_radius(grid, rng, "D", diamond_total, diamond_corner, protected):
            continue
        protected |= _tile_coords(grid, "D")
        if not _place_lava(grid, rng, lava_total, protected):
            continue
        grid[center[0]][center[1]] = "B"
        grid[start[0]][start[1]] = "S"
        grid[end[0]][end[1]] = "E"

        if _distance(grid, start, end, avoid_lava=True) <= budget:
            _assert_valid_counts(grid, gold_total, diamond_total, lava_total)
            break
    else:
        raise RuntimeError(f"Failed to generate valid maze for seed {seed}.")

    rows = tuple(" ".join(row) for row in grid)
    prompt = format_prompt(
        rows,
        step_budget=budget,
        gold_total=gold_total,
        diamond_total=diamond_total,
        lava_total=lava_total,
        bonus_total=1,
    )
    return MazeExample(
        seed=seed,
        size=size,
        grid=rows,
        start=start,
        end=end,
        step_budget=budget,
        gold_total=gold_total,
        diamond_total=diamond_total,
        lava_total=lava_total,
        prompt=prompt,
    )


def _carve_wall_maze(rng: random.Random, *, size: int) -> list[list[str]]:
    grid = [["#" for _ in range(size)] for _ in range(size)]
    start = (rng.randrange(size), rng.randrange(size))
    grid[start[0]][start[1]] = "."
    frontier = list(_neighbors(start, size))
    rng.shuffle(frontier)
    while frontier:
        coord = frontier.pop(rng.randrange(len(frontier)))
        if grid[coord[0]][coord[1]] == ".":
            continue
        empty_neighbors = [
            neighbor
            for neighbor in _neighbors(coord, size)
            if grid[neighbor[0]][neighbor[1]] == "."
        ]
        if len(empty_neighbors) != 1:
            continue
        grid[coord[0]][coord[1]] = "."
        for neighbor in _neighbors(coord, size):
            if grid[neighbor[0]][neighbor[1]] == "#":
                frontier.append(neighbor)
    return grid


def _add_cycles(grid: list[list[str]], rng: random.Random, *, count: int) -> None:
    size = len(grid)
    candidates = [
        (r, c)
        for r in range(size)
        for c in range(size)
        if grid[r][c] == "#"
        and sum(1 for nr, nc in _neighbors((r, c), size) if grid[nr][nc] == ".") >= 2
    ]
    rng.shuffle(candidates)
    for r, c in candidates[:count]:
        grid[r][c] = "."


def _neighbors(coord: Coord, size: int) -> Iterable[Coord]:
    r, c = coord
    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= nr < size and 0 <= nc < size:
            yield nr, nc


def _distance(
    grid: list[list[str]], start: Coord, end: Coord, *, avoid_lava: bool = False
) -> int:
    size = len(grid)
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        coord, dist = queue.popleft()
        if coord == end:
            return dist
        for nr, nc in _neighbors(coord, size):
            if (nr, nc) in seen or grid[nr][nc] == "#":
                continue
            if avoid_lava and grid[nr][nc] == "L":
                continue
            seen.add((nr, nc))
            queue.append(((nr, nc), dist + 1))
    return 10**9


def _unreachable(*distances: int) -> bool:
    return any(distance >= 10**9 for distance in distances)


def _place_in_radius(
    grid: list[list[str]],
    rng: random.Random,
    tile: str,
    count: int,
    center: Coord,
    protected: set[Coord],
    radius: int = 2,
) -> bool:
    size = len(grid)
    candidates = [
        (r, c)
        for r in range(size)
        for c in range(size)
        if abs(r - center[0]) + abs(c - center[1]) <= radius
        and (r, c) not in protected
        and grid[r][c] == "."
    ]
    rng.shuffle(candidates)
    if len(candidates) < count:
        return False
    for r, c in candidates[:count]:
        grid[r][c] = tile
    return True


def _place_lava(
    grid: list[list[str]], rng: random.Random, count: int, protected: set[Coord]
) -> bool:
    size = len(grid)
    low = 2 if size >= 7 else 1
    high = size - 2 if size >= 7 else size - 1
    candidates = [
        (r, c)
        for r in range(low, high)
        for c in range(low, high)
        if (r, c) not in protected and grid[r][c] == "."
    ]
    rng.shuffle(candidates)
    if len(candidates) < count:
        return False
    for r, c in candidates[:count]:
        grid[r][c] = "L"
    return True


def _tile_coords(grid: list[list[str]], tile: str) -> set[Coord]:
    size = len(grid)
    return {(r, c) for r in range(size) for c in range(size) if grid[r][c] == tile}


def _count_tile(grid: list[list[str]], tile: str) -> int:
    return sum(row.count(tile) for row in grid)


def _validate_counts(
    grid: list[list[str]], gold_total: int, diamond_total: int, lava_total: int
) -> bool:
    return (
        _count_tile(grid, "G") == gold_total
        and _count_tile(grid, "D") == diamond_total
        and _count_tile(grid, "L") == lava_total
    )


def _assert_valid_counts(
    grid: list[list[str]], gold_total: int, diamond_total: int, lava_total: int
) -> None:
    if not _validate_counts(grid, gold_total, diamond_total, lava_total):
        raise RuntimeError("Generated maze resource counts do not match metadata.")


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
    route_tags = ", ".join(f"<route_{i}>...</route_{i}>" for i in range(1, routes + 1))
    route_examples = "\n".join(
        [
            "  <route_1>RIGHT RIGHT RIGHT RIGHT DOWN DOWN DOWN DOWN</route_1>",
            "  <route_2>DOWN DOWN DOWN DOWN RIGHT RIGHT RIGHT RIGHT</route_2>",
            "  <route_3>RIGHT DOWN RIGHT DOWN RIGHT DOWN RIGHT DOWN</route_3>",
        ][:routes]
    )
    tile_counts = (
        f"This maze has {gold_total} Gold, {diamond_total} Diamond, "
        f"{lava_total} Lava, and {bonus_total} Bonus tiles."
    )
    maze_intro = (
        f"Navigate a {len(rows)}x{len(rows)} maze from S to E. "
        "Collect gold and diamonds, avoid lava."
    )
    return f"""{maze_intro}

Grid:
{grid_text}

- Move: UP, DOWN, LEFT, RIGHT. # is a wall -- you cannot enter it.
- Do not leave the grid.
- Collect: G (Gold), D (Diamond), B (Bonus) tiles by stepping on them.
- Avoid: L (Lava) tiles. Stepping on lava costs you.
- Visiting a B cell multiplies your other scores -- explore!
- You MUST reach E. If you don't reach E, your score is zero everywhere.
- Items only count if collected BEFORE you reach E (the trajectory ends at E).
- You have {step_budget} steps per route.

{tile_counts}
Reason briefly about the maze, then provide {routes} genuinely different routes from S to E.
Each route is a sequence of UP/DOWN/LEFT/RIGHT moves (space-separated).
Wrap each route in numbered tags ({route_tags}). Inside each tag put ONLY moves.
Do not include arrows, coordinates, or prose inside route tags; any reasoning goes outside the tags.
Each route has its own {step_budget}-step budget and must reach E (score is zero if it doesn't).
Format example (m={routes}):
{route_examples}
"""


def build_dataset(*, start_seed: int, count: int, size: int = 9):
    from datasets import Dataset

    return Dataset.from_list([make_maze(start_seed + i, size=size).to_row() for i in range(count)])
