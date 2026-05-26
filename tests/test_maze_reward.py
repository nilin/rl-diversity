from diversity_muon.maze import make_maze
from diversity_muon.maze_reward import parse_routes, score_completion_routes
from diversity_muon.objectives import MazeReward, RewardConfig


def test_parse_routes_numbered_tags():
    completion = """
    <route_1>RIGHT DOWN LEFT</route_1>
    <route_2>UP UP BADTOKEN</route_2>
    <route_3>DOWN</route_3>
    """
    assert parse_routes(completion) == [
        ["RIGHT", "DOWN", "LEFT"],
        ["UP", "UP"],
        ["DOWN"],
    ]


def test_score_completion_routes_returns_three_vectors():
    maze = make_maze(42)
    vectors = score_completion_routes(
        "<route_1>RIGHT</route_1><route_2>DOWN</route_2><route_3>LEFT</route_3>",
        grid=list(maze.grid),
        start=maze.start,
        end=maze.end,
        step_budget=maze.step_budget,
        gold_total=maze.gold_total,
        diamond_total=maze.diamond_total,
        lava_total=maze.lava_total,
    )
    assert len(vectors) == 3
    assert all(vector.shape == (4,) for vector in vectors)


def test_score_route_stops_at_exit_before_extra_moves():
    grid = [
        "S E .",
        ". . .",
        ". . .",
    ]
    vectors = score_completion_routes(
        "<route_1>RIGHT LEFT</route_1><route_2>RIGHT</route_2><route_3>RIGHT</route_3>",
        grid=grid,
        start=(0, 0),
        end=(0, 1),
        step_budget=2,
        gold_total=1,
        diamond_total=1,
        lava_total=1,
    )
    assert vectors[0].tolist() == [1.0, 0.0, 0.0, 1.0]


def test_score_route_rejects_walls():
    grid = [
        "S # E",
        ". . .",
        ". . .",
    ]
    vectors = score_completion_routes(
        "<route_1>RIGHT RIGHT</route_1><route_2>DOWN</route_2><route_3>DOWN</route_3>",
        grid=grid,
        start=(0, 0),
        end=(0, 2),
        step_budget=2,
        gold_total=1,
        diamond_total=1,
        lava_total=1,
    )
    assert vectors[0].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_paper_style_maze_prompt_and_counts():
    maze = make_maze(42)
    assert "Format example (m=3):" in maze.prompt
    assert "# is a wall -- you cannot enter it" in maze.prompt
    assert sum(row.split().count("G") for row in maze.grid) == maze.gold_total
    assert sum(row.split().count("D") for row in maze.grid) == maze.diamond_total
    assert sum(row.split().count("L") for row in maze.grid) == maze.lava_total


def test_vpo_reward_is_scalar():
    reward = MazeReward(RewardConfig(objective="vpo", seed=123))
    values = reward(
        completions=["<route_1>RIGHT</route_1><route_2>DOWN</route_2><route_3>LEFT</route_3>"],
        grid=[list(make_maze(42).grid)],
        start=[list(make_maze(42).start)],
        end=[list(make_maze(42).end)],
        step_budget=[make_maze(42).step_budget],
        gold_total=[make_maze(42).gold_total],
        diamond_total=[make_maze(42).diamond_total],
        lava_total=[make_maze(42).lava_total],
    )
    assert len(values) == 1
    assert isinstance(values[0], float)
