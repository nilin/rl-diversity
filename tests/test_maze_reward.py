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
