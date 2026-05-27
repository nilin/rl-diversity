from __future__ import annotations

import argparse
from dataclasses import replace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from diversity_muon.maze import MazeExample, format_prompt, make_maze
from diversity_muon.maze_reward import parse_routes, score_completion_routes
from diversity_muon.prompting import format_chat_prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--num-seeds", type=int, default=4)
    parser.add_argument("--maze-size", type=int, default=7)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--include-tiny", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, padding_side="left", trust_remote_code=True, fix_mistral_regex=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    model.eval()

    variants = [
        ("paper_multi", lambda maze: maze),
        ("paper_single", to_single_route),
        ("paper_single_no_reason", to_single_route_no_reason),
        ("paper_single_greedy_path", to_single_route_with_greedy_path_hint),
    ]
    if args.include_tiny:
        variants.extend(
            [
                ("tiny_open_single", lambda maze: tiny_open_single()),
                ("tiny_open_multi", lambda maze: tiny_open_multi()),
            ]
        )

    for name, build_variant in variants:
        print(f"\n## variant={name}")
        for seed in range(args.seed_start, args.seed_start + args.num_seeds):
            maze = build_variant(make_maze(seed, size=args.maze_size))
            print(f"seed={seed} budget={maze.step_budget} start={maze.start} end={maze.end}")
            prompt = format_chat_prompt(tokenizer, maze.prompt)
            encoded = tokenizer([prompt], return_tensors="pt").to(model.device)
            for sample_idx in range(args.samples):
                with torch.no_grad():
                    generated = model.generate(
                        **encoded,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_new_tokens=args.max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                completion_ids = generated[0, encoded["input_ids"].shape[1] :]
                completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
                vectors = score_completion_routes(
                    completion,
                    grid=list(maze.grid),
                    start=maze.start,
                    end=maze.end,
                    step_budget=maze.step_budget,
                    gold_total=maze.gold_total,
                    diamond_total=maze.diamond_total,
                    lava_total=maze.lava_total,
                    expected_routes=route_count(name),
                )
                scores = [float(vector.mean()) for vector in vectors]
                if any(vector[0] == 1.0 for vector in vectors):
                    print(f"FOUND variant={name} seed={seed} sample={sample_idx}")
                    print("grid:")
                    print("\n".join(maze.grid))
                    print("completion:")
                    print(completion)
                    print("routes:")
                    print(parse_routes(completion, expected_routes=route_count(name)))
                    print("vectors:")
                    print([vector.tolist() for vector in vectors])
                    print("scalar_scores:")
                    print(scores)
                    return
                print(f"  sample={sample_idx} best={max(scores):.3f}")

    raise SystemExit("No correct route found in searched settings.")


def route_count(name: str) -> int:
    return 3 if name in {"paper_multi", "tiny_open_multi"} else 1


def tiny_open_single() -> MazeExample:
    grid = (
        "S . .",
        ". . .",
        ". . E",
    )
    prompt = """Navigate a 3x3 maze from S to E.

Grid:
S . .
. . .
. . E

- Move: UP, DOWN, LEFT, RIGHT.
- Do not leave the grid.
- You have 4 steps.

Return exactly one route in <route_1>...</route_1> tags.
The route must reach E. Example format:
<route_1>RIGHT RIGHT DOWN DOWN</route_1>
"""
    return MazeExample(
        seed=0,
        size=3,
        grid=grid,
        start=(0, 0),
        end=(2, 2),
        step_budget=4,
        gold_total=1,
        diamond_total=1,
        lava_total=1,
        prompt=prompt,
    )


def tiny_open_multi() -> MazeExample:
    grid = (
        "S . .",
        ". . .",
        ". . E",
    )
    prompt = """Navigate a 3x3 maze from S to E.

Grid:
S . .
. . .
. . E

- Move: UP, DOWN, LEFT, RIGHT.
- Do not leave the grid.
- You have 4 steps per route.

Return 3 different routes in <route_1>, <route_2>, and <route_3> tags.
The routes must reach E. Example format:
<route_1>RIGHT RIGHT DOWN DOWN</route_1>
<route_2>DOWN DOWN RIGHT RIGHT</route_2>
<route_3>RIGHT DOWN RIGHT DOWN</route_3>
"""
    return MazeExample(
        seed=0,
        size=3,
        grid=grid,
        start=(0, 0),
        end=(2, 2),
        step_budget=4,
        gold_total=1,
        diamond_total=1,
        lava_total=1,
        prompt=prompt,
    )


def to_single_route(maze: MazeExample) -> MazeExample:
    prompt = single_route_prompt(maze, include_reasoning=True, include_hint=False)
    return replace(maze, prompt=prompt)


def to_single_route_no_reason(maze: MazeExample) -> MazeExample:
    prompt = single_route_prompt(maze, include_reasoning=False, include_hint=False)
    return replace(maze, prompt=prompt)


def to_single_route_with_greedy_path_hint(maze: MazeExample) -> MazeExample:
    prompt = single_route_prompt(maze, include_reasoning=False, include_hint=True)
    return replace(maze, prompt=prompt)


def single_route_prompt(
    maze: MazeExample, *, include_reasoning: bool, include_hint: bool
) -> str:
    rows = tuple(maze.grid)
    base = format_prompt(
        rows,
        step_budget=maze.step_budget,
        gold_total=maze.gold_total,
        diamond_total=maze.diamond_total,
        lava_total=maze.lava_total,
        bonus_total=1,
        routes=1,
    )
    prompt = base.replace("provide 1 genuinely different routes", "provide 1 route")
    prompt = prompt.replace(
        "Wrap each route in numbered tags (<route_1>...</route_1>).",
        "Wrap the route in <route_1>...</route_1> tags.",
    )
    if not include_reasoning:
        prompt = prompt.replace(
            "Reason briefly about the maze, then provide 1 route from S to E.\n", ""
        )
        prompt += "\nReturn only the route tag and moves.\n"
    if include_hint:
        prompt += "\nA simple route can ignore rewards; first focus on reaching E.\n"
    return prompt


if __name__ == "__main__":
    main()
