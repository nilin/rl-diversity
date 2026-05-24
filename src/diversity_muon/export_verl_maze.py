from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset

from diversity_muon.maze import make_maze


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/maze_verl")
    parser.add_argument("--train-size", type=int, default=128)
    parser.add_argument("--val-size", type=int, default=32)
    parser.add_argument("--train-seed-start", type=int, default=42)
    parser.add_argument("--val-seed-start", type=int, default=4242)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_split(
        output_dir / "train.parquet",
        split="train",
        seed_start=args.train_seed_start,
        count=args.train_size,
    )
    write_split(
        output_dir / "test.parquet",
        split="test",
        seed_start=args.val_seed_start,
        count=args.val_size,
    )
    print(f"Wrote veRL Maze data to {output_dir}")


def write_split(path: Path, *, split: str, seed_start: int, count: int) -> None:
    rows = [to_verl_row(make_maze(seed_start + idx), split=split, idx=idx) for idx in range(count)]
    Dataset.from_list(rows).to_parquet(str(path))


def to_verl_row(example, *, split: str, idx: int) -> dict[str, object]:
    ground_truth = {
        "grid": list(example.grid),
        "start": list(example.start),
        "end": list(example.end),
        "step_budget": example.step_budget,
        "gold_total": example.gold_total,
        "diamond_total": example.diamond_total,
        "lava_total": example.lava_total,
    }
    return {
        "data_source": "maze",
        "prompt": [{"role": "user", "content": example.prompt}],
        "ability": "maze",
        "reward_model": {"style": "rule", "ground_truth": json.dumps(ground_truth)},
        "extra_info": {
            "split": split,
            "index": idx,
            "seed": example.seed,
            "ground_truth": json.dumps(ground_truth),
        },
    }


if __name__ == "__main__":
    main()
