from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot in-training diversity_eval.jsonl files.")
    parser.add_argument(
        "--input-glob",
        default="outputs/*/diversity_eval.jsonl",
        help="Glob for in-training diversity JSONL files.",
    )
    parser.add_argument("--output", default="outputs/plots/training_diversity.png")
    args = parser.parse_args()

    rows = load_rows(args.input_glob)
    if rows.empty:
        raise SystemExit(f"No JSONL files matched {args.input_glob!r}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_path.with_suffix(".csv"), index=False)
    plot(rows, output_path)
    print(f"Wrote {output_path} and {output_path.with_suffix('.csv')}")


def load_rows(input_glob: str) -> pd.DataFrame:
    loaded: list[dict[str, object]] = []
    for path in sorted(Path().glob(input_glob)):
        run = path.parent.name
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            loaded.append(
                {
                    "run": run,
                    "step": row["step"],
                    "mean_chain_diversity": row["train_eval/mean_chain_diversity"],
                    "mean_pooled_diversity": row["train_eval/mean_pooled_diversity"],
                    "mean_best_at_3": row["train_eval/mean_best_at_3"],
                    "mean_best_at_6": row["train_eval/mean_best_at_6"],
                    "mean_best_at_12": row["train_eval/mean_best_at_12"],
                }
            )
    return pd.DataFrame(loaded)


def plot(rows: pd.DataFrame, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.2, 5.2))
    for run, subset in rows.sort_values("step").groupby("run"):
        axis.plot(
            subset["step"],
            subset["mean_chain_diversity"],
            marker="o",
            linewidth=2.0,
            label=run,
        )
    axis.set_xlabel("training step")
    axis.set_ylabel("mean pairwise L1 diversity")
    axis.set_title("Reward-Space Diversity Over Training")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
