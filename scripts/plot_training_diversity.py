from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RUN_LABELS = {
    "qwen17b_7x7_adam_multirlvr": "AdamW Multi-RLVR",
    "qwen17b_7x7_adam_vpo": "AdamW VPO",
    "qwen17b_7x7_muon_multirlvr": "Muon Multi-RLVR",
    "qwen17b_7x7_soft_muon_p04_fixed_coeffs_multirlvr": "Soft-Muon p=0.4 fixed",
    "qwen17b_7x7_soft_muon_p05_fixed_coeffs_multirlvr": "Soft-Muon p=0.5 fixed",
}

RUN_COLORS = {
    "qwen17b_7x7_adam_multirlvr": "#4C78A8",
    "qwen17b_7x7_adam_vpo": "#E45756",
    "qwen17b_7x7_muon_multirlvr": "#F58518",
    "qwen17b_7x7_soft_muon_p04_fixed_coeffs_multirlvr": "#54A24B",
    "qwen17b_7x7_soft_muon_p05_fixed_coeffs_multirlvr": "#B279A2",
}


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
        run_base = re.sub(r"_seed\d+$", "", run)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            high_best_key = (
                "train_eval/mean_best_at_9"
                if "train_eval/mean_best_at_9" in row
                else "train_eval/mean_best_at_12"
            )
            loaded.append(
                {
                    "run": run,
                    "run_base": run_base,
                    "step": row["step"],
                    "mean_chain_diversity": row["train_eval/mean_chain_diversity"],
                    "mean_pooled_diversity": row["train_eval/mean_pooled_diversity"],
                    "mean_best_at_3": row["train_eval/mean_best_at_3"],
                    "mean_best_at_6": row["train_eval/mean_best_at_6"],
                    "mean_best_at_high": row[high_best_key],
                    "high_best_k": int(high_best_key.rsplit("_", 1)[-1]),
                }
            )
    return pd.DataFrame(loaded)


def plot(rows: pd.DataFrame, output_path: Path) -> None:
    panels = [
        ("mean_chain_diversity", "Mean chain diversity", "pairwise L1"),
        ("mean_pooled_diversity", "Mean pooled diversity", "pairwise L1"),
        ("mean_best_at_3", "Mean best@3", "scalar reward"),
        ("mean_best_at_high", "Mean high-k best", "scalar reward"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.8), sharex=True)
    handles = []
    labels = []
    for axis, (column, title, ylabel) in zip(axes.flat, panels, strict=True):
        title_text = title
        for run, subset in rows.sort_values(["run", "step"]).groupby("run"):
            run_base = subset["run_base"].iloc[0]
            label = RUN_LABELS.get(run_base, run)
            (line,) = axis.plot(
                subset["step"],
                subset[column],
                marker="o",
                linewidth=2.0,
                label=label,
                color=RUN_COLORS.get(run_base),
            )
            if column == panels[0][0]:
                handles.append(line)
                labels.append(label)
        if column == "mean_best_at_high" and "high_best_k" in rows:
            high_ks = sorted(rows["high_best_k"].dropna().unique())
            if len(high_ks) == 1:
                title_text = f"Mean best@{int(high_ks[0])}"
        axis.set_title(title_text)
        axis.set_ylabel(ylabel)
        axis.grid(True, axis="y", alpha=0.25)

    for axis in axes[-1]:
        axis.set_xlabel("training step")

    fig.suptitle("In-Training Diversity Eval")
    fig.legend(handles, labels, loc="lower center", ncols=2)
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
