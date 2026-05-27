from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RUN_LABELS = {
    "adam_multirlvr": "AdamW Multi-RLVR",
    "muon_multirlvr": "Muon Multi-RLVR",
    "soft_muon_multirlvr": "Soft-Muon Multi-RLVR",
    "soft_muon_p05_multirlvr": "Soft-Muon p=0.5",
    "adam_vpo": "AdamW VPO",
}

RUN_COLORS = {
    "adam_multirlvr": "#4C78A8",
    "muon_multirlvr": "#F58518",
    "soft_muon_multirlvr": "#54A24B",
    "soft_muon_p05_multirlvr": "#B279A2",
    "adam_vpo": "#E45756",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot eval JSONs from diversity runs.")
    parser.add_argument(
        "--input-glob",
        default="outputs/*_eval/*.json",
        help="Glob for eval JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/plots",
        help="Directory for plots and summary CSVs.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df, prompt_df = load_eval_outputs(args.input_glob)
    if summary_df.empty:
        raise SystemExit(f"No eval JSONs matched {args.input_glob!r}")

    summary_df.to_csv(output_dir / "eval_summary.csv", index=False)
    prompt_df.to_csv(output_dir / "prompt_metrics.csv", index=False)

    plot_best_at_k(summary_df, output_dir / "best_at_k_by_eval.png")
    plot_best10_vs_baseline(summary_df, output_dir / "best10_vs_best_available.png")
    plot_mean_diversity(summary_df, output_dir / "mean_diversity_by_eval.png")
    plot_prompt_diversity(prompt_df, output_dir / "prompt_diversity_boxplot.png")
    plot_best10_vs_diversity(summary_df, output_dir / "best10_vs_diversity.png")
    plot_success_rate(summary_df, output_dir / "nonzero_rates_by_eval.png")

    print(f"Wrote plots and CSVs to {output_dir}")
    if "mean_best_at_1" not in summary_df.columns:
        print("Note: best@1 is not present in these JSONs; best@3 was used as the baseline plot.")


def load_eval_outputs(input_glob: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    prompt_rows: list[dict[str, object]] = []

    for path in sorted(Path().glob(input_glob)):
        data = json.loads(path.read_text(encoding="utf-8"))
        eval_set = path.parent.name
        run = path.stem
        prompts = data.get("prompts", [])

        row: dict[str, object] = {
            "eval_set": eval_set,
            "run": run,
            "run_label": RUN_LABELS.get(run, run),
            "path": str(path),
            "model": data.get("model"),
            "num_prompts": data.get("num_prompts"),
            "samples_per_prompt": data.get("samples_per_prompt"),
        }
        for key, value in data.items():
            if key.startswith("mean_"):
                row[key] = value

        if prompts:
            prompt_df = pd.DataFrame(prompts)
            row["median_diversity"] = float(prompt_df["diversity"].median())
            row["nonzero_diversity_rate"] = float((prompt_df["diversity"] > 0).mean())
            if "best_at_10" in prompt_df:
                row["nonzero_best10_rate"] = float((prompt_df["best_at_10"] > 0).mean())
            for prompt in prompts:
                prompt_rows.append(
                    {
                        "eval_set": eval_set,
                        "run": run,
                        "run_label": RUN_LABELS.get(run, run),
                        **prompt,
                    }
                )
        summary_rows.append(row)

    return pd.DataFrame(summary_rows), pd.DataFrame(prompt_rows)


def plot_best_at_k(summary_df: pd.DataFrame, output_path: Path) -> None:
    best_columns = sorted(
        [
            column
            for column in summary_df.columns
            if column.startswith("mean_best_at_") and summary_df[column].notna().any()
        ],
        key=lambda name: int(name.rsplit("_", 1)[-1]),
    )
    eval_sets = sorted(summary_df["eval_set"].unique())
    fig, axes = plt.subplots(
        1,
        len(eval_sets),
        figsize=(5.0 * len(eval_sets), 4.2),
        sharey=True,
        squeeze=False,
    )

    for axis, eval_set in zip(axes[0], eval_sets, strict=True):
        subset = summary_df[summary_df["eval_set"] == eval_set]
        for _, row in subset.sort_values("run").iterrows():
            ks = [int(column.rsplit("_", 1)[-1]) for column in best_columns]
            values = [row[column] for column in best_columns]
            axis.plot(
                ks,
                values,
                marker="o",
                linewidth=2.0,
                label=row["run_label"],
                color=RUN_COLORS.get(str(row["run"])),
            )
        axis.set_title(eval_set)
        axis.set_xlabel("k")
        axis.set_xticks(ks)
        axis.grid(True, axis="y", alpha=0.25)

    axes[0][0].set_ylabel("mean best@k")
    handles, labels = axes[0][-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=min(5, len(labels)))
    fig.suptitle("Validation Best@k")
    fig.tight_layout(rect=(0, 0.16, 1, 0.92))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_best10_vs_baseline(summary_df: pd.DataFrame, output_path: Path) -> None:
    baseline_column = "mean_best_at_1" if "mean_best_at_1" in summary_df.columns else "mean_best_at_3"
    baseline_label = "best@1" if baseline_column.endswith("_1") else "best@3"

    rows = summary_df.dropna(subset=[baseline_column, "mean_best_at_10"]).copy()
    rows["delta"] = rows["mean_best_at_10"] - rows[baseline_column]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.2))
    grouped_bar(
        axes[0],
        rows,
        value_columns=[baseline_column, "mean_best_at_10"],
        labels=[baseline_label, "best@10"],
    )
    axes[0].set_ylabel("mean score")
    axes[0].set_title(f"{baseline_label} vs best@10")

    bar_by_eval_run(axes[1], rows, "delta")
    axes[1].set_ylabel(f"best@10 - {baseline_label}")
    axes[1].set_title("Best@10 Lift")

    add_row_key(fig, rows)
    fig.tight_layout(rect=(0, 0.22, 1, 1))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_mean_diversity(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.2))
    bar_by_eval_run(axes[0], summary_df, "mean_diversity")
    axes[0].set_ylabel("mean pairwise L1 diversity")
    axes[0].set_title("Mean Reward-Space Diversity")

    if "nonzero_diversity_rate" in summary_df:
        bar_by_eval_run(axes[1], summary_df, "nonzero_diversity_rate")
        axes[1].set_ylabel("fraction of prompts")
        axes[1].set_title("Prompts With Nonzero Diversity")
    else:
        axes[1].axis("off")

    add_row_key(fig, summary_df.sort_values(["eval_set", "run"]))
    fig.tight_layout(rect=(0, 0.22, 1, 1))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_prompt_diversity(prompt_df: pd.DataFrame, output_path: Path) -> None:
    if prompt_df.empty:
        return

    eval_sets = sorted(prompt_df["eval_set"].unique())
    fig, axes = plt.subplots(
        1,
        len(eval_sets),
        figsize=(5.2 * len(eval_sets), 7.2),
        sharey=True,
        squeeze=False,
    )

    for axis, eval_set in zip(axes[0], eval_sets, strict=True):
        subset = prompt_df[prompt_df["eval_set"] == eval_set]
        runs = sorted(subset["run"].unique())
        values = [subset[subset["run"] == run]["diversity"].to_numpy() for run in runs]
        box = axis.boxplot(values, patch_artist=True, showfliers=True)
        for patch, run in zip(box["boxes"], runs, strict=True):
            patch.set_facecolor(RUN_COLORS.get(run, "#999999"))
            patch.set_alpha(0.55)
        axis.set_xticks(range(1, len(runs) + 1), [str(index) for index in range(1, len(runs) + 1)])
        axis.set_xlabel("run id")
        axis.set_title(eval_set)
        axis.grid(True, axis="y", alpha=0.25)

    axes[0][0].set_ylabel("per-prompt pairwise L1 diversity")
    fig.suptitle("Prompt-Level Diversity Distribution")
    add_prompt_key(fig, prompt_df)
    fig.tight_layout(rect=(0, 0.22, 1, 0.92))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_best10_vs_diversity(summary_df: pd.DataFrame, output_path: Path) -> None:
    rows = summary_df.dropna(subset=["mean_diversity", "mean_best_at_10"])
    fig, axis = plt.subplots(figsize=(7.6, 5.6))
    for _, row in rows.iterrows():
        run = str(row["run"])
        axis.scatter(
            row["mean_diversity"],
            row["mean_best_at_10"],
            s=90,
            color=RUN_COLORS.get(run, "#666666"),
            edgecolor="black",
            linewidth=0.6,
        )
        axis.annotate(
            f"{row['eval_set']} / {row['run_label']}",
            (row["mean_diversity"], row["mean_best_at_10"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("mean diversity")
    axis.set_ylabel("mean best@10")
    axis.set_title("Best@10 vs Reward-Space Diversity")
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_success_rate(summary_df: pd.DataFrame, output_path: Path) -> None:
    columns = [
        column
        for column in ("nonzero_best10_rate", "nonzero_diversity_rate")
        if column in summary_df.columns
    ]
    if not columns:
        return

    fig, axes = plt.subplots(1, len(columns), figsize=(6.8 * len(columns), 7.2), squeeze=False)
    titles = {
        "nonzero_best10_rate": "Prompts With best@10 > 0",
        "nonzero_diversity_rate": "Prompts With Diversity > 0",
    }
    for axis, column in zip(axes[0], columns, strict=True):
        bar_by_eval_run(axis, summary_df, column)
        axis.set_ylabel("fraction of prompts")
        axis.set_ylim(0, 1)
        axis.set_title(titles[column])
    add_row_key(fig, summary_df.sort_values(["eval_set", "run"]))
    fig.tight_layout(rect=(0, 0.22, 1, 1))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def grouped_bar(
    axis: plt.Axes,
    rows: pd.DataFrame,
    *,
    value_columns: list[str],
    labels: list[str],
) -> None:
    width = 0.8 / len(value_columns)
    x_positions = list(range(len(rows)))
    for index, (column, label) in enumerate(zip(value_columns, labels, strict=True)):
        offsets = [x - 0.4 + width / 2 + index * width for x in x_positions]
        axis.bar(offsets, rows[column].to_numpy(), width=width, label=label)
    axis.set_xticks(x_positions, [str(index) for index in range(1, len(rows) + 1)])
    axis.set_xlabel("run id")
    axis.legend()
    axis.grid(True, axis="y", alpha=0.25)


def bar_by_eval_run(axis: plt.Axes, summary_df: pd.DataFrame, value_column: str) -> None:
    rows = summary_df.dropna(subset=[value_column]).sort_values(["eval_set", "run"])
    colors = [RUN_COLORS.get(str(run), "#666666") for run in rows["run"]]
    axis.bar(range(len(rows)), rows[value_column], color=colors)
    axis.set_xticks(range(len(rows)), [str(index) for index in range(1, len(rows) + 1)])
    axis.set_xlabel("run id")
    axis.grid(True, axis="y", alpha=0.25)


def add_row_key(fig: plt.Figure, rows: pd.DataFrame) -> None:
    labels = [
        f"{index}. {row.eval_set}: {row.run_label}"
        for index, row in enumerate(rows.itertuples(), start=1)
    ]
    fig.text(
        0.02,
        0.02,
        "\n".join(wrap_key_labels(labels)),
        ha="left",
        va="bottom",
        fontsize=8,
        family="monospace",
    )


def add_prompt_key(fig: plt.Figure, prompt_df: pd.DataFrame) -> None:
    labels: list[str] = []
    for eval_set in sorted(prompt_df["eval_set"].unique()):
        subset = prompt_df[prompt_df["eval_set"] == eval_set]
        runs = sorted(subset["run"].unique())
        for index, run in enumerate(runs, start=1):
            labels.append(f"{eval_set} {index}. {RUN_LABELS.get(run, run)}")
    fig.text(
        0.02,
        0.02,
        "\n".join(wrap_key_labels(labels)),
        ha="left",
        va="bottom",
        fontsize=8,
        family="monospace",
    )


def wrap_key_labels(labels: list[str], *, columns: int = 2) -> list[str]:
    if len(labels) <= 6:
        return labels
    midpoint = (len(labels) + columns - 1) // columns
    left = labels[:midpoint]
    right = labels[midpoint:]
    width = max(len(label) for label in left) + 4
    lines: list[str] = []
    for index in range(midpoint):
        left_label = left[index] if index < len(left) else ""
        right_label = right[index] if index < len(right) else ""
        lines.append(f"{left_label:<{width}}{right_label}")
    return lines


if __name__ == "__main__":
    main()
