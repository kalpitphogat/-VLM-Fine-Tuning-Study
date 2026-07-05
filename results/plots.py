"""Generates results/accuracy_vs_rank.png and results/filtered_vs_unfiltered.png
from results/metrics.csv.

Usage:
    python results/plots.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "scripts"))
from common import load_config  # noqa: E402

# Palette from the repo's dataviz skill reference (references/palette.md)
BLUE = "#2a78d6"
AQUA = "#1baf7a"
MUTED = "#898781"
GRID = "#e1e0d9"
INK = "#0b0b0b"


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def plot_accuracy_vs_rank(df, baseline_acc, out_path):
    subset = df[(df["lr_schedule"] == "cosine") & (df["data_variant"] == "filtered")]
    subset = subset.sort_values("rank")
    if subset.empty:
        print("No rank-sweep rows (schedule=cosine, data=filtered) found, skipping accuracy_vs_rank.png")
        return

    fig, ax = plt.subplots(figsize=(6, 4), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.plot(subset["rank"], subset["accuracy"], color=BLUE, linewidth=2, marker="o", markersize=8, label="LoRA fine-tuned")
    ax.axhline(baseline_acc, color=MUTED, linewidth=2, linestyle="--", label="zero-shot baseline")

    for x, y in zip(subset["rank"], subset["accuracy"]):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", color=INK, fontsize=9)

    ax.set_xticks(subset["rank"])
    ax.set_xlabel("LoRA rank", color=INK)
    ax.set_ylabel("Test accuracy", color=INK)
    ax.set_title("Accuracy vs. LoRA rank (cosine schedule, filtered data)", color=INK)
    style_axes(ax)
    ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_filtered_vs_unfiltered(df, baseline_acc, out_path):
    subset = df[(df["lr_schedule"] == "cosine") & (df["data_variant"].isin(["filtered", "unfiltered"]))]
    if subset.empty:
        print("No filtered/unfiltered rows found, skipping filtered_vs_unfiltered.png")
        return

    # Compare at whichever single rank has both variants (the ablation's data-quality step).
    ranks_with_both = subset.groupby("rank")["data_variant"].nunique()
    candidate_ranks = ranks_with_both[ranks_with_both == 2].index
    if len(candidate_ranks) == 0:
        print("No single rank has both filtered and unfiltered runs, skipping filtered_vs_unfiltered.png")
        return
    rank = candidate_ranks[0]
    subset = subset[subset["rank"] == rank].set_index("data_variant")

    labels = ["filtered", "unfiltered"]
    values = [subset.loc[v, "accuracy"] if v in subset.index else 0 for v in labels]
    colors = [BLUE, AQUA]

    fig, ax = plt.subplots(figsize=(5, 4), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    bars = ax.bar(labels, values, color=colors, width=0.5)
    ax.axhline(baseline_acc, color=MUTED, linewidth=2, linestyle="--", label="zero-shot baseline")

    for bar, v in zip(bars, values):
        ax.annotate(f"{v:.3f}", (bar.get_x() + bar.get_width() / 2, v), textcoords="offset points",
                    xytext=(0, 6), ha="center", color=INK, fontsize=9)

    ax.set_ylabel("Test accuracy", color=INK)
    ax.set_title(f"Data filtering effect (rank {rank}, cosine schedule)", color=INK)
    style_axes(ax)
    ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    repo_root = Path(__file__).resolve().parent.parent
    config = load_config(str(repo_root / "config.yaml"))
    metrics_csv = repo_root / config["paths"]["metrics_csv"]
    results_dir = repo_root / config["paths"]["results_dir"]

    df = pd.read_csv(metrics_csv)
    baseline_rows = df[df["config_name"] == "zero_shot_baseline"]
    if baseline_rows.empty:
        raise RuntimeError(f"No zero_shot_baseline row in {metrics_csv} - run scripts/baseline.py first.")
    baseline_acc = float(baseline_rows.iloc[-1]["accuracy"])

    plot_accuracy_vs_rank(df, baseline_acc, results_dir / "accuracy_vs_rank.png")
    plot_filtered_vs_unfiltered(df, baseline_acc, results_dir / "filtered_vs_unfiltered.png")


if __name__ == "__main__":
    main()
