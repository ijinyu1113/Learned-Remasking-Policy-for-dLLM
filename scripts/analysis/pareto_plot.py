"""NFE-vs-accuracy Pareto plot across alpha sweep and baselines.

Consumes the `detailed_results.csv` emitted by eval.aggregate_results and
produces a scatter of (avg_steps, accuracy), one point per (run, checkpoint, dataset).

Usage:
    python -m scripts.analysis.pareto_plot \
        --results eval_results/detailed_results.csv \
        --dataset gsm8k \
        --out eval_results/analysis/pareto_gsm8k.png
"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="detailed_results.csv")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--runs", nargs="*", default=None, help="Optional filter on run names")
    args = p.parse_args()

    df = pd.read_csv(args.results)
    df = df[df["dataset"] == args.dataset].copy()
    if args.runs:
        df = df[df["run"].isin(args.runs)]
    if df.empty:
        raise SystemExit(f"No rows for dataset={args.dataset}")

    # Average across seeds per (run, checkpoint, temperature)
    grp = df.groupby(["run", "checkpoint", "temperature"], as_index=False).agg(
        accuracy=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        avg_steps=("avg_steps", "mean"),
        avg_steps_std=("avg_steps", "std"),
    )
    grp.to_csv(Path(args.out).with_suffix(".csv"), index=False)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for run, sub in grp.groupby("run"):
        ax.errorbar(
            sub["avg_steps"], sub["accuracy"],
            xerr=sub["avg_steps_std"].fillna(0),
            yerr=sub["accuracy_std"].fillna(0),
            marker="o", linestyle="", label=run, capsize=2, markersize=5,
        )
    ax.set_xlabel("Number of Forward Evaluations (NFE)")
    ax.set_ylabel(f"Accuracy on {args.dataset.upper()} (%)")
    ax.set_title(f"Accuracy vs NFE — {args.dataset.upper()}")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
