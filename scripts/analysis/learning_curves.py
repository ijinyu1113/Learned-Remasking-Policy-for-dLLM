"""Per-checkpoint accuracy curves from aggregated eval CSV.

Useful for: "at which checkpoint does 3-way start to beat 2-way?" and for
picking the best-checkpoint on a validation subset before full eval.

Usage:
    python -m scripts.analysis.learning_curves \
      --results eval_results/detailed_results.csv \
      --dataset gsm8k \
      --out eval_results/analysis/learning_curves_gsm8k.png
"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--runs", nargs="*", default=None)
    args = p.parse_args()

    df = pd.read_csv(args.results)
    df = df[df["dataset"] == args.dataset].copy()
    if args.runs:
        df = df[df["run"].isin(args.runs)]
    # Only numeric checkpoints (drop baseline- rows)
    df = df[pd.to_numeric(df["checkpoint"], errors="coerce").notna()]
    df["checkpoint"] = df["checkpoint"].astype(int)
    if df.empty:
        raise SystemExit("No numeric checkpoints to plot.")

    grp = df.groupby(["run", "checkpoint"], as_index=False).agg(
        accuracy=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        avg_steps=("avg_steps", "mean"),
    )

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for run, sub in grp.groupby("run"):
        sub = sub.sort_values("checkpoint")
        ax1.errorbar(
            sub["checkpoint"], sub["accuracy"],
            yerr=sub["accuracy_std"].fillna(0),
            marker="o", label=run, capsize=2, markersize=4,
        )
        ax2.plot(sub["checkpoint"], sub["avg_steps"], marker="o", label=run, markersize=4)
    ax1.set_xlabel("Checkpoint step")
    ax1.set_ylabel(f"Accuracy on {args.dataset.upper()} (%)")
    ax1.set_title("Accuracy vs training step")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)
    ax2.set_xlabel("Checkpoint step")
    ax2.set_ylabel("Avg NFE")
    ax2.set_title("NFE vs training step")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
