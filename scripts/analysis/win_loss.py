"""Win/loss breakdown: 2-way vs 3-way on the same dataset.

Given two *_generations.json files produced by eval.eval on the SAME dataset+seed,
categorize each problem as one of:
    - both_correct
    - both_wrong
    - won_by_3way   (2-way wrong, 3-way correct)
    - lost_by_3way  (2-way correct, 3-way wrong)

Emits a JSON summary and a bar chart. Also dumps up to K worked examples per
category so you can inspect the actual qualitative wins/losses in your paper.

Usage:
    python -m scripts.analysis.win_loss \
      --two_way  eval_results/2way_ckpt700/.../gsm8k_*_generations.json \
      --three_way eval_results/3way_alpha0/.../gsm8k_*_generations.json \
      --dataset gsm8k \
      --out eval_results/analysis/win_loss_gsm8k \
      --examples_per_category 5
"""
import argparse
import json
import os
from pathlib import Path

from common.parsing.parse_and_get_acc import (
    parse_code_answers,
    parse_gsm_answers,
    parse_math_answers,
)


PARSERS = {
    "gsm8k": parse_gsm_answers,
    "math": parse_math_answers,
    "humaneval": parse_code_answers,
    "mbpp": parse_code_answers,
}


def parse_problems(path, dataset):
    with open(path) as f:
        data = json.load(f)
    parser = PARSERS[dataset]
    _, _, items, _, steps, _ = parser(json_data=data)
    # Build a stable key per problem so we can align two runs.
    keyed = {}
    for i, item in enumerate(items):
        key = item.get("question") or item.get("text") or f"idx:{i}"
        keyed[key] = {
            "is_correct": item["is_correct"],
            "raw_generation": item["raw_generation"],
            "ground_truth": item.get("ground_truth"),
            "extracted_answer": item.get("extracted_answer"),
            "steps": steps[i] if i < len(steps) else None,
        }
    return keyed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--two_way", required=True)
    p.add_argument("--three_way", required=True)
    p.add_argument("--dataset", required=True, choices=list(PARSERS))
    p.add_argument("--out", required=True)
    p.add_argument("--examples_per_category", type=int, default=5)
    args = p.parse_args()

    two = parse_problems(args.two_way, args.dataset)
    three = parse_problems(args.three_way, args.dataset)
    common_keys = set(two.keys()) & set(three.keys())
    if not common_keys:
        raise SystemExit(
            "No problems aligned across the two files — are they the same dataset/seed?"
        )
    print(f"Aligned {len(common_keys)} problems "
          f"(2way had {len(two)}, 3way had {len(three)})")

    buckets = {
        "both_correct": [],
        "both_wrong": [],
        "won_by_3way": [],
        "lost_by_3way": [],
    }
    for k in common_keys:
        t = two[k]
        h = three[k]
        if t["is_correct"] and h["is_correct"]:
            buckets["both_correct"].append((k, t, h))
        elif (not t["is_correct"]) and (not h["is_correct"]):
            buckets["both_wrong"].append((k, t, h))
        elif (not t["is_correct"]) and h["is_correct"]:
            buckets["won_by_3way"].append((k, t, h))
        else:
            buckets["lost_by_3way"].append((k, t, h))

    total = sum(len(v) for v in buckets.values())
    counts = {k: len(v) for k, v in buckets.items()}
    pct = {k: 100.0 * v / max(total, 1) for k, v in counts.items()}

    os.makedirs(args.out, exist_ok=True)
    summary = {
        "dataset": args.dataset,
        "total": total,
        "counts": counts,
        "pct": pct,
        "acc_two_way": 100.0 * (counts["both_correct"] + counts["lost_by_3way"]) / total,
        "acc_three_way": 100.0 * (counts["both_correct"] + counts["won_by_3way"]) / total,
        "net_gain": 100.0 * (counts["won_by_3way"] - counts["lost_by_3way"]) / total,
    }
    with open(Path(args.out) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Worked examples
    examples = {}
    for name, rows in buckets.items():
        examples[name] = []
        for k, t, h in rows[: args.examples_per_category]:
            examples[name].append({
                "question": k,
                "ground_truth": t["ground_truth"],
                "two_way": {
                    "extracted": t["extracted_answer"],
                    "gen": t["raw_generation"][:1500],
                    "steps": t["steps"],
                },
                "three_way": {
                    "extracted": h["extracted_answer"],
                    "gen": h["raw_generation"][:1500],
                    "steps": h["steps"],
                },
            })
    with open(Path(args.out) / "examples.json", "w") as f:
        json.dump(examples, f, indent=2)

    # Bar chart (lazy import — not everyone has matplotlib installed locally)
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        names = ["both_correct", "won_by_3way", "lost_by_3way", "both_wrong"]
        colors = ["#2a6", "#28d", "#c44", "#555"]
        ax.bar(names, [counts[n] for n in names], color=colors)
        for i, n in enumerate(names):
            ax.text(i, counts[n] + 1, f"{counts[n]}", ha="center")
        ax.set_ylabel("# problems")
        ax.set_title(
            f"{args.dataset.upper()}  2-way {summary['acc_two_way']:.1f}% "
            f"vs 3-way {summary['acc_three_way']:.1f}%  "
            f"(net: {summary['net_gain']:+.1f}pp)"
        )
        fig.tight_layout()
        fig.savefig(Path(args.out) / "win_loss.png", dpi=140)
        print(f"Wrote {Path(args.out) / 'win_loss.png'}")
    except ImportError:
        print("matplotlib not available; skipping plot")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
