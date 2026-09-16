"""Aggregate metrics across seed runs.

Reads every run_seed*/metrics.json under a checkpoints directory and reports
mean and standard deviation for the RAGTruth (and overall) headline metrics.

Run:  python summarize_runs.py /kaggle/working/checkpoints
      python summarize_runs.py checkpoints          # local default
"""

from __future__ import annotations
import glob
import json
import os
import sys
import statistics


def load_runs(root: str) -> list[dict]:
    """Find every metrics.json under root/run_seed*/ and load it."""
    pattern = os.path.join(root, "run_seed*", "metrics.json")
    paths = sorted(glob.glob(pattern))
    runs = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        data["_path"] = p
        runs.append(data)
    return runs


def collect(runs: list[dict], source: str, metric: str) -> list[float]:
    """Pull one metric for one source across all runs, skipping any that lack it."""
    vals = []
    for r in runs:
        per_source = r.get("per_source", {})
        if source in per_source and metric in per_source[source]:
            vals.append(per_source[source][metric])
    return vals


def summarize(vals: list[float]) -> str:
    if not vals:
        return "no data"
    if len(vals) == 1:
        return f"{vals[0]:.4f} (1 run)"
    mean = statistics.mean(vals)
    std = statistics.stdev(vals)
    return f"{mean:.4f} ± {std:.4f}  (n={len(vals)}, min={min(vals):.4f}, max={max(vals):.4f})"


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "checkpoints"
    runs = load_runs(root)

    if not runs:
        print(f"No run_seed*/metrics.json found under {root!r}")
        return

    print(f"Found {len(runs)} run(s) under {root!r}:")
    for r in runs:
        seed = r.get("seed", "?")
        sha = r.get("git_sha", "?")
        thr = r.get("threshold", "?")
        print(f"  seed {seed}  sha {sha}  threshold {thr}  ({r['_path']})")
    print()

    # the headline: RAGTruth is the real target, report it first and fullest
    print("RAGTruth (the target metric):")
    for metric in ["pr_auc", "roc_auc", "f1", "precision", "recall"]:
        vals = collect(runs, "ragtruth", metric)
        print(f"  {metric:10s} {summarize(vals)}")
    print()

    print("HaluEval (context only, expected near-ceiling):")
    for metric in ["pr_auc", "roc_auc", "f1"]:
        vals = collect(runs, "halueval", metric)
        print(f"  {metric:10s} {summarize(vals)}")


if __name__ == "__main__":
    main()