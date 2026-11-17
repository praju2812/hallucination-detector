"""Aggregate test metrics across seed runs, per source (Phase 2).

Reads every run_seed*/ folder under a root (metrics.json, config_snapshot.json, val_history.json,
predictions.jsonl), keeps the valid runs of one experiment, and reports per source:
prevalence, PR-AUC, F1 at each run's validation-tuned threshold, AUROC, as mean +/- sample std [min, max]
(decision D5). The blended all-source number in metrics.json is never reported.
With a truncation flags file it also reports RAGTruth test rows split into context-truncated vs not.

Run:  python -m scripts.summarize_runs runs
      python -m scripts.summarize_runs runs --out results/phase2_G.md
      python -m scripts.summarize_runs /kaggle/working/runs --trunc-flags results/truncation_flags_v2.jsonl

Spec: tests/test_summarize_runs.py
"""

from __future__ import annotations
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from src.metrics import ranking_metrics, thresholded_metrics  # (for truncation_split_metrics)
from utils.io import read_jsonl

EXPECTED_MAX_LENGTH = 1024
EXPECTED_DATA = {"G": "data/unified_v2/"}     # data dir every path of a valid run must sit in
METRICS = ("pr_auc", "f1", "roc_auc")
TRUNC_SOURCES = ("ragtruth_truncated", "ragtruth_untruncated")


# ---- loading and filtering (plumbing) ----------------------------------------

def _load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_runs(root: str) -> list[dict]:
    """Every root/run_seed*/ folder that has a metrics.json (a crashed run has none and is skipped).
    Each run: {name, dir, metrics, snapshot, val_history, predictions_path}; missing files -> None."""
    runs = []
    for d in sorted(glob.glob(os.path.join(root, "run_seed*"))):
        metrics = _load_json(os.path.join(d, "metrics.json"))
        if metrics is None:
            continue
        pred = os.path.join(d, "predictions.jsonl")
        runs.append({
            "name": os.path.basename(d),
            "dir": d,
            "metrics": metrics,
            "snapshot": _load_json(os.path.join(d, "config_snapshot.json")),
            "val_history": _load_json(os.path.join(d, "val_history.json")),
            "predictions_path": pred if os.path.exists(pred) else None,
        })
    return runs


def run_problems(run: dict, experiment: str) -> list[str]:
    """Reasons a run must not be reported for `experiment` (empty list = valid)."""
    snap = run["snapshot"]
    if snap is None:
        return ["no config_snapshot.json"]
    cfg = snap.get("config", {})
    problems = []
    if snap.get("experiment") != experiment:
        problems.append(f"experiment={snap.get('experiment')!r}")
    ml = cfg.get("model", {}).get("max_length")
    if ml != EXPECTED_MAX_LENGTH:
        problems.append(f"max_length={ml}")
    paths = [cfg.get("data", {}).get(k, "") for k in ("train_path", "val_path", "test_path")]
    if any("tiny" in str(p) for p in paths):
        problems.append("tiny smoke-test data")
    want = EXPECTED_DATA.get(experiment)
    if want and not all(str(p).startswith(want) for p in paths):
        problems.append(f"data not under {want}")
    if run["val_history"] is None:
        problems.append("no val_history.json")
    return problems


def filter_runs(runs: list[dict], experiment: str) -> tuple[list[dict], list[tuple[str, list[str]]]]:
    kept, dropped = [], []
    for r in runs:
        p = run_problems(r, experiment)
        if p:
            dropped.append((r["name"], p))
        else:
            kept.append(r)
    return kept, dropped

def summarize_source(runs: list[dict], source: str) -> dict:
    """Aggregate one source's test metrics across seed runs.

    Input:
        runs: run dicts; each has run["metrics"]["per_source"][source] with keys
              pr_auc, f1, roc_auc, n, n_pos (as written by src/train.py -> metrics.json).
        source: e.g. "ragtruth".
    Output:
        {"source": source, "n_runs": int, "n": int, "n_pos": int, "prevalence": float,
         "pr_auc":  {"mean": float, "std": float | None, "min": float, "max": float},
         "f1":      {... same four keys ...},
         "roc_auc": {... same four keys ...}}
        std = SAMPLE standard deviation (n-1), decision D5. With one run std is None (undefined).
    Invariants:
        every run is evaluated on the same test file, so n and n_pos are identical across runs.
    """
     # check for an empty run list
    if runs is None or len(runs) == 0:
        raise ValueError("no runs to summarize")
    
    # for each run, get its per-source block for `source`; a run without it is an error,
    # not something to skip (skipping would silently shrink n_runs).
    for r in runs:
        if "per_source" not in r["metrics"]:
            raise ValueError(f"{r['name']}: no per_source block in metrics.json")
        if source not in r["metrics"]["per_source"]:
            raise ValueError(f"{r['name']}: no {source!r} in metrics.json per_source block")
        if "n" not in r["metrics"]["per_source"][source] or "n_pos" not in r["metrics"]["per_source"][source]:
            raise ValueError(f"{r['name']}: {source!r} block missing n or n_pos")
        
        # check that all expected metrics are present in the per_source block for this source
        for metric in METRICS:
            if metric not in r["metrics"]["per_source"][source]:
                raise ValueError(f"{r['name']}: {source!r} block missing {metric}")

    # check n and n_pos are consistent across runs for this source
    n = runs[0]["metrics"]["per_source"][source]["n"]
    n_pos = runs[0]["metrics"]["per_source"][source]["n_pos"]
    for r in runs[1:]:
        if r["metrics"]["per_source"][source]["n"] != n:
            raise ValueError(f"{r['name']}: {source!r} block has different n")
        if r["metrics"]["per_source"][source]["n_pos"] != n_pos:
            raise ValueError(f"{r['name']}: {source!r} block has different n_pos")

    # Store the results in the output dictionary
    stats = {}
    for metric in METRICS:
        values = [r["metrics"]["per_source"][source][metric] for r in runs]
        mean = sum(values) / len(values)
        if len(values) > 1:
            std = (sum((x - mean) ** 2 for x in values) / (len(values) - 1)) ** 0.5
        else:
            std = None
        stats[metric] = {"mean": mean, "std": std, "min": min(values), "max": max(values)}

    prevalence = n_pos / n if n > 0 else 0.0
    return {
        "source": source,
        "n_runs": len(runs),
        "n": n,
        "n_pos": n_pos,
        "prevalence": prevalence,
        **stats
    }


def peak_epoch(val_history: list[dict], select_on: str = "pr_auc") -> int:
    """Epoch with the best validation `select_on` value (the epoch best.pt came from).

    Input:  val_history.json content: [{"epoch": 0, "pr_auc": .., "roc_auc": ..}, ...]
    Output: the "epoch" field of the best entry (not its list position).
    Invariant: matches train.py's selection, which saves best.pt only on a STRICT improvement,
               so on a tie the earlier epoch is the one that was kept.
    """

    # reject an empty history.
    if not val_history:
        raise ValueError("Empty validation history")
    
    # find the entry with the highest `select_on`, keeping the earliest on ties.
    # if select_on is missing, raise ValueError (this should not happen if the run is valid).
    for entry in val_history:
        if select_on not in entry:
            raise ValueError(f"Missing {select_on} in val_history entry: {entry}")
    
    best_entry = max(val_history, key=lambda x: (x.get(select_on, float('-inf')), -x.get("epoch", float('inf'))))
    # return that entry's "epoch"
    return best_entry.get("epoch", None)

def truncation_split_metrics(preds: list[dict], flags: list[dict], threshold: float) -> dict:
    """RAGTruth test metrics for context-truncated vs untruncated rows (one run).

    Input:
        preds: predictions.jsonl lines {split, idx, source, task, group_id, label, prob}, all splits and sources.
        flags: truncation flags lines {split, idx, source, group_id, ctx_truncated}
               (scripts/report_truncation.py --per-row), all splits and sources.
        threshold: this run's validation-tuned threshold (metrics.json "threshold"), used for F1.
    Output:
        {"truncated":   {"pr_auc", "roc_auc", "f1", "n", "n_pos", "prevalence", ...},
         "untruncated": {... same keys ...}}
        computed on RAGTruth TEST rows only.
    Invariants:
        rows join on (split, idx), never idx alone (val and test both start at idx 0).
        group_id is not unique for RAGTruth, so it cannot be the join key, but it IS a consistency check:
        a joined pair must agree on source and group_id, otherwise the two files are misaligned.
    """

    # index the flags by (split, idx).
    flags_index = {(f["split"], f["idx"]): f for f in flags}

    # keep only predictions with split "test" and source "ragtruth".
    test_preds = [p for p in preds if p["split"] == "test" and p["source"] == "ragtruth"]

    # the flags file lists every row, so its RAGTruth test rows are the complete set
    test_flags = [f for f in flags if f["split"] == "test" and f["source"] == "ragtruth"]

    if not test_preds:
        raise ValueError("no RAGTruth test predictions found (wrong or empty predictions.jsonl?)")
    if len(test_preds) != len(test_flags):
        raise ValueError(f"{len(test_preds)} RAGTruth test predictions but {len(test_flags)} flags: "
                         "predictions.jsonl looks incomplete")
    if len({(p["split"], p["idx"]) for p in test_preds}) != len(test_preds):
        raise ValueError("duplicate (split, idx) in RAGTruth test predictions")

    # for each kept prediction, look up its flag; missing flag, or source / group_id disagreement -> ValueError.
    truncated_rows = []
    untruncated_rows = []
    for p in test_preds:
        key = (p["split"], p["idx"])

        if key not in flags_index:
            raise ValueError(f"Missing truncation flag for {key}")

        # get the corresponding flag
        f = flags_index[key]

        # check that the source and group_id match
        if p["source"] != f["source"]:
            raise ValueError(f"Source mismatch for {key}: pred={p['source']} flag={f['source']}")
        if p["group_id"] != f["group_id"]:
            raise ValueError(f"group_id mismatch for {key}: pred={p['group_id']} flag={f['group_id']}")
        
        # classify the row based on the truncation flag
        if f.get("ctx_truncated"):
            truncated_rows.append(p)
        else:
            untruncated_rows.append(p)

    # helper function to compute metrics for each group
    def compute_metrics(rows: list[dict]) -> dict:
        labels = np.array([r["label"] for r in rows])
        probs = np.array([r["prob"] for r in rows])
        n = len(rows)
        n_pos = int(np.sum(labels))
        prevalence = float(n_pos) / n if n > 0 else 0.0

        # if n_pos is 0 or n_pos == n, raise a ValueError.
        if n_pos == 0 or n_pos == n:
            raise ValueError(f"All labels are the same for this group: n={n}, n_pos={n_pos}")

        # per group, compute the threshold-free metrics and the thresholded metrics at `threshold`

        # compute PR-AUC and AUROC using ranking_metrics
        ranking_results = ranking_metrics(labels, probs)

        # compute F1 at the given threshold using thresholded_metrics
        thresholded_results = thresholded_metrics(labels, probs, threshold)

        return {
            "pr_auc": ranking_results["pr_auc"],
            "roc_auc": ranking_results["roc_auc"],
            "f1": thresholded_results["f1"],
            "n": n,
            "n_pos": n_pos,
            "prevalence": prevalence
        }
    
    # Compute metrics for the truncated and untruncated groups
    # put each row's label and prob into the truncated or the untruncated group.

    if not truncated_rows:
        raise ValueError("No truncated rows found")
    else:
        truncated_metrics = compute_metrics(truncated_rows)
    if not untruncated_rows:
        raise ValueError("No untruncated rows found")
    else:
        untruncated_metrics = compute_metrics(untruncated_rows)

    return {
        "truncated": truncated_metrics,
        "untruncated": untruncated_metrics
    }

# ---- report-------------------------------------------------------

def fmt_spread(d: dict) -> str:
    if d["std"] is None:
        return f"{d['mean']:.4f} (1 run)"
    return f"{d['mean']:.4f} ± {d['std']:.4f} [{d['min']:.4f}, {d['max']:.4f}]"


def _source_table(runs: list[dict], sources: list[str], label: str = "source") -> list[str]:
    lines = [f"| {label} | prevalence (n_pos / n) | PR-AUC | F1 @ val threshold | AUROC | runs |",
             "|---|---|---|---|---|---|"]
    for src in sources:
        s = summarize_source(runs, src)
        lines.append(f"| {src} | {s['prevalence']:.1%} ({s['n_pos']} / {s['n']}) | {fmt_spread(s['pr_auc'])} "
                     f"| {fmt_spread(s['f1'])} | {fmt_spread(s['roc_auc'])} | {s['n_runs']} |")
    return lines


def truncation_runs(runs: list[dict], flags: list[dict]) -> list[dict]:
    """Per run, the truncation split shaped like a run (two pseudo-sources), so summarize_source
    aggregates it across seeds exactly like the real sources."""
    out = []
    for r in runs:
        if r["predictions_path"] is None:
            raise ValueError(f"{r['name']}: no predictions.jsonl, cannot split by truncation")
        split = truncation_split_metrics(read_jsonl(r["predictions_path"]), flags, r["metrics"]["threshold"])
        out.append({"name": r["name"], "metrics": {"per_source": {
            "ragtruth_truncated": split["truncated"], "ragtruth_untruncated": split["untruncated"]}}})
    return out


def format_report(runs: list[dict], sources: list[str] | None = None, flags: list[dict] | None = None,
                  title: str = "G test metrics") -> str:
    if sources is None:
        sources = sorted({s for r in runs for s in r["metrics"].get("per_source", {})})
    lines = [f"# {title}", "",
             f"{len(runs)} run(s). Spread = mean ± sample std [min, max] across seeds. "
             "F1 uses each run's RAGTruth-validation threshold.", "",
             "## Runs", "",
             "| run folder | seed | git sha | dirty | device | threshold | peak val epoch |",
             "|---|---|---|---|---|---|---|"]
    for r in runs:
        snap = r["snapshot"] or {}
        select_on = snap.get("config", {}).get("train", {}).get("select_on", "pr_auc")
        peak = peak_epoch(r["val_history"], select_on) if r["val_history"] else "n/a"
        lines.append(f"| {r['name']} | {snap.get('seed', r['metrics'].get('seed', '?'))} "
                     f"| {snap.get('git_sha', r['metrics'].get('git_sha', '?'))} | {snap.get('git_dirty', '?')} "
                     f"| {snap.get('device_name', '?')} | {r['metrics'].get('threshold', '?')} | {peak} |")
    shas = {(r["snapshot"] or {}).get("git_sha") for r in runs}
    if len(shas) > 1:
        lines += ["", f"WARNING: runs come from {len(shas)} different commits {sorted(map(str, shas))}."]
    if any((r["snapshot"] or {}).get("git_dirty") for r in runs):
        lines += ["", "WARNING: at least one run had uncommitted changes to tracked files (git_dirty)."]

    lines += ["", "## Per source (test)", ""] + _source_table(runs, sources)

    if flags is not None:
        truns = truncation_runs(runs, flags)
        lines += ["", "## RAGTruth test: context truncated vs untruncated", "", "Per seed:", "",
                  "| run folder | subset | prevalence (n_pos / n) | PR-AUC | F1 | AUROC |",
                  "|---|---|---|---|---|---|"]
        for t in truns:
            for src in TRUNC_SOURCES:
                m = t["metrics"]["per_source"][src]
                lines.append(f"| {t['name']} | {src} | {m['prevalence']:.1%} ({m['n_pos']} / {m['n']}) "
                             f"| {m['pr_auc']:.4f} | {m['f1']:.4f} | {m['roc_auc']:.4f} |")
        lines += ["", "Across seeds:", ""] + _source_table(truns, list(TRUNC_SOURCES), label="subset")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="runs", help="folder containing run_seed*/ folders")
    ap.add_argument("--experiment", default="G", choices=("G", "C", "G+C"))
    ap.add_argument("--sources", nargs="*", default=None, help="default: every source in the runs")
    ap.add_argument("--trunc-flags", default="results/truncation_flags_v2.jsonl",
                    help="per-row truncation flags; skipped if the file does not exist")
    ap.add_argument("--out", default=None, help="also write the report here, e.g. results/phase2_G.md")
    args = ap.parse_args(argv)

    runs = load_runs(args.root)
    if not runs:
        print(f"No run_seed*/metrics.json found under {args.root!r}")
        return
    kept, dropped = filter_runs(runs, args.experiment)
    for name, reasons in dropped:
        print(f"  skipped {name}: {'; '.join(reasons)}", file=sys.stderr)
    if not kept:
        print(f"No valid {args.experiment} runs under {args.root!r}")
        return

    flags = read_jsonl(args.trunc_flags) if args.trunc_flags and os.path.exists(args.trunc_flags) else None
    if flags is None:
        print(f"  no truncation flags at {args.trunc_flags!r}: truncation split skipped", file=sys.stderr)
    text = format_report(kept, args.sources, flags, title=f"{args.experiment} test metrics ({args.root})")
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
