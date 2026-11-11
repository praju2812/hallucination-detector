"""Part A, clinical half (supervisor analysis, 11 Nov): untruncated input lengths of the clinical dataset.

Rows come from build() in scripts/build_clinical.py (pinned to BUILD_CLINICAL_SHA), run in memory with the
build_clinical.main() defaults (seed 42, val_frac 0.1); nothing is written under data/. Length per row is measured
exactly as for the general data (scripts/analysis/a_lengths.py): special tokens + "Question: {query} Context: {context}"
+ response, ModernBERT tokenizer, no truncation, self-checked against the Collator.

Sources: ragbench_pubmedqa, ragbench_covidqa, medhallu, and all clinical rows pooled; scopes train / test / all splits.

Run: python -m scripts.analysis.a_lengths_clinical
Writes: results/analysis_2026-11-11/a_lengths_clinical.json; prints markdown tables.
"""

from __future__ import annotations

import numpy as np

from scripts.analysis.a_lengths import COVERAGE_TARGETS, WINDOW_STEP, WINDOWS, coverage, describe, smallest_window
from scripts.analysis.common import (BUILD_CLINICAL_SHA, CLINICAL_SEED, CLINICAL_VAL_FRAC, SPLITS, full_lengths,
                                     import_build_clinical, load_config, load_tokenizer, md_table, self_check_lengths,
                                     write_json)
from utils.io import read_jsonl

GENERAL_TRAIN = "data/unified_v2/train.jsonl"


def main() -> None:
    bc, sha = import_build_clinical()
    print(f"scripts/build_clinical.py at {BUILD_CLINICAL_SHA} ({sha}); build() in memory, seed {CLINICAL_SEED}, "
          f"val_frac {CLINICAL_VAL_FRAC}")
    # build() returns (splits, report) at e36a747 and (splits, removed, report) after the 11 Nov rule change
    splits = bc.build(bc.load_raw(), read_jsonl(GENERAL_TRAIN), seed=CLINICAL_SEED, val_frac=CLINICAL_VAL_FRAC)[0]
    # train / val / test (clean after the rule change), plus the full official test sets when the build returns them
    measured = SPLITS + (("test_full",) if "test_full" in splits else ())
    splits = {s: splits[s] for s in measured}

    cfg = load_config()
    resp_max = cfg["model"]["resp_max"]
    tok = load_tokenizer(cfg)
    rows, split_of = [], []
    for s in measured:
        rows += splits[s]
        split_of += [s] * len(splits[s])
    split_of = np.array(split_of)
    source = np.array([r["source"] for r in rows])
    lens = full_lengths(tok, rows)
    checked = self_check_lengths(tok, rows, lens["total"])
    print(f"Self-check: full_lengths == Collator.encode_with_response_budget (window 10^6) on {checked} sampled rows. OK.")

    groups = {s: source == s for s in ("ragbench_pubmedqa", "ragbench_covidqa", "medhallu")}
    groups["clinical pooled"] = np.ones(len(rows), bool)
    scopes = {"train": split_of == "train", "test": split_of == "test"}
    if "test_full" in splits:
        scopes["test_full"] = split_of == "test_full"
    scopes["all splits"] = split_of != "test_full"  # train + val + test; test_full would count test rows twice
    result = {"build_clinical_sha": sha, "seed": CLINICAL_SEED, "val_frac": CLINICAL_VAL_FRAC,
              "self_check_rows": checked, "rows": {s: len(splits[s]) for s in measured}, "scopes": {}}

    for scope, sm in scopes.items():
        res = {}
        for g, gm in groups.items():
            m = sm & gm
            L = lens["total"][m]
            d = describe(L)
            entry = {"describe": d, "coverage": coverage(L, lens["query"][m], lens["response"][m], lens["n_special"], resp_max),
                     "pct_response_over_resp_max": float((lens["response"][m] > resp_max).mean() * 100)}
            for t in COVERAGE_TARGETS:
                w = smallest_window(L, t)
                entry[f"window_{t}"] = w
                entry[f"coverage_at_window_{t}"] = float((L <= w).mean() * 100)
            res[g] = entry
        result["scopes"][scope] = res

    for scope, res in result["scopes"].items():
        print(f"\n### A7a. Clinical length distribution, {scope} (tokens, untruncated)")
        print(md_table(["source", "n", "mean", "SD", "skew", "p50", "p90", "p95", "p97.5", "p99", "max"],
                       [[g, d["n"], round(d["mean"]), round(d["sd"]), f"{d['skew']:.2f}", round(d["p50"]), round(d["p90"]),
                         round(d["p95"]), round(d["p97.5"]), round(d["p99"]), round(d["max"])]
                        for g, v in res.items() for d in [v["describe"]]]))
    for scope, res in result["scopes"].items():
        print(f"\n### A7b. Clinical window rules and coverage, {scope} (smallest multiple of {WINDOW_STEP} with >= target % <= window)")
        print(md_table(["source", "mean+2SD", "its percentile", "% above", "log mean+2SD", "p95", "p97.5",
                        "window >= 95%", "coverage there", "window >= 97.5%", "coverage there"],
                       [[g, round(d["mean_2sd"]), f"{d['mean_2sd_pctile']:.1f}", f"{d['pct_above_mean_2sd']:.1f}",
                         round(d["log_mean_2sd"]), round(d["p95"]), round(d["p97.5"]),
                         v["window_95.0"], f"{v['coverage_at_window_95.0']:.1f}",
                         v["window_97.5"], f"{v['coverage_at_window_97.5']:.1f}"]
                        for g, v in res.items() for d in [v["describe"]]]))
    for scope, res in result["scopes"].items():
        print(f"\n### A7c. % clinical rows truncated (length > window), {scope}")
        print(md_table(["source"] + [str(w) for w in WINDOWS] + [f"% response > {resp_max}"],
                       [[g] + [f"{v['coverage'][w]['pct_truncated']:.1f}" for w in WINDOWS]
                        + [f"{v['pct_response_over_resp_max']:.1f}"] for g, v in res.items()]))

    path = write_json(result, "a_lengths_clinical.json")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
