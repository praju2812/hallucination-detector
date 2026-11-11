"""Part A (supervisor analysis, 11 Nov): untruncated input length distribution on data/unified_v2.

Length per row = special tokens + "Question: {query} Context: {context}" + response, tokenised with the
ModernBERT tokenizer exactly as src/encoding.py does, with NO truncation (self-checked against the Collator).

Groups: halueval, ragtruth_qa, ragtruth_summary, ragtruth (QA + Summary), all; each for all splits and train only.
Also: A2b smallest 256-multiple window covering 95% / 97.5% of each group (train), and A3 G's AUROC / PR-AUC on
RAGTruth Summary test rows with vs without context truncation (Phase 2 flags + predictions).
Clinical sources: pending (they need Prajakta's build_clinical functions).

Run: python -m scripts.analysis.a_lengths [--flags ../hd-phase2/results/truncation_flags_v2.jsonl] [--runs-glob ...]
Writes: results/analysis_2026-11-11/a_lengths.json; prints markdown tables.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.analysis.common import (REPO, RUNS_GLOB, SPLITS, load_config, load_tokenizer, load_unified_v2, full_lengths,
                                     self_check_lengths, md_table, write_json)
from utils.io import read_jsonl

WINDOWS = (512, 1024, 1536, 2048, 4096)
PCTS = (50, 90, 95, 97.5, 99)
REF_WINDOW = 1024
WINDOW_STEP = 256
COVERAGE_TARGETS = (95.0, 97.5)
COVERAGE_GROUPS = ("halueval", "ragtruth_qa", "ragtruth_summary", "all sources")
DEFAULT_FLAGS = str(REPO.parent / "hd-phase2" / "results" / "truncation_flags_v2.jsonl")
# Phase 1: all 269 truncated RAGTruth test rows; Part A: no QA row exceeds 1024 -> (Summary truncated, untruncated, QA truncated)
EXPECTED_SUMMARY_TRUNC = (269, 631, 0)


def group_of(r: dict) -> str:
    return "halueval" if r["source"] == "halueval" else f"ragtruth_{r['task']}"


GROUPS = {
    "halueval": lambda g: g == "halueval",
    "ragtruth_qa": lambda g: g == "ragtruth_qa",
    "ragtruth_summary": lambda g: g == "ragtruth_summary",
    "ragtruth (qa+summary)": lambda g: g.startswith("ragtruth"),
    "all sources": lambda g: True,
}


def describe(L: np.ndarray) -> dict:
    L = L.astype(float)
    mean, sd = L.mean(), L.std(ddof=1)
    m2 = mean + 2 * sd
    logL = np.log(L)
    log_m2 = float(np.exp(logL.mean() + 2 * logL.std(ddof=1)))
    p95 = float(np.percentile(L, 95))
    return {
        "n": int(len(L)), "mean": mean, "sd": sd, "skew": float(stats.skew(L, bias=False)),
        "mean_2sd": m2,
        "mean_2sd_pctile": float(stats.percentileofscore(L, m2, kind="weak")),
        "pct_above_mean_2sd": float((L > m2).mean() * 100),
        "log_mean_2sd": log_m2,
        "log_mean_2sd_pctile": float(stats.percentileofscore(L, log_m2, kind="weak")),
        "pct_above_log_mean_2sd": float((L > log_m2).mean() * 100),
        "p95": p95, "pct_above_p95": float((L > p95).mean() * 100),
        **{f"p{p}": float(np.percentile(L, p)) for p in PCTS},
        "max": float(L.max()),
    }


def coverage(L: np.ndarray, query: np.ndarray, resp: np.ndarray, n_special: int, resp_max: int) -> dict:
    out = {}
    ref_lin = np.minimum(L, REF_WINDOW).astype(float).mean()
    ref_sq = (np.minimum(L, REF_WINDOW).astype(float) ** 2).mean()
    for w in WINDOWS:
        kept = np.minimum(L, w).astype(float)
        # response kept by encode_with_response_budget: min(resp, resp_max, budget - query part)
        resp_cap = np.minimum(resp_max, (w - n_special) - query)
        out[w] = {
            "pct_truncated": float((L > w).mean() * 100),
            "pct_response_cut": float((resp > resp_cap).mean() * 100),
            "mean_tokens_kept": float(kept.mean()),
            "proxy_lower_linear": float(kept.mean() / ref_lin),
            "proxy_upper_squared": float((kept ** 2).mean() / ref_sq),
        }
    return out


def smallest_window(L: np.ndarray, target_pct: float, step: int = WINDOW_STEP) -> int:
    """Smallest multiple of `step` with at least target_pct % of rows of length <= it."""
    w = step
    while (L <= w).mean() * 100 < target_pct:
        w += step
    return w


def truncation_effect(flags_path: str, runs_glob: str, test_rows: list[dict], lens: dict, test_offset: int,
                      n_special: int, resp_max: int, max_length: int = REF_WINDOW) -> dict:
    """G on RAGTruth Summary test rows, context-truncated vs untruncated (flags from Phase 2's report_truncation.py)."""
    flags = {}
    for f in read_jsonl(flags_path):
        if f["split"] != "test":
            continue
        r = test_rows[f["idx"]]
        if (f["source"], f["group_id"]) != (r["source"], r["group_id"]):
            raise ValueError(f"flags idx {f['idx']} does not match data/unified_v2/test.jsonl")
        flags[f["idx"]] = bool(f["ctx_truncated"])
    if len(flags) != len(test_rows):
        raise ValueError(f"{len(flags)} test flags, expected {len(test_rows)}")

    # recompute: context is cut when first segment + kept response + specials exceed the window
    mismatches = 0
    for i, r in enumerate(test_rows):
        j = test_offset + i
        expected = lens["first"][j] + min(lens["response"][j], resp_max) + n_special > max_length
        mismatches += expected != flags[i]
    if mismatches:
        raise ValueError(f"{mismatches} test rows: flag disagrees with recomputed truncation")

    summ = [i for i, r in enumerate(test_rows) if r["source"] == "ragtruth" and r["task"] == "summary"]
    qa_trunc = sum(flags[i] for i, r in enumerate(test_rows) if r["source"] == "ragtruth" and r["task"] == "qa")
    n_tr = sum(flags[i] for i in summ)
    if (n_tr, len(summ) - n_tr, qa_trunc) != EXPECTED_SUMMARY_TRUNC:
        raise ValueError(f"truncated/untruncated Summary test rows {n_tr}/{len(summ) - n_tr}, QA truncated {qa_trunc}; "
                         f"expected {EXPECTED_SUMMARY_TRUNC}")

    runs = []
    for path in sorted(glob.glob(runs_glob)):
        preds = {p["idx"]: p for p in read_jsonl(path) if p["split"] == "test"}
        if len(preds) != len(test_rows):
            raise ValueError(f"{path}: {len(preds)} test predictions, expected {len(test_rows)}")
        run = {"run": Path(path).parent.name}
        for sub, want in (("truncated", True), ("untruncated", False)):
            ids = [i for i in summ if flags[i] == want]
            for i in ids:
                p, r = preds[i], test_rows[i]
                if (p["source"], p["task"], p["group_id"], p["label"]) != (r["source"], r["task"], r["group_id"], r["label"]):
                    raise ValueError(f"{path}: idx {i} does not match data/unified_v2/test.jsonl")
            y = np.array([preds[i]["label"] for i in ids])
            prob = np.array([preds[i]["prob"] for i in ids])
            run[sub] = {"n": len(ids), "positives": int(y.sum()), "prevalence": float(y.mean()),
                        "auroc": float(roc_auc_score(y, prob)), "pr_auc": float(average_precision_score(y, prob))}
        runs.append(run)
    if not runs:
        raise FileNotFoundError(f"no predictions match {runs_glob}")
    summary = {}
    for sub in ("truncated", "untruncated"):
        a = np.array([r[sub]["auroc"] for r in runs]); p = np.array([r[sub]["pr_auc"] for r in runs])
        summary[sub] = {"auroc_mean": float(a.mean()), "auroc_sd": float(a.std(ddof=1)),
                        "pr_auc_mean": float(p.mean()), "pr_auc_sd": float(p.std(ddof=1)), "runs": len(runs)}
    return {"flags_path": flags_path, "n_truncated": n_tr, "n_untruncated": len(summ) - n_tr, "qa_truncated": qa_trunc,
            "mismatches": mismatches, "runs": runs, "summary": summary}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flags", default=DEFAULT_FLAGS, help="per-row truncation flags from Phase 2 report_truncation.py")
    ap.add_argument("--runs-glob", default=RUNS_GLOB)
    args = ap.parse_args()
    cfg = load_config()
    resp_max = cfg["model"]["resp_max"]
    tok = load_tokenizer(cfg)
    data = load_unified_v2()

    rows, split_of = [], []
    for s in SPLITS:
        rows += data[s]
        split_of += [s] * len(data[s])
    split_of = np.array(split_of)
    groups = np.array([group_of(r) for r in rows])

    lens = full_lengths(tok, rows)
    checked = self_check_lengths(tok, rows, lens["total"])
    n_special = lens["n_special"]
    print(f"Self-check: full_lengths == Collator.encode_with_response_budget (window 10^6) on {checked} sampled rows. OK.")
    print(f"Tokenizer: {cfg['model']['name']}; special tokens per pair: {n_special}; resp_max: {resp_max}")

    result = {"tokenizer": cfg["model"]["name"], "n_special": n_special, "resp_max": resp_max,
              "self_check_rows": checked, "scopes": {}}
    for scope, mask_split in (("all splits", np.ones(len(rows), bool)), ("train only", split_of == "train")):
        res_scope = {}
        for name, pred in GROUPS.items():
            m = mask_split & np.array([pred(g) for g in groups])
            L, q, r = lens["total"][m], lens["query"][m], lens["response"][m]
            res_scope[name] = {
                "describe": describe(L),
                "coverage": coverage(L, q, r, n_special, resp_max),
                "query_part": {"mean": float(q.mean()), "max": int(q.max())},
                "response": {"mean": float(r.mean()), "p95": float(np.percentile(r, 95)), "max": int(r.max()),
                             "pct_over_resp_max": float((r > resp_max).mean() * 100)},
            }
        result["scopes"][scope] = res_scope

    scopes = result["scopes"]
    for scope, res in scopes.items():
        print(f"\n### A1. Length distribution, {scope} (tokens, untruncated)")
        print(md_table(["group", "n", "mean", "SD", "skew", "p50", "p90", "p95", "p97.5", "p99", "max"],
                       [[g, d["n"], round(d["mean"]), round(d["sd"]), f"{d['skew']:.2f}", round(d["p50"]), round(d["p90"]),
                         round(d["p95"]), round(d["p97.5"]), round(d["p99"]), round(d["max"])]
                        for g, v in res.items() for d in [v["describe"]]]))

    for scope, res in scopes.items():
        print(f"\n### A2a. Candidate max_length rules, {scope}")
        print(md_table(["group", "mean+2SD", "its percentile", "% above", "log mean+2SD", "its percentile", "% above",
                        "p95", "% above"],
                       [[g, round(d["mean_2sd"]), round(d["mean_2sd_pctile"], 1), round(d["pct_above_mean_2sd"], 1),
                         round(d["log_mean_2sd"]), round(d["log_mean_2sd_pctile"], 1), round(d["pct_above_log_mean_2sd"], 1),
                         round(d["p95"]), round(d["pct_above_p95"], 1)]
                        for g, v in res.items() for d in [v["describe"]]]))

    # A2b window by coverage (train split)
    train_mask = split_of == "train"
    cov_rows = {}
    for name in COVERAGE_GROUPS:
        m = train_mask & np.array([GROUPS[name](g) for g in groups])
        L = lens["total"][m]
        d = describe(L)
        entry = {"n": int(len(L)), "p95": d["p95"], "p97.5": d["p97.5"], "mean_2sd": d["mean_2sd"],
                 "mean_2sd_pctile": d["mean_2sd_pctile"]}
        for t in COVERAGE_TARGETS:
            w = smallest_window(L, t)
            entry[f"window_{t}"] = w
            entry[f"coverage_at_window_{t}"] = float((L <= w).mean() * 100)
        cov_rows[name] = entry
    worst = {t: max(cov_rows, key=lambda g: cov_rows[g][f"window_{t}"]) for t in COVERAGE_TARGETS}
    result["window_by_coverage_train"] = {"groups": cov_rows, "step": WINDOW_STEP,
                                          "worst_group": {str(t): {"group": g, "window": cov_rows[g][f"window_{t}"]}
                                                          for t, g in worst.items()}}
    print(f"\n### A2b. Window by coverage, TRAIN split (smallest multiple of {WINDOW_STEP} with >= target % of rows <= window)")
    print(md_table(["group", "n", "p95", "p97.5", "mean+2SD", "its percentile",
                    "window >= 95%", "coverage there", "window >= 97.5%", "coverage there"],
                   [[g, e["n"], round(e["p95"]), round(e["p97.5"]), round(e["mean_2sd"]), round(e["mean_2sd_pctile"], 1),
                     e["window_95.0"], f"{e['coverage_at_window_95.0']:.1f}",
                     e["window_97.5"], f"{e['coverage_at_window_97.5']:.1f}"] for g, e in cov_rows.items()]
                   + [["**worst group**", "", "", "", "", "",
                       f"{cov_rows[worst[95.0]]['window_95.0']} ({worst[95.0]})", "",
                       f"{cov_rows[worst[97.5]]['window_97.5']} ({worst[97.5]})", ""]]))

    # A3 within-Summary truncation effect
    test_offset = len(data["train"]) + len(data["val"])
    trunc = truncation_effect(args.flags, args.runs_glob, data["test"], lens, test_offset, n_special, resp_max)
    result["summary_truncation_effect"] = trunc
    print("\n### A3. RAGTruth Summary TEST rows: context truncated vs untruncated at 1024 (G, Phase 2 runs)")
    print(f"Flags: {args.flags}. Truncated {trunc['n_truncated']} / untruncated {trunc['n_untruncated']}; "
          f"RAGTruth QA test rows truncated: {trunc['qa_truncated']}; flag vs recomputed-length mismatches: {trunc['mismatches']}.")
    body = []
    for run in trunc["runs"]:
        for sub in ("truncated", "untruncated"):
            s = run[sub]
            body.append([run["run"], sub, s["n"], s["positives"], f"{s['prevalence']:.3f}", f"{s['auroc']:.3f}", f"{s['pr_auc']:.3f}"])
    print(md_table(["run", "subset", "n", "positives", "prevalence", "AUROC", "PR-AUC"], body))
    print("\nMean +/- SD over runs (sample SD):")
    print(md_table(["subset", "AUROC", "PR-AUC"],
                   [[sub, f"{trunc['summary'][sub]['auroc_mean']:.3f} +/- {trunc['summary'][sub]['auroc_sd']:.3f}",
                     f"{trunc['summary'][sub]['pr_auc_mean']:.3f} +/- {trunc['summary'][sub]['pr_auc_sd']:.3f}"]
                    for sub in ("truncated", "untruncated")]))

    for scope, res in scopes.items():
        print(f"\n### A4. % rows truncated (length > window), {scope}")
        print(md_table(["group"] + [str(w) for w in WINDOWS],
                       [[g] + [round(v["coverage"][w]["pct_truncated"], 1) for w in WINDOWS] for g, v in res.items()]))

    for scope, res in scopes.items():
        print(f"\n### A5a. % rows whose response is cut (response > min(resp_max, window budget - query part)), {scope}")
        print(md_table(["group"] + [str(w) for w in WINDOWS],
                       [[g] + [round(v["coverage"][w]["pct_response_cut"], 1) for w in WINDOWS] for g, v in res.items()]))

    for scope, res in scopes.items():
        print(f"\n### A6. Compute PROXY relative to 1024 (lower = mean kept tokens; upper = mean squared kept tokens), {scope}")
        print(md_table(["group"] + [f"{w} (lower / upper)" for w in WINDOWS],
                       [[g] + [f"{v['coverage'][w]['proxy_lower_linear']:.2f} / {v['coverage'][w]['proxy_upper_squared']:.2f}"
                               for w in WINDOWS] for g, v in res.items()]))

    print("\n### A5b. Query part and response lengths (all splits)")
    res = result["scopes"]["all splits"]
    print(md_table(["group", "query part mean", "query part max", "response mean", "response p95", "response max",
                    f"% response > {resp_max}"],
                   [[g, round(v["query_part"]["mean"], 1), v["query_part"]["max"], round(v["response"]["mean"], 1),
                     round(v["response"]["p95"]), v["response"]["max"], round(v["response"]["pct_over_resp_max"], 1)]
                    for g, v in res.items()]))

    path = write_json(result, "a_lengths.json")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
