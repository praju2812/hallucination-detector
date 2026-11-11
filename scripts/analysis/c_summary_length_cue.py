"""Part C (supervisor analysis, 11 Nov): does the RAGTruth Summary word limit N carry label information?

The detector input drops N (query = "Summarize the following news"), but the response length is visible to it.
This script measures, on all RAGTruth Summary rows of data/unified_v2 (train/val/test):
  C1  hallucination rate by ratio = response words / N, and by N
  C2  AUROC of N, response words and ratio as single-feature predictors of the label (group-bootstrap 95% CI)
  C3  per-generator rates and length AUROCs, and length AUROC after stratifying by generator
  C4  correlation of G's predicted prob with response length and ratio on the Summary TEST rows, per seed
  C5  RAGTruth QA: is the dropped instruction identical on all 989 QA sources?

Run: python -m scripts.analysis.c_summary_length_cue [--runs-glob ../hd-phase2/runs/run_seed*/predictions.jsonl]
Writes: results/analysis_2026-11-11/c_summary_length_cue.json; prints markdown tables.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from scripts.analysis.common import (BOOT_SEED, N_BOOT, RUNS_GLOB, SPLITS, auroc, group_bootstrap_ci, load_unified_v2,
                                     load_source_info, load_responses, md_table, stratified_auroc, write_json)
from utils.io import read_jsonl

N_RE = re.compile(r"Summarize the following news within (\d+) words:")
QA_LINE1 = "Briefly answer the following question:"
RATIO_BINS = [(-np.inf, 0.8, "< 0.8"), (0.8, 1.0, "0.8 to < 1.0"), (1.0, 1.2, "1.0 to < 1.2"), (1.2, np.inf, ">= 1.2")]
EXPECTED_SUMMARY_SOURCES = 943
EXPECTED_QA_SOURCES = 989


def rate_table(y: np.ndarray, bins_of: np.ndarray, order: list[str]) -> list[dict]:
    out = []
    for b in order:
        m = bins_of == b
        n, pos = int(m.sum()), int(y[m].sum())
        out.append({"bin": b, "n": n, "positives": pos, "rate": pos / n if n else float("nan")})
    return out


def summary_rows(data: dict, info: dict) -> list[dict]:
    rows = []
    for split in SPLITS:
        for i, r in enumerate(data[split]):
            if not (r["source"] == "ragtruth" and r["task"] == "summary"):
                continue
            s = info[r["group_id"]]  # KeyError = a Summary row without its source record
            m = N_RE.fullmatch(s["prompt"].split("\n")[0])
            if m is None:
                raise ValueError(f"source {r['group_id']}: unexpected Summary prompt line 1")
            n_limit = int(m.group(1))
            words = len(r["answer"].split())
            rows.append({"split": split, "idx": i, "group_id": r["group_id"], "label": r["label"],
                         "answer": r["answer"], "N": n_limit, "words": words, "ratio": words / n_limit})
    return rows


def attach_generator(rows: list[dict], responses: list[dict]) -> None:
    """Add `model` by matching (source_id, response text); every row must match exactly one response record."""
    by_key = defaultdict(list)
    for r in responses:
        by_key[(r["source_id"], r["response"])].append(r)
    for row in rows:
        hits = by_key.get((row["group_id"], row["answer"]), [])
        models = {h["model"] for h in hits}
        if len(models) != 1:
            raise ValueError(f"source {row['group_id']}: {len(hits)} response matches, models {models}")
        if any((1 if h["labels"] else 0) != row["label"] for h in hits):
            raise ValueError(f"source {row['group_id']}: label disagrees with response.jsonl")
        row["model"] = models.pop()


def qa_instruction_check(info: dict, data: dict) -> dict:
    qa = [s for s in info.values() if s["task_type"] == "QA"]
    line1 = Counter(s["prompt"].split("\n")[0] for s in qa)
    templates = Counter()
    count_words = Counter()
    for s in qa:
        q, p = s["source_info"]["question"], s["source_info"]["passages"]
        prompt = s["prompt"]
        if prompt.count(q) < 1 or prompt.count(p) != 1:
            raise ValueError(f"QA source {s['source_id']}: question or passages not found verbatim in prompt")
        t = prompt.replace(p, "{passages}", 1).replace(q, "{question}", 1)
        templates[t] += 1
        m = re.search(r"strictly based on the following (\w+) passages?", t)
        count_words[m.group(1) if m else None] += 1
    generic = Counter(re.sub(r"the following \w+ passages?", "the following {k} passages", t) for t in templates.elements())
    # unified_v2 QA query = the bare question (instruction dropped)
    qa_rows = [r for s in SPLITS for r in data[s] if r["source"] == "ragtruth" and r["task"] == "qa"]
    query_is_question = sum(r["query"] == info[r["group_id"]]["source_info"]["question"].strip() for r in qa_rows)
    return {"qa_sources": len(qa), "line1_counts": dict(line1), "distinct_templates": len(templates),
            "templates": dict(templates),
            "passage_count_words": dict(count_words), "distinct_templates_count_word_masked": len(generic),
            "templates_masked": dict(generic), "unified_v2_qa_rows": len(qa_rows),
            "unified_v2_query_equals_question": query_is_question}


def prediction_correlations(pattern: str, test_rows: list[dict], summ_test: dict) -> list[dict]:
    """Per run: Pearson / Spearman of prob vs response words and vs ratio on Summary test rows."""
    out = []
    for path in sorted(glob.glob(pattern)):
        preds = [p for p in read_jsonl(path) if p["split"] == "test"]
        if len(preds) != len(test_rows):
            raise ValueError(f"{path}: {len(preds)} test predictions, expected {len(test_rows)}")
        y, prob, words, ratio = [], [], [], []
        for p in preds:
            r = test_rows[p["idx"]]
            if (p["source"], p["task"], p["group_id"], p["label"]) != (r["source"], r["task"], r["group_id"], r["label"]):
                raise ValueError(f"{path}: idx {p['idx']} does not match data/unified_v2/test.jsonl")
            if p["idx"] in summ_test:
                s = summ_test[p["idx"]]
                y.append(p["label"]); prob.append(p["prob"]); words.append(s["words"]); ratio.append(s["ratio"])
        y, prob, words, ratio = map(np.array, (y, prob, words, ratio))
        res = {"run": Path(path).parent.name, "n": int(len(y)), "positives": int(y.sum()), "prob_auroc": auroc(y, prob)}
        for subset, m in (("all", np.ones(len(y), bool)), ("label 0", y == 0), ("label 1", y == 1)):
            for feat, x in (("words", words), ("ratio", ratio)):
                res[f"{subset}|{feat}|pearson"] = float(stats.pearsonr(prob[m], x[m])[0])
                res[f"{subset}|{feat}|spearman"] = float(stats.spearmanr(prob[m], x[m])[0])
                res[f"{subset}|n"] = int(m.sum())
        out.append(res)
    if not out:
        raise FileNotFoundError(f"no predictions match {pattern}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-glob", default=RUNS_GLOB)
    args = ap.parse_args()

    data = load_unified_v2()
    info = load_source_info()
    n_summary_sources = sum(s["task_type"] == "Summary" for s in info.values())
    if n_summary_sources != EXPECTED_SUMMARY_SOURCES:
        raise ValueError(f"{n_summary_sources} Summary sources, expected {EXPECTED_SUMMARY_SOURCES}")

    rows = summary_rows(data, info)
    attach_generator(rows, load_responses())
    y = np.array([r["label"] for r in rows])
    feats = {"N (word limit)": np.array([r["N"] for r in rows], float),
             "response words": np.array([r["words"] for r in rows], float),
             "ratio = words / N": np.array([r["ratio"] for r in rows], float)}
    groups = np.array([r["group_id"] for r in rows])
    splits = np.array([r["split"] for r in rows])
    models = np.array([r["model"] for r in rows])
    result = {"summary_rows": len(rows), "summary_sources_in_rows": len(set(groups)),
              "positives": int(y.sum()), "bootstrap": {"resamples": N_BOOT, "seed": BOOT_SEED, "unit": "source_id"}}

    print(f"RAGTruth Summary rows (unified_v2, all splits): {len(rows)}, positives {int(y.sum())} "
          f"({y.mean():.1%}), sources {len(set(groups))} of {n_summary_sources}")

    # C1 rates by ratio and by N
    ratio = feats["ratio = words / N"]
    ratio_bin = np.array([next(lbl for lo, hi, lbl in RATIO_BINS if lo <= v < hi) for v in ratio])
    by_ratio = rate_table(y, ratio_bin, [b[2] for b in RATIO_BINS])
    n_vals = feats["N (word limit)"]
    distinct_n = sorted(set(n_vals.astype(int)))
    if len(distinct_n) <= 10:
        n_bin = np.array([str(int(v)) for v in n_vals]); n_order = [str(v) for v in distinct_n]; n_binning = "distinct values"
    else:
        q = np.quantile(n_vals, [0, .25, .5, .75, 1])
        edges = [(q[0], q[1]), (q[1], q[2]), (q[2], q[3]), (q[3], q[4])]
        labels = [f"{int(lo)} to {int(hi)}" if k == 0 else f">{int(lo)} to {int(hi)}" for k, (lo, hi) in enumerate(edges)]
        n_bin = np.array([labels[0] if v <= q[1] else labels[1] if v <= q[2] else labels[2] if v <= q[3] else labels[3]
                          for v in n_vals])
        n_order, n_binning = labels, f"quartiles ({len(distinct_n)} distinct N values)"
    by_n = rate_table(y, n_bin, n_order)
    result["N_distinct"] = len(distinct_n)
    result["N_range"] = [int(min(distinct_n)), int(max(distinct_n))]
    result["ratio_summary"] = {"mean": float(ratio.mean()), "p5": float(np.percentile(ratio, 5)),
                               "p50": float(np.median(ratio)), "p95": float(np.percentile(ratio, 95))}
    result["rate_by_ratio"] = by_ratio
    result["rate_by_N"] = {"binning": n_binning, "rows": by_n}

    print("\n### C1a. Hallucination rate by ratio = response words / N (all splits)")
    print(md_table(["ratio bin", "n", "positives", "rate"],
                   [[b["bin"], b["n"], b["positives"], f"{b['rate']:.3f}"] for b in by_ratio]))
    print(f"\nRatio: mean {ratio.mean():.2f}, p5 {np.percentile(ratio, 5):.2f}, median {np.median(ratio):.2f}, "
          f"p95 {np.percentile(ratio, 95):.2f}. N: {len(distinct_n)} distinct values, {min(distinct_n)} to {max(distinct_n)}.")
    print(f"\n### C1b. Hallucination rate by N ({n_binning})")
    print(md_table(["N bin", "n", "positives", "rate"],
                   [[b["bin"], b["n"], b["positives"], f"{b['rate']:.3f}"] for b in by_n]))

    # C2 single-feature AUROC, pooled and per split
    result["auroc"] = {}
    table = []
    for scope in ("all splits",) + SPLITS:
        m = np.ones(len(y), bool) if scope == "all splits" else splits == scope
        for name, x in feats.items():
            a = auroc(y[m], x[m])
            lo, hi = group_bootstrap_ci(y[m], x[m], groups[m])
            result["auroc"][f"{scope}|{name}"] = {"n": int(m.sum()), "positives": int(y[m].sum()), "auroc": a, "ci95": [lo, hi]}
            table.append([scope, name, int(m.sum()), int(y[m].sum()), f"{a:.3f}", f"{lo:.3f} to {hi:.3f}"])
    print("\n### C2. Single-feature AUROC for the label (> 0.5: larger value, more often hallucinated)")
    print(md_table(["rows", "feature", "n", "positives", "AUROC", "95% CI (group bootstrap)"], table))

    # C3 generators
    gen = []
    for mname in sorted(set(models)):
        m = models == mname
        gen.append({"model": mname, "n": int(m.sum()), "positives": int(y[m].sum()), "rate": float(y[m].mean()),
                    "mean_words": float(feats["response words"][m].mean()),
                    "mean_ratio": float(ratio[m].mean()),
                    "auroc_words": auroc(y[m], feats["response words"][m]), "auroc_ratio": auroc(y[m], ratio[m])})
    result["generators"] = gen
    print("\n### C3a. Per generator (all splits)")
    print(md_table(["generator", "n", "positives", "rate", "mean words", "mean ratio", "AUROC words", "AUROC ratio"],
                   [[g["model"], g["n"], g["positives"], f"{g['rate']:.3f}", round(g["mean_words"]), f"{g['mean_ratio']:.2f}",
                     f"{g['auroc_words']:.3f}", f"{g['auroc_ratio']:.3f}"] for g in gen]))

    strat = {}
    for fname in ("response words", "ratio = words / N"):
        x = feats[fname]
        weighted, within_rank = stratified_auroc(y, x, models)
        strat[fname] = {"pooled_raw": auroc(y, x), "weighted_within_generator": weighted,
                        "pooled_within_generator_rank": auroc(y, within_rank),
                        "pooled_within_generator_rank_ci95": list(group_bootstrap_ci(y, within_rank, groups))}
    # generator identity alone (score = generator's in-sample rate): optimistic, context only
    gen_rate = {g["model"]: g["rate"] for g in gen}
    strat["generator identity (in-sample rate)"] = {"pooled_raw": auroc(y, np.array([gen_rate[m] for m in models]))}
    result["stratified"] = strat
    print("\n### C3b. Length AUROC after stratifying by generator (all splits)")
    print(md_table(["feature", "pooled AUROC", "weighted mean of within-generator AUROCs",
                    "pooled AUROC of within-generator rank", "95% CI (rank version)"],
                   [[f, f"{v['pooled_raw']:.3f}", f"{v['weighted_within_generator']:.3f}",
                     f"{v['pooled_within_generator_rank']:.3f}",
                     "{:.3f} to {:.3f}".format(*v["pooled_within_generator_rank_ci95"])]
                    for f, v in strat.items() if "weighted_within_generator" in v]))
    print(f"\nGenerator identity alone (score = generator's in-sample hallucination rate; optimistic): "
          f"AUROC {strat['generator identity (in-sample rate)']['pooled_raw']:.3f}")

    # C4 G's predictions
    test_rows = data["test"]
    summ_test = {r["idx"]: r for r in rows if r["split"] == "test"}
    corr = prediction_correlations(args.runs_glob, test_rows, summ_test)
    result["prediction_correlations"] = corr
    result["summary_test_length_auroc"] = {k: result["auroc"][f"test|{k}"]["auroc"] for k in feats}
    print("\n### C4. G (Phase 2 runs): prob vs response length and ratio, RAGTruth Summary TEST rows")
    hdr = ["run", "n", "G prob AUROC"]
    body = []
    for c in corr:
        line = [c["run"], c["n"], f"{c['prob_auroc']:.3f}"]
        for subset in ("all", "label 0", "label 1"):
            for feat in ("words", "ratio"):
                line.append(f"{c[f'{subset}|{feat}|pearson']:.2f} / {c[f'{subset}|{feat}|spearman']:.2f}")
        body.append(line)
    for subset in ("all", "label 0", "label 1"):
        for feat in ("words", "ratio"):
            hdr.append(f"{subset}: prob vs {feat} (Pearson / Spearman)")
    print(md_table(hdr, body))
    print(f"\nrows per subset: all {corr[0]['all|n']}, label 0 {corr[0]['label 0|n']}, label 1 {corr[0]['label 1|n']}; "
          "single-feature AUROC on the same test rows: "
          + ", ".join(f"{k} {v:.3f}" for k, v in result["summary_test_length_auroc"].items()))

    # C5 QA instruction
    qa = qa_instruction_check(info, data)
    result["qa_instruction"] = qa
    if qa["qa_sources"] != EXPECTED_QA_SOURCES:
        raise ValueError(f"{qa['qa_sources']} QA sources, expected {EXPECTED_QA_SOURCES}")
    print("\n### C5. RAGTruth QA: the instruction dropped from the query")
    print(md_table(["check", "value"], [
        ["QA sources", qa["qa_sources"]],
        ["distinct prompt line 1", len(qa["line1_counts"])],
        [f"sources with line 1 = '{QA_LINE1}'", qa["line1_counts"].get(QA_LINE1, 0)],
        ["distinct templates (prompt minus question and passages)", qa["distinct_templates"]],
        ["distinct templates with the passage-count word masked", qa["distinct_templates_count_word_masked"]],
        ["passage-count words", ", ".join(f"{k}: {v}" for k, v in sorted(qa["passage_count_words"].items(), key=str))],
        ["unified_v2 QA rows with query == stripped question", f"{qa['unified_v2_query_equals_question']} of {qa['unified_v2_qa_rows']}"],
    ]))
    for t, n in qa["templates_masked"].items():
        print(f"\nTemplate ({n} sources):\n```\n{t}\n```")

    path = write_json(result, "c_summary_length_cue.json")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
