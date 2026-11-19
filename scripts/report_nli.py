"""Phase 6 (M5): metrics and tables for the zero-shot NLI baseline.

Run (after scripts/score_nli.py):
    python -m scripts.report_nli

Reads results/nli/<model>/predictions.jsonl + scoring.json and data/clinical/test.jsonl (clean rows).
Writes:
    results/nli/<model>/metrics.json    full detail
    results/nli/<model>.json            M5 summary read by scripts/mvp_check.py: {"test": {source: {...}}}
    results/nli_baseline.md             tables

Thresholds as in RQ1 (decision, 19 Nov), same grid (src.metrics.best_threshold):
    (a) tuned on RAGTruth validation, applied to every source (zero-shot, primary);
    (b) tuned on each clinical source's own validation rows; CovidQA on pooled RAGBench validation.
For RAGTruth, (a) and (b) are the same threshold. Clean clinical test is derived from test_full by the RQ1
row key. NLI is deterministic, so there is one run and no seed spread. Metric logic lives in src/rq1.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from scripts.report_rq1 import NAMES, SOURCES, _f, _thr, all_flagged_f1, group_by, prev_cell
from scripts.score_nli import model_dir_name
from src.nli import MODEL_ID
from src.rq1 import derive_clean_predictions, subset_metrics, tune_clinical_thresholds
from src.train import get_git_dirty, get_git_sha
from utils.io import read_jsonl

RAGTRUTH = "ragtruth"
TASK_NAMES = {"qa": "QA", "summary": "Summary", "data2txt": "Data2txt"}
ALL_NAMES = {RAGTRUTH: "RAGTruth", **NAMES}


# ---- loading ------------------------------------------------------------------------------------

def load_scored(nli_root: str, model_dir: str, allow_partial: bool = False) -> dict:
    d = os.path.join(nli_root, model_dir)
    with open(os.path.join(d, "scoring.json")) as f:
        scoring = json.load(f)
    if scoring.get("limit") is not None and not allow_partial:
        raise ValueError(f"{d}: scored with --limit {scoring['limit']} (smoke output); not reportable")
    preds = read_jsonl(os.path.join(d, "predictions.jsonl"))
    for split, info in scoring["data_files"].items():
        n = sum(p["split"] == split for p in preds)
        if n != info["rows_expected"] and not allow_partial:
            raise ValueError(f"{d}: {n} {split} predictions but {info['rows_expected']} rows expected from {info['path']}")
    return {"dir": d, "scoring": scoring, "preds": preds}


# ---- metrics ---------------------------------------------------------------------------------------

def compute_metrics(scored: dict, clean_rows: list[dict]) -> dict:
    preds, scoring = scored["preds"], scored["scoring"]
    by_split = group_by(preds, "split")
    thr_rag = tune_clinical_thresholds(by_split["ragtruth_val"])[RAGTRUTH]
    thr_cl = tune_clinical_thresholds(by_split["clinical_val"])
    a = thr_rag["threshold"]
    full = by_split["clinical_test_full"]
    sets = {"test_full": full, "test_clean": derive_clean_predictions(full, clean_rows)}

    rt_test = by_split["ragtruth_test"]
    out = {
        "model": scoring["model"],
        "revision": scoring.get("revision"),
        "hypothesis": scoring["hypothesis"],
        "thresholds": {"ragtruth_val": thr_rag, "clinical_val": thr_cl},
        "ragtruth_test": subset_metrics(rt_test, a, a),
        "ragtruth_test_by_task": {str(t): subset_metrics(ps, a, a)
                                  for t, ps in sorted(group_by(rt_test, "task").items(), key=lambda kv: str(kv[0]))},
    }
    for name, ps in sets.items():
        by_src = group_by(ps, "source")
        out[name] = {s: subset_metrics(by_src[s], a, thr_cl[s]["threshold"]) for s in SOURCES if s in by_src}
    out["scoring"] = scoring
    out["report_git_sha"] = get_git_sha()
    out["report_git_dirty"] = get_git_dirty()
    return out


def build_m5_summary(metrics: dict, details_path: str) -> dict:
    """Thin file in the shape mvp_check.py M5 reads: "test": {source: {"pr_auc", ...}} for RAGTruth test and the
    clinical full test. F1 = F1 at threshold (a). No top-level "per_source" key (mvp_check would read it instead)."""
    def slim(m):
        return {"pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"], "f1": m["at_ragtruth_thr"]["f1"],
                "f1_at_clinical_thr": m["at_clinical_thr"]["f1"], "threshold": m["at_ragtruth_thr"]["threshold"],
                "prevalence": m["prevalence"], "n": m["n"], "n_pos": m["n_pos"]}
    return {
        "model": metrics["model"],
        "method": "nli_zero_shot",
        "hypothesis": metrics["hypothesis"],
        "test": {RAGTRUTH: slim(metrics["ragtruth_test"]), **{s: slim(m) for s, m in metrics["test_full"].items()}},
        "test_clean": {s: slim(m) for s, m in metrics["test_clean"].items()},
        "details": details_path,
    }


# ---- formatting ------------------------------------------------------------------------------------

COLS = [("PR-AUC", ("pr_auc",)), ("AUROC", ("roc_auc",)),
        ("P @ (a)", ("at_ragtruth_thr", "precision")), ("R @ (a)", ("at_ragtruth_thr", "recall")),
        ("F1 @ (a)", ("at_ragtruth_thr", "f1")), ("F1 @ (b)", ("at_clinical_thr", "f1"))]


def _get(d, path):
    for k in path:
        d = d[k]
    return d


def metrics_rows(label: str, items: list[tuple[str, dict]]) -> list[str]:
    head = (f"| {label} | prevalence (n_pos / n) | " + " | ".join(c for c, _ in COLS) + " | F1 if every row flagged |")
    lines = [head, "|" + "---|" * (len(COLS) + 3)]
    for name, m in items:
        cells = [_f(_get(m, path)) for _, path in COLS] + [_f(all_flagged_f1(m["prevalence"]))]
        lines.append(f"| {name} | {prev_cell(m)} | " + " | ".join(cells) + " |")
    return lines


def thresholds_table(m: dict) -> list[str]:
    tr, tc = m["thresholds"]["ragtruth_val"], m["thresholds"]["clinical_val"]
    lines = ["| threshold | used for | tuned on (validation) | positives / rows | value |", "|---|---|---|---|---|",
             f"| (a) | every source (primary) | RAGTruth | {tr['n_pos']} / {tr['n']} | {_thr(tr['threshold'])} |"]
    for s in SOURCES:
        t = tc[s]
        lines.append(f"| (b) | {NAMES[s]} | {', '.join(NAMES.get(x, x) for x in t['tuned_on'])} | "
                     f"{t['n_pos']} / {t['n']} | {_thr(t['threshold'])} |")
    return lines


def provenance_table(m: dict) -> list[str]:
    sc = m["scoring"]
    det = sc.get("determinism") or {}
    return ["| model | revision | hypothesis | windows | hypothesis cap | splitter | device | precision | "
            "scoring git sha | dirty | determinism |", "|" + "---|" * 11,
            f"| {sc['model']} | {sc.get('revision')} | {sc['hypothesis']} | {sc['max_length']} tokens, stride {sc['stride']} | "
            f"{sc['hyp_cap']} tokens | {sc['splitter']['name']} {sc['splitter']['version']} (newlines first, "
            f"< {sc['min_words']} words merged) | {sc['device']} ({sc['device_name']}) | "
            f"{sc['precision']['param_dtype']}, autocast={sc['precision']['autocast']} | {sc['git_sha']} | "
            f"{sc['git_dirty']} | {det.get('n')} rows, max abs diff {det.get('max_abs_diff', float('nan')):.1e} "
            f"({'PASS' if det.get('identical') else 'FAIL'}) |"]


def counts_note(sc: dict) -> str:
    parts = []
    for split, by_src in sc["counts"].items():
        for src, c in by_src.items():
            if c["fallback_rows"] or c["capped_hypotheses"]:
                parts.append(f"{split} {ALL_NAMES.get(src, src)}: {c['fallback_rows']} fallback rows, "
                             f"{c['capped_hypotheses']} capped hypotheses")
    return "; ".join(parts) if parts else "none in any split"


def format_report(m: dict) -> str:
    sc = m["scoring"]
    rt_items = [("RAGTruth", m["ragtruth_test"])] + [
        (f"RAGTruth {TASK_NAMES.get(t, t)}", v) for t, v in m["ragtruth_test_by_task"].items()]
    L = ["# NLI baseline: zero-shot DeBERTa NLI", "",
         f"Model {sc['model']}, no training. Score = 1 - min over response sentences of max over context windows of "
         "P(entailment). One deterministic run, so cells are single values (no seed spread). Threshold (a) = tuned "
         "on RAGTruth validation, applied to every source (primary, zero-shot). Threshold (b) = tuned on each "
         "clinical source's own validation rows (secondary). Same grid as training.", "",
         "## Run", ""] + provenance_table(m) + [
         "", "## Thresholds", ""] + thresholds_table(m) + [
         "", "## RAGTruth test", "",
         "For RAGTruth, (a) and (b) are the same threshold. The QA and Summary rows are a breakdown of the RAGTruth row "
         "(min over sentences may penalise long responses).", ""] + metrics_rows("source", rt_items) + [
         "", "## Clinical test: full", ""] + metrics_rows("source", [(NAMES[s], m["test_full"][s]) for s in SOURCES]) + [
         "", "## Clinical test: clean", ""] + metrics_rows("source", [(NAMES[s], m["test_clean"][s]) for s in SOURCES])
    tc = m["thresholds"]["clinical_val"]
    L += ["", "## Notes", "",
          f"- Hypothesis format: {sc['hypothesis']} (question in the hypothesis for QA rows: "
          f"{'yes' if sc['include_question'] else 'no'}; Summary rows never get a question).",
          f"- Sentences: newline split first, then {sc['splitter']['name']}; pieces under {sc['min_words']} words merged "
          f"into a neighbour; a response with nothing left is scored whole. Hypotheses capped at {sc['hyp_cap']} tokens "
          f"(question shortened first). Fallback rows and capped hypotheses: {counts_note(sc)}.",
          "- Threshold (b) tuning rows: " + "; ".join(
              f"{NAMES[s]} on {', '.join(NAMES.get(t, t) for t in tc[s]['tuned_on'])} ({tc[s]['n_pos']} / {tc[s]['n']} positives)"
              for s in SOURCES) + ".",
          "- MedHallu validation is pqa_artificial; MedHallu test is pqa_labeled.",
          "- Clean-test predictions are taken from the test_full predictions by row key (no row scored twice).",
          f"- Scoring: torch {sc['torch_version']}, transformers {sc['transformers_version']}, "
          f"{sc['pairs_scored']} (premise window, hypothesis) pairs.", ""]
    return "\n".join(L)


# ---- main ---------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nli-root", default="results/nli")
    ap.add_argument("--model-dir", default=model_dir_name(MODEL_ID), help="folder under --nli-root")
    ap.add_argument("--data-dir", default="data/clinical")
    ap.add_argument("--out-md", default="results/nli_baseline.md")
    ap.add_argument("--allow-partial", action="store_true", help="accept --limit (smoke) predictions; never for reporting")
    args = ap.parse_args(argv)

    scored = load_scored(args.nli_root, args.model_dir, args.allow_partial)
    clean_rows = read_jsonl(os.path.join(args.data_dir, "test.jsonl"))
    m = compute_metrics(scored, clean_rows)
    details = os.path.join(scored["dir"], "metrics.json")
    with open(details, "w") as f:
        json.dump(m, f, indent=2)
    with open(os.path.join(args.nli_root, f"{args.model_dir}.json"), "w") as f:
        json.dump(build_m5_summary(m, details), f, indent=2)
    text = format_report(m)
    os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
    with open(args.out_md, "w") as f:
        f.write(text)
    print(text)
    print(f"wrote {details}, {args.nli_root}/{args.model_dir}.json and {args.out_md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
