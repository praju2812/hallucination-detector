"""Phase 5 (RQ2): G, C and G+C side by side on the general and clinical test sets.

Run from the repo root, once the C and G+C runs are in runs/ and G has been scored for RQ1:
    python -m scripts.report_rq2

Reads:
    runs/<run>/config_snapshot.json, metrics.json, predictions.jsonl    every G, C and G+C run folder
    results/rq1/<G run>/metrics.json                                      G on clinical data (scripts/report_rq1.py)
    the data files named in each run's snapshot                           row keys, MedHallu fields, training-set sizes
    data/clinical/test.jsonl                                              clean clinical test (derived from test_full)
Writes:
    results/rq2/<run>/metrics.json   per run: the RQ1 layout (at_ragtruth_thr renamed at_own_thr), plus the general
                                     test block, thresholds with what they were tuned on, and the training data
    results/rq2/<run>.json           thin summary (clean clinical test, full clinical test, general test)
    results/rq2.md                   G | C | G+C tables

Decisions (Prajakta, 21 Nov): headline = clean clinical test, full beside it (contamination contrast); threshold (b)
kept for all three models, reporting only; differences are descriptive (mean +/- SD, difference of means).
All metric logic is reused: src/metrics.py, src/rq1.py and scripts/report_rq1.compute_seed_metrics.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import statistics
import sys

import numpy as np

from scripts.report_rq1 import BOOT_SOURCE, NAMES, SOURCES, compute_seed_metrics
from src.metrics import best_threshold, ranking_metrics, thresholded_metrics
from src.rq1 import assign_row_keys
from src.train import get_git_dirty, get_git_sha, named_test_sets, resolved_training_values, select_sources_from_cfg
from utils.io import read_jsonl

EXPERIMENTS = ["G", "C", "G+C"]
GEN = ["ragtruth", "halueval"]
ALL_NAMES = {**NAMES, "ragtruth": "RAGTruth", "halueval": "HaluEval"}
EXPECTED_MAX_LENGTH, EXPECTED_RESP_MAX, EXPECTED_TRUNCATION = 1024, 384, "fit_first"
RENAME = {"at_ragtruth_thr": "at_own_thr", "f1_at_ragtruth_thr": "f1_at_own_thr"}
RESOLUTION = 0.05            # decision D4: with 3 seeds, smaller PR-AUC differences are not resolvable
TEST_SETS = [("test_clean", "Clean clinical test (headline)"), ("test_full", "Full clinical test (contamination contrast)")]
MEDHALLU_FIELDS = ["difficulty", "category"]
DIFFICULTY_ORDER = ["easy", "medium", "hard"]


# ---- loading and checks -----------------------------------------------------------------------

def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _as_list(v) -> list[str]:
    return [v] if isinstance(v, str) else list(v)


def load_runs(runs_root: str) -> list[dict]:
    runs = []
    for snap_path in sorted(glob.glob(os.path.join(runs_root, "*", "config_snapshot.json"))):
        d = os.path.dirname(snap_path)
        mp, pp = os.path.join(d, "metrics.json"), os.path.join(d, "predictions.jsonl")
        runs.append({"name": os.path.basename(d), "dir": d, "snap": _load_json(snap_path),
                     "metrics": _load_json(mp) if os.path.exists(mp) else None,
                     "preds_path": pp if os.path.exists(pp) else None})
    return runs


def split_files(cfg_data: dict) -> dict[str, list[str]]:
    """{split: files in the order the run's loader concatenated them}: val plus every named test set."""
    out = {"val": _as_list(cfg_data["val_path"])}
    out.update({name: _as_list(p) for name, p in named_test_sets(cfg_data).items()})
    return out


def check_reportable(run: dict, rq1_root: str) -> None:
    """Raise ValueError unless the run can go in the RQ2 tables (G also needs its RQ1 metrics)."""
    name, snap = run["name"], run["snap"]
    cfg = snap.get("config", {})
    exp = snap.get("experiment")
    if exp not in EXPERIMENTS:
        raise ValueError(f"{name}: experiment {exp!r} is not one of {EXPERIMENTS}")
    if cfg["model"].get("max_length") != EXPECTED_MAX_LENGTH or cfg["model"].get("resp_max") != EXPECTED_RESP_MAX:
        raise ValueError(f"{name}: max_length/resp_max {cfg['model'].get('max_length')}/{cfg['model'].get('resp_max')}, "
                         f"expected {EXPECTED_MAX_LENGTH}/{EXPECTED_RESP_MAX}")
    if snap.get("truncation") != EXPECTED_TRUNCATION:
        raise ValueError(f"{name}: truncation {snap.get('truncation')!r}, expected {EXPECTED_TRUNCATION!r}")
    paths = _as_list(cfg["data"]["train_path"]) + [p for ps in split_files(cfg["data"]).values() for p in ps]
    tiny = [p for p in paths if "tiny" in str(p)]
    if tiny:
        raise ValueError(f"{name}: tiny smoke-test data {tiny}; not reportable")
    if run["metrics"] is None or run["preds_path"] is None:
        raise ValueError(f"{name}: metrics.json or predictions.jsonl missing")
    if exp == "G" and not os.path.exists(os.path.join(rq1_root, name, "metrics.json")):
        raise ValueError(f"{name}: no {rq1_root}/{name}/metrics.json; run scripts/score_rq1.py and "
                         f"scripts/report_rq1.py for this G run first")


def _rows(path: str, cache: dict) -> list[dict]:
    if path not in cache:
        cache[path] = read_jsonl(path)
    return cache[path]


def attach_row_info(preds: list[dict], split_rows: dict[str, list[dict]]) -> list[dict]:
    """Copies of preds with row_key, difficulty and category taken from the rows they were scored on.

    split_rows: {split: rows in loader order}. Splits not listed pass through unchanged.
    Raises ValueError if a split's prediction count differs from its row count, or a label or source disagrees."""
    counts = collections.Counter(p["split"] for p in preds)
    for split, rows in split_rows.items():
        if counts.get(split, 0) != len(rows):
            raise ValueError(f"{split}: {counts.get(split, 0)} predictions but {len(rows)} rows")
    keys = {split: assign_row_keys(rows) for split, rows in split_rows.items()}
    out = []
    for p in preds:
        q = dict(p)
        split = p["split"]
        if split in split_rows:
            i = p["idx"]
            if not 0 <= i < len(split_rows[split]):
                raise ValueError(f"{split}: idx {i} out of range")
            r = split_rows[split][i]
            if int(r["label"]) != int(p["label"]) or r["source"] != p["source"]:
                raise ValueError(f"{split} idx {i}: file has {r['source']}/{r['label']}, "
                                 f"prediction has {p['source']}/{p['label']}")
            q.update(row_key=keys[split][i], difficulty=r.get("difficulty"), category=r.get("category"))
        out.append(q)
    return out


def relabel(obj):
    """RQ1 calls the run's own threshold 'ragtruth'; for C and G+C it is not, so every such key becomes 'own'."""
    if isinstance(obj, dict):
        return {RENAME.get(k, k): relabel(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [relabel(v) for v in obj]
    return obj


# ---- per-run blocks -----------------------------------------------------------------------------

def _yp(preds):
    return np.array([p["label"] for p in preds], dtype=float), np.array([p["prob"] for p in preds], dtype=float)


def general_block(preds: list[dict], own_thr: float) -> tuple[dict, float]:
    """RAGTruth and HaluEval test metrics at the run's own threshold and at a threshold tuned on its RAGTruth
    validation predictions (decision D1). Returns (block, ragtruth_val_threshold)."""
    val_rt = [p for p in preds if p["split"] == "val" and p["source"] == "ragtruth"]
    if not val_rt:
        raise ValueError("no RAGTruth validation predictions")
    thr_rt = best_threshold(*_yp(val_rt))
    out = {}
    for s in GEN:
        y, p = _yp([q for q in preds if q["split"] == "test" and q["source"] == s])
        if len(y) == 0:
            raise ValueError(f"no {s} test predictions")
        m = ranking_metrics(y, p)
        m.update(n=int(len(y)), n_pos=int(y.sum()), prevalence=float(y.mean()),
                 at_own_thr=thresholded_metrics(y, p, own_thr),
                 at_ragtruth_val_thr=thresholded_metrics(y, p, thr_rt))
        out[s] = m
    return out, float(thr_rt)


def training_data(cfg: dict, cache: dict) -> dict:
    files = _as_list(cfg["data"]["train_path"])
    rows = [r for f in files for r in _rows(f, cache)]
    keep = cfg["train"].get("train_sources")
    if keep:
        rows = [r for r in rows if r["source"] in keep]
    per = {}
    for r in rows:
        e = per.setdefault(r["source"], {"n": 0, "n_pos": 0})
        e["n"] += 1
        e["n_pos"] += int(r["label"])
    return {"files": files, "n": len(rows), "n_pos": sum(e["n_pos"] for e in per.values()), "per_source": per}


def build_record(run: dict, rq1_root: str, clean_rows: list[dict], cache: dict, n_boot: int, boot_seed: int) -> dict:
    name, snap, metrics = run["name"], run["snap"], run["metrics"]
    cfg, exp, seed = snap["config"], snap["experiment"], snap["seed"]
    preds = read_jsonl(run["preds_path"])
    files = split_files(cfg["data"])
    attached = attach_row_info(preds, {s: [r for f in fs for r in _rows(f, cache)] for s, fs in files.items()})
    own_thr = float(metrics["threshold"])
    tuned_on = select_sources_from_cfg(cfg)
    if exp != "G" and metrics.get("threshold_tuned_on") != tuned_on:
        raise ValueError(f"{name}: metrics threshold_tuned_on {metrics.get('threshold_tuned_on')} != config {tuned_on}")
    general, thr_rt = general_block(attached, own_thr)
    for s in GEN:
        recorded = metrics["per_source"][s]["pr_auc"]
        if not math.isclose(general[s]["pr_auc"], recorded, abs_tol=1e-9):
            raise ValueError(f"{name}: {s} PR-AUC from predictions {general[s]['pr_auc']} != metrics.json {recorded}")

    if exp == "G":
        m = _load_json(os.path.join(rq1_root, name, "metrics.json"))
        for k, want in (("experiment", "G"), ("truncation", EXPECTED_TRUNCATION), ("run", name), ("seed", seed)):
            if m.get(k) != want:
                raise ValueError(f"{name}: RQ1 metrics {k} = {m.get(k)!r}, expected {want!r}")
        if not math.isclose(m["thresholds"]["ragtruth_val"], own_thr) or not math.isclose(thr_rt, own_thr):
            raise ValueError(f"{name}: G's own threshold {own_thr} should equal its RAGTruth-val threshold "
                             f"(RQ1 {m['thresholds']['ragtruth_val']}, recomputed {thr_rt})")
        clinical_from = os.path.join(rq1_root, name, "metrics.json")
    else:
        keep = [p for p in attached if (p["split"] == "val" and p["source"] in SOURCES) or p["split"] == "test_full"]
        scoring = {"seed": seed, "experiment": exp, "truncation": snap["truncation"], "git_sha": snap.get("git_sha"),
                   "git_dirty": snap.get("git_dirty"), "device_name": snap.get("device_name"),
                   "predictions": run["preds_path"]}
        m = compute_seed_metrics({"run": name, "preds": keep, "phase2": metrics, "scoring": scoring},
                                 clean_rows, n_boot, boot_seed)
        clinical_from = run["preds_path"]

    td = training_data(cfg, cache)
    resolved = snap.get("resolved")
    if resolved is None:          # the locked G runs predate "resolved": recompute from the same rule, and say so
        steps = math.ceil(td["n"] / cfg["train"]["batch_size"])
        resolved = {**resolved_training_values(cfg, td["n"], steps), "computed_by_report": True}
    return relabel({
        "run": name, "experiment": exp, "seed": seed, "truncation": snap["truncation"],
        "git_sha": snap.get("git_sha"), "git_dirty": snap.get("git_dirty"), "device_name": snap.get("device_name"),
        "resolved": resolved,
        "thresholds": {"own": {"threshold": own_thr, "tuned_on": tuned_on}, "ragtruth_val": thr_rt,
                       "clinical_val": m["thresholds"]["clinical_val"]},
        "training_data": td,
        "general": general,
        "test_clean": m["test_clean"],
        "test_full": m["test_full"],
        "medhallu_by_difficulty": m["medhallu_by_difficulty"],
        "medhallu_by_category": m["medhallu_by_category"],
        "covidqa_bootstrap": m["covidqa_bootstrap"],
        "n_duplicate_rows_test_full": m["n_duplicate_rows_test_full"],
        "clinical_metrics_from": clinical_from,
        "report_git_sha": get_git_sha(),
        "report_git_dirty": get_git_dirty(),
    })


def summary(rec: dict) -> dict:
    """Thin per-run file. No top-level "per_source" (mvp_check reads that key from run folders only)."""
    def slim(block):
        return {s: {"pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"], "f1": m["at_own_thr"]["f1"],
                    "threshold": m["at_own_thr"]["threshold"], "prevalence": m["prevalence"],
                    "n": m["n"], "n_pos": m["n_pos"]} for s, m in block.items()}
    return {"seed": rec["seed"], "experiment": rec["experiment"], "run": rec["run"],
            "test": slim(rec["test_clean"]), "test_full": slim(rec["test_full"]), "general": slim(rec["general"]),
            "details": os.path.join("results", "rq2", rec["run"], "metrics.json")}


# ---- formatting -----------------------------------------------------------------------------------

def _dig(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _vals(recs, path):
    return [_dig(r, path) for r in recs]


def cell(values: list) -> str:
    if not values:
        return "no runs"
    if any(v is None for v in values):
        return f"undefined in {sum(v is None for v in values)} of {len(values)}"
    if len(values) == 1:
        return f"{values[0]:.4f} (1 seed)"
    return f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}"


def diff_cell(a: list, b: list) -> str:
    if not a or not b or any(v is None for v in a + b):
        return "n/a"
    return f"{statistics.mean(a) - statistics.mean(b):+.4f}"


def _prevalence(by_exp, block, key) -> str:
    seen = {(m["n"], m["n_pos"]) for recs in by_exp.values() for r in recs
            for m in [_dig(r, block + [key])] if m is not None}
    if len(seen) > 1:
        raise ValueError(f"{'/'.join(block)} {key}: runs were tested on different rows {sorted(seen)}")
    if not seen:
        return "n/a"
    n, n_pos = seen.pop()
    return f"{100 * n_pos / n:.1f}% ({n_pos} / {n})"


def compare_table(by_exp: dict, block: list, keys: list, metric: tuple, label: str) -> list[str]:
    lines = [f"| {label} | prevalence (n_pos / n) | G | C | G+C | C - G | G+C - G |", "|---|---|---|---|---|---|---|"]
    for k in keys:
        v = {e: _vals(by_exp[e], block + [k] + list(metric)) for e in EXPERIMENTS}
        lines.append(f"| {ALL_NAMES.get(k, k)} | {_prevalence(by_exp, block, k)} | "
                     + " | ".join(cell(v[e]) for e in EXPERIMENTS)
                     + f" | {diff_cell(v['C'], v['G'])} | {diff_cell(v['G+C'], v['G'])} |")
    return lines


def training_table(by_exp: dict) -> list[str]:
    lines = ["| experiment | training files | rows | positives (prevalence) | per source: rows (positives) | rows vs G |",
             "|---|---|---|---|---|---|"]
    tds = {}
    for e in EXPERIMENTS:
        if not by_exp[e]:
            continue
        kinds = {json.dumps(r["training_data"], sort_keys=True) for r in by_exp[e]}
        if len(kinds) > 1:
            raise ValueError(f"{e}: runs were trained on different data")
        tds[e] = by_exp[e][0]["training_data"]
    for e, td in tds.items():
        per = "; ".join(f"{ALL_NAMES.get(s, s)} {m['n']:,} ({m['n_pos']:,})" for s, m in td["per_source"].items())
        ratio = f"{td['n'] / tds['G']['n']:.2f}x" if "G" in tds else "n/a"
        lines.append(f"| {e} | {', '.join(td['files'])} | {td['n']:,} | {td['n_pos']:,} "
                     f"({100 * td['n_pos'] / td['n']:.1f}%) | {per} | {ratio} |")
    return lines


def runs_table(by_exp: dict) -> list[str]:
    lines = ["| run folder | experiment | seed | git sha | dirty | device | own threshold (tuned on) | "
             "RAGTruth-val threshold | thr (b) PubMedQA | thr (b) CovidQA (pooled) | thr (b) MedHallu | warmup steps |",
             "|" + "---|" * 12]
    for e in EXPERIMENTS:
        for r in by_exp[e]:
            t = r["thresholds"]
            w = r["resolved"]
            warm = f"{w['warmup_steps']}" + (" (computed by the report; not recorded by the run)"
                                             if w.get("computed_by_report") else "")
            lines.append(f"| {r['run']} | {e} | {r['seed']} | {r['git_sha']} | {r['git_dirty']} | {r['device_name']} | "
                         f"{t['own']['threshold']:.2f} ({', '.join(t['own']['tuned_on'])}) | {t['ragtruth_val']:.2f} | "
                         + " | ".join(f"{t['clinical_val'][s]['threshold']:.2f}" for s in SOURCES) + f" | {warm} |")
    return lines


def _ci(r):
    return f"{r['point']:.4f} [undefined]" if r["lo"] is None else f"{r['point']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}]"


def bootstrap_table(by_exp: dict) -> list[str]:
    lines = ["| experiment | run | seed | test set | PR-AUC [95% CI] | AUROC [95% CI] | F1 @ own threshold [95% CI] | groups |",
             "|" + "---|" * 8]
    for ts, ts_name in TEST_SETS:
        for e in EXPERIMENTS:
            for r in by_exp[e]:
                b = r["covidqa_bootstrap"][ts]
                ci = b["ci"]
                lines.append(f"| {e} | {r['run']} | {r['seed']} | {ts_name.split(' (')[0].replace(' clinical test', '')} | "
                             f"{_ci(ci['pr_auc'])} | {_ci(ci['roc_auc'])} | {_ci(ci['f1_at_own_thr'])} | {b['n_groups']} |")
    return lines


def _subset_keys(by_exp, field):
    keys = sorted({k for recs in by_exp.values() for r in recs for k in r[f"medhallu_by_{field}"]["test_clean"]})
    if field == "difficulty":
        return [k for k in DIFFICULTY_ORDER if k in keys] + [k for k in keys if k not in DIFFICULTY_ORDER]
    return keys


def removed_rows_note(by_exp) -> str:
    rec = next(r for e in EXPERIMENTS for r in by_exp[e])
    parts = [f"{NAMES[s]} {rec['test_full'][s]['n'] - rec['test_clean'][s]['n']:,} of {rec['test_full'][s]['n']:,}"
             for s in SOURCES]
    return "; ".join(parts)


def format_report(by_exp: dict) -> str:
    n = {e: len(by_exp[e]) for e in EXPERIMENTS}
    metrics_general = [("PR-AUC", ("pr_auc",)), ("AUROC", ("roc_auc",)),
                       ("F1 @ own threshold", ("at_own_thr", "f1")),
                       ("F1 @ RAGTruth-val threshold", ("at_ragtruth_val_thr", "f1"))]
    metrics_clinical = [("PR-AUC", ("pr_auc",)), ("AUROC", ("roc_auc",)),
                        ("F1 @ own threshold", ("at_own_thr", "f1")),
                        ("F1 @ threshold (b), reporting only", ("at_clinical_thr", "f1"))]
    L = ["# RQ2: G, C and G+C", "",
         f"Runs: G {n['G']}, C {n['C']}, G+C {n['G+C']}. Cells: mean ± sample SD across seeds; the last two columns are "
         "differences of means. Own threshold = the one each run was selected with: G on RAGTruth validation, "
         "C and G+C on pooled PubMedQA + CovidQA clean validation. Headline clinical test set = clean.", "",
         "## Training data", "",
         "Data-volume confound: C and G+C train on more rows than G, so a difference between experiments mixes the "
         "kind of data with the amount.", ""] + training_table(by_exp) + [
         "", "## Runs", ""] + runs_table(by_exp) + [
         "", "## General test (does clinical training data cost general performance?)", ""]
    for title, metric in metrics_general:
        L += [f"### {title}", ""] + compare_table(by_exp, ["general"], GEN, metric, "source") + [""]
    for ts, ts_title in TEST_SETS:
        L += [f"## {ts_title}", ""]
        if ts == "test_full":
            L += [f"Rows sharing a question or an 8+ word passage with training data (removed from clean test): "
                  f"{removed_rows_note(by_exp)}. C and G+C trained on clinical train, so on these rows they may have "
                  "seen the passage; G did not.", ""]
        for title, metric in metrics_clinical:
            L += [f"### {title}", ""] + compare_table(by_exp, [ts], SOURCES, metric, "source") + [""]
    for field in MEDHALLU_FIELDS:
        L += [f"## MedHallu by {field} (clean test)", ""]
        keys = _subset_keys(by_exp, field)
        for title, metric in metrics_clinical[:1] + metrics_clinical[2:3]:
            L += [f"### {title}", ""] + compare_table(by_exp, [f"medhallu_by_{field}", "test_clean"], keys, metric,
                                                       field) + [""]
    L += [f"## CovidQA bootstrap 95% CIs (per run)", "",
          f"Percentile bootstrap over group_id clusters ({NAMES[BOOT_SOURCE]}), thresholds fixed; full-test breakdowns "
          "for MedHallu are in each run's metrics.json.", ""] + bootstrap_table(by_exp)
    L += ["", "## Notes", "",
          f"- With 3 seeds, differences below about {RESOLUTION:.2f} PR-AUC are not resolvable; they are reported, "
          "not interpreted. Seeds are not paired across experiments (different data), so no paired test is used.",
          "- Own threshold: G = RAGTruth validation (Phase 2); C and G+C = pooled PubMedQA + CovidQA clean validation "
          "(one F1 grid over both; MedHallu never used for selection or this threshold).",
          "- Threshold (b): reporting only. Tuned per source on clean clinical validation for every model, as in RQ1 "
          "(CovidQA pooled with PubMedQA; MedHallu on MedHallu validation).",
          "- RAGTruth-val threshold: tuned on each run's RAGTruth validation predictions (scored, never used for "
          "selection in C and G+C). For G it equals the own threshold.",
          "- Clean clinical test predictions are taken from the test_full predictions by row key (no row scored twice).",
          "- G's clinical numbers come from results/rq1 (scored locally from the Phase 2 checkpoints); C and G+C's come "
          "from the training run's own predictions.jsonl (scored on Kaggle).", ""]
    return "\n".join(L)


# ---- main -------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--rq1-root", default="results/rq1")
    ap.add_argument("--clean-test", default="data/clinical/test.jsonl")
    ap.add_argument("--out-root", default="results/rq2")
    ap.add_argument("--out-md", default="results/rq2.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--boot-seed", type=int, default=0)
    args = ap.parse_args(argv)

    runs = load_runs(args.runs_root)
    if not runs:
        print(f"no {args.runs_root}/*/config_snapshot.json found", file=sys.stderr)
        return 1
    for r in runs:                      # every check before anything is written
        check_reportable(r, args.rq1_root)
    clean_rows = read_jsonl(args.clean_test)
    cache = {}
    records = []
    for r in runs:
        rec = build_record(r, args.rq1_root, clean_rows, cache, args.n_boot, args.boot_seed)
        records.append(rec)
        print(f"  {r['name']}: {rec['experiment']} seed {rec['seed']}")
    by_exp = {e: [rec for rec in records if rec["experiment"] == e] for e in EXPERIMENTS}
    text = format_report(by_exp)            # may raise (e.g. runs tested on different rows): nothing written yet
    for rec in records:
        os.makedirs(os.path.join(args.out_root, rec["run"]), exist_ok=True)
        with open(os.path.join(args.out_root, rec["run"], "metrics.json"), "w") as f:
            json.dump(rec, f, indent=2)
        with open(os.path.join(args.out_root, f"{rec['run']}.json"), "w") as f:
            json.dump(summary(rec), f, indent=2)
    os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
    with open(args.out_md, "w") as f:
        f.write(text)
    print(text)
    print(f"wrote {args.out_md} and {len(records)} run file(s) under {args.out_root}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
