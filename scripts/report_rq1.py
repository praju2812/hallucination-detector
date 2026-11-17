"""Phase 4 (RQ1): metrics and tables from the scored G predictions.

Run (after scripts/score_rq1.py for every seed):
    python -m scripts.report_rq1

Reads, per seed:
    results/rq1/<run>/predictions.jsonl, scoring.json   (scripts/score_rq1.py)
    runs/<run>/metrics.json                              (Phase 2: threshold (a) and RAGTruth test metrics)
    data/clinical/test.jsonl                             (clean test rows; their predictions come from test_full)
Writes, per seed:
    results/rq1/<run>/metrics.json   full detail
    results/rq1/<run>.json           M3 summary read by scripts/mvp_check.py: {"seed", "test": {source: {...}}}
and results/rq1_G.md.

Headline = test_full (overlap removal was against clinical train only, so G has no leakage on
full); clean test reported beside it. All metric logic lives in src/rq1.py.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.summarize_runs import fmt_spread
from src.metrics import thresholded_metrics
from src.rq1 import (
    cluster_bootstrap_ci,
    derive_clean_predictions,
    spread,
    subset_metrics,
    tune_clinical_thresholds,
)
from src.train import get_git_dirty, get_git_sha
from utils.io import read_jsonl

SOURCES = ["ragbench_pubmedqa", "ragbench_covidqa", "medhallu"]
NAMES = {"ragbench_pubmedqa": "PubMedQA", "ragbench_covidqa": "CovidQA", "medhallu": "MedHallu"}
TEST_SETS = ["test_full", "test_clean"]
TEST_NAMES = {"test_full": "full", "test_clean": "clean"}
BOOT_SOURCE = "ragbench_covidqa"
MEDHALLU_FIELDS = ["difficulty", "category"]
SMALL_N = 30
DIFFICULTY_ORDER = ["easy", "medium", "hard"]


# ---- loading ------------------------------------------------------------------------

def _load_json(path):
    with open(path) as f:
        return json.load(f)


def find_scored_runs(rq1_root: str) -> list[str]:
    return sorted(os.path.basename(os.path.dirname(p))
                  for p in glob.glob(os.path.join(rq1_root, "run_seed*", "predictions.jsonl")))


def load_seed(run: str, rq1_root: str, runs_root: str, allow_partial: bool = False) -> dict:
    d = os.path.join(rq1_root, run)
    scoring = _load_json(os.path.join(d, "scoring.json"))
    if scoring.get("limit") is not None and not allow_partial:
        raise ValueError(f"{run}: scored with --limit {scoring['limit']} (smoke output); not reportable")
    preds = read_jsonl(os.path.join(d, "predictions.jsonl"))
    phase2 = _load_json(os.path.join(runs_root, run, "metrics.json"))
    for split, info in scoring["data_files"].items():
        n = sum(p["split"] == split for p in preds)
        if n != info["rows_in_file"] and not allow_partial:
            raise ValueError(f"{run}: {n} {split} predictions but {info['rows_in_file']} rows in {info['path']}")
    return {"run": run, "dir": d, "scoring": scoring, "preds": preds, "phase2": phase2}


def group_by(preds: list[dict], field: str) -> dict:
    out = {}
    for p in preds:
        out.setdefault(p.get(field), []).append(p)
    return out


def _lpg(preds):
    return (np.array([p["label"] for p in preds]), np.array([p["prob"] for p in preds]),
            np.array([str(p["group_id"]) for p in preds]))


# ---- per-seed metrics -------------------------------------------------------------------

def boot_metric_fns(thr_a: float, thr_b: float) -> dict:
    return {
        "pr_auc": lambda y, p: float(average_precision_score(y, p)),
        "roc_auc": lambda y, p: float(roc_auc_score(y, p)),
        "f1_at_ragtruth_thr": lambda y, p: thresholded_metrics(y, p, thr_a)["f1"],
        "f1_at_clinical_thr": lambda y, p: thresholded_metrics(y, p, thr_b)["f1"],
    }


def compute_seed_metrics(seed_data: dict, clean_rows: list[dict], n_boot: int = 2000, boot_seed: int = 0) -> dict:
    preds = seed_data["preds"]
    phase2 = seed_data["phase2"]
    thr_a = float(phase2["threshold"])
    val = [p for p in preds if p["split"] == "val"]
    full = [p for p in preds if p["split"] == "test_full"]
    sets = {"test_full": full, "test_clean": derive_clean_predictions(full, clean_rows)}
    thr_b = tune_clinical_thresholds(val)

    out = {
        "seed": seed_data["scoring"]["seed"],
        "experiment": seed_data["scoring"]["experiment"],
        "run": seed_data["run"],
        "thresholds": {"ragtruth_val": thr_a, "clinical_val": thr_b},
        "ragtruth_test": phase2["per_source"]["ragtruth"],
        "n_duplicate_rows_test_full": sum(not p["row_key"].endswith("#0") for p in full),
    }
    for name, ps in sets.items():
        by_src = group_by(ps, "source")
        out[name] = {s: subset_metrics(by_src[s], thr_a, thr_b[s]["threshold"]) for s in SOURCES if s in by_src}
        mh = by_src.get("medhallu", [])
        for field in MEDHALLU_FIELDS:
            out.setdefault(f"medhallu_by_{field}", {})[name] = {
                str(k): subset_metrics(v, thr_a, thr_b["medhallu"]["threshold"])
                for k, v in sorted(group_by(mh, field).items(), key=lambda kv: str(kv[0]))}
        if BOOT_SOURCE in by_src:
            y, p, g = _lpg(by_src[BOOT_SOURCE])
            fns = boot_metric_fns(thr_a, thr_b[BOOT_SOURCE]["threshold"])
            out.setdefault("covidqa_bootstrap", {})[name] = {
                "n_boot": n_boot, "seed": boot_seed, "n_groups": int(len(set(g))),
                "ci": cluster_bootstrap_ci(y, p, g, fns, n_boot=n_boot, seed=boot_seed)}
    out["scoring"] = seed_data["scoring"]
    out["report_git_sha"] = get_git_sha()
    out["report_git_dirty"] = get_git_dirty()
    return out


def build_m3_summary(metrics: dict) -> dict:
    """Thin per-seed file in the shape mvp_check.py M3 reads: top-level "seed" and "test": {source: {...}}.
    "test" = test_full (headline); F1 at threshold (a). No top-level "per_source" key (mvp_check would read it instead)."""
    def slim(block):
        return {s: {"pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"], "f1": m["at_ragtruth_thr"]["f1"],
                    "threshold": m["at_ragtruth_thr"]["threshold"], "prevalence": m["prevalence"],
                    "n": m["n"], "n_pos": m["n_pos"]} for s, m in block.items()}
    return {
        "seed": metrics["seed"],
        "experiment": metrics["experiment"],
        "run": metrics["run"],
        "test": slim(metrics["test_full"]),
        "test_clean": slim(metrics["test_clean"]),
        "details": os.path.join("results", "rq1", metrics["run"], "metrics.json"),
    }


# ---- formatting ---------------------------------------------------------------------------

def _get(d, path):
    for k in path:
        d = d[k]
    return d


def agg_cell(values: list) -> str:
    if any(v is None for v in values):
        k = sum(v is None for v in values)
        return f"undefined in {k} of {len(values)} seeds"
    return fmt_spread(spread(values))


def prev_cell(m: dict) -> str:
    return f"{100 * m['prevalence']:.1f}% ({m['n_pos']} / {m['n']})"


def _f(v, nd=4):
    return "n/a" if v is None else f"{v:.{nd}f}"


def metrics_table(all_m: list[dict], block: list, keys: list[str], label: str, flag_small: bool = False) -> list[str]:
    cols = [("PR-AUC", ("pr_auc",)), ("AUROC", ("roc_auc",)),
            ("P @ (a)", ("at_ragtruth_thr", "precision")), ("R @ (a)", ("at_ragtruth_thr", "recall")),
            ("F1 @ (a)", ("at_ragtruth_thr", "f1")), ("F1 @ (b)", ("at_clinical_thr", "f1"))]
    if flag_small:
        cols += [("faithful prec @ (a)", ("at_ragtruth_thr", "faithful_precision")),
                 ("faithful prec @ (b)", ("at_clinical_thr", "faithful_precision"))]
    head = f"| {label} | prevalence (n_pos / n) | " + " | ".join(c for c, _ in cols) + " |"
    lines = [head, "|" + "---|" * (len(cols) + 2)]
    for k in keys:
        ms = [_get(m, block)[k] for m in all_m]
        name = NAMES.get(k, k)
        if flag_small and ms[0]["n"] < SMALL_N:
            name += f" (n < {SMALL_N})"
        cells = [agg_cell([_get(m, path) for m in ms]) for _, path in cols]
        lines.append(f"| {name} | {prev_cell(ms[0])} | " + " | ".join(cells) + " |")
    return lines


def runs_table(all_m: list[dict]) -> list[str]:
    lines = ["| run folder | seed | scoring git sha | dirty | checkpoint | device | precision | thr (a) | "
             "thr (b) PubMedQA | thr (b) CovidQA (pooled) | thr (b) MedHallu | parity max / mean abs diff | "
             "parity AUROC Phase 2 / MPS | parity crossings |",
             "|" + "---|" * 14]
    for m in all_m:
        sc, tb = m["scoring"], m["thresholds"]["clinical_val"]
        par = sc.get("parity") or {}
        prec = sc.get("precision") or {}
        lines.append(
            f"| {m['run']} | {m['seed']} | {sc['git_sha']} | {sc['git_dirty']} | {sc['checkpoint']} | "
            f"{sc['device']} ({sc['device_name']}) | {prec.get('logits_dtype')}, autocast={prec.get('autocast')} | "
            f"{m['thresholds']['ragtruth_val']:.2f} | "
            + " | ".join(f"{tb[s]['threshold']:.2f}" for s in SOURCES)
            + (f" | {par['max_abs_diff']:.2e} / {par['mean_abs_diff']:.2e} | {_f(par['auroc_phase2'])} / "
               f"{_f(par['auroc_new'])} | {par['crossings']} of {par['n']} ({'PASS' if par['passed'] else 'FAIL'}) |"
               if par else " | not run | not run | not run |"))
    return lines


def drop_table(all_m: list[dict]) -> list[str]:
    lines = ["| test set | source | RAGTruth test prevalence | clinical prevalence | RAGTruth PR-AUC | clinical PR-AUC | "
             "PR-AUC drop | RAGTruth AUROC | clinical AUROC | AUROC drop |", "|" + "---|" * 10]
    rt0 = all_m[0]["ragtruth_test"]
    rt_prev = f"{100 * rt0['n_pos'] / rt0['n']:.1f}% ({rt0['n_pos']} / {rt0['n']})"
    for ts in TEST_SETS:
        for s in SOURCES:
            rt_pr = [m["ragtruth_test"]["pr_auc"] for m in all_m]
            rt_au = [m["ragtruth_test"]["roc_auc"] for m in all_m]
            cl_pr = [m[ts][s]["pr_auc"] for m in all_m]
            cl_au = [m[ts][s]["roc_auc"] for m in all_m]
            d_pr = [a - b for a, b in zip(rt_pr, cl_pr)]          # paired by seed
            d_au = [a - b for a, b in zip(rt_au, cl_au)]
            lines.append(f"| {TEST_NAMES[ts]} | {NAMES[s]} | {rt_prev} | {prev_cell(all_m[0][ts][s])} | "
                         f"{agg_cell(rt_pr)} | {agg_cell(cl_pr)} | {agg_cell(d_pr)} | "
                         f"{agg_cell(rt_au)} | {agg_cell(cl_au)} | {agg_cell(d_au)} |")
    return lines


def _ci(r):
    if r["lo"] is None:
        return f"{r['point']:.4f} [undefined]"
    return f"{r['point']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}]"


def bootstrap_table(all_m: list[dict]) -> list[str]:
    lines = ["| test set | seed | PR-AUC [95% CI] | AUROC [95% CI] | F1 @ (a) [95% CI] | F1 @ (b) [95% CI] | "
             "groups | valid resamples |", "|" + "---|" * 8]
    for ts in TEST_SETS:
        for m in all_m:
            b = m["covidqa_bootstrap"][ts]
            ci = b["ci"]
            lines.append(f"| {TEST_NAMES[ts]} | {m['seed']} | {_ci(ci['pr_auc'])} | {_ci(ci['roc_auc'])} | "
                         f"{_ci(ci['f1_at_ragtruth_thr'])} | {_ci(ci['f1_at_clinical_thr'])} | {b['n_groups']} | "
                         f"{ci['pr_auc']['n_valid']} of {b['n_boot']} |")
    return lines


def selection_table(all_m: list[dict]) -> list[str]:
    lines = ["| source | seed | prevalence full | prevalence clean | PR-AUC full | PR-AUC clean | PR-AUC diff (clean - full) | "
             "AUROC full | AUROC clean | AUROC diff (clean - full) |", "|" + "---|" * 10]
    for s in SOURCES:
        f0, c0 = all_m[0]["test_full"][s], all_m[0]["test_clean"][s]
        pf, pc = prev_cell(f0), prev_cell(c0)
        for m in all_m:
            f, c = m["test_full"][s], m["test_clean"][s]
            lines.append(f"| {NAMES[s]} | {m['seed']} | {pf} | {pc} | {_f(f['pr_auc'])} | {_f(c['pr_auc'])} | "
                         f"{c['pr_auc'] - f['pr_auc']:+.4f} | {_f(f['roc_auc'])} | {_f(c['roc_auc'])} | "
                         f"{c['roc_auc'] - f['roc_auc']:+.4f} |")
        col = lambda ts, k: [m[ts][s][k] for m in all_m]
        dpr = [c - f for c, f in zip(col("test_clean", "pr_auc"), col("test_full", "pr_auc"))]
        dau = [c - f for c, f in zip(col("test_clean", "roc_auc"), col("test_full", "roc_auc"))]
        lines.append(f"| {NAMES[s]} | mean | {pf} | {pc} | {agg_cell(col('test_full', 'pr_auc'))} | "
                     f"{agg_cell(col('test_clean', 'pr_auc'))} | {agg_cell(dpr)} | {agg_cell(col('test_full', 'roc_auc'))} | "
                     f"{agg_cell(col('test_clean', 'roc_auc'))} | {agg_cell(dau)} |")
    return lines


def _order(keys, field):
    if field == "difficulty":
        return [k for k in DIFFICULTY_ORDER if k in keys] + sorted(k for k in keys if k not in DIFFICULTY_ORDER)
    return sorted(keys)


def format_report(all_m: list[dict]) -> str:
    tb = all_m[0]["thresholds"]["clinical_val"]
    n_dup = all_m[0]["n_duplicate_rows_test_full"]
    L = ["# RQ1: G zero-shot on clinical data", "",
         f"{len(all_m)} G seed(s). Cells: mean ± sample SD [min, max] across seeds. "
         "Threshold (a) = the seed's RAGTruth-validation threshold (Phase 2, primary, zero-shot). "
         "Threshold (b) = tuned on clean clinical validation, same grid as training (secondary). "
         "Headline test set = test_full; clean test reported beside it.", "",
         "## Runs", ""] + runs_table(all_m) + [
         "", "## Headline: full clinical test", ""] + metrics_table(all_m, ["test_full"], SOURCES, "source") + [
         "", "## Clean clinical test", ""] + metrics_table(all_m, ["test_clean"], SOURCES, "source") + [
         "", "## Drop from RAGTruth test (Phase 2) to each clinical source", "",
         "Drop = RAGTruth minus clinical, paired by seed, then mean ± SD. PR-AUC depends on prevalence (compare it with "
         "the prevalence columns); AUROC does not.", ""] + drop_table(all_m)
    for field in MEDHALLU_FIELDS:
        keys = _order(list(all_m[0][f"medhallu_by_{field}"]["test_full"]), field)
        L += ["", f"## MedHallu by {field} (full test)", "",
              f"Cells with n < {SMALL_N} are flagged. Faithful-class precision = TN / (TN + FN). "
              "Clean-test breakdowns are in each seed's metrics.json.", ""]
        L += metrics_table(all_m, ["test_full"], ["medhallu"], "subset", flag_small=True)
        L += metrics_table(all_m, [f"medhallu_by_{field}", "test_full"], keys, field, flag_small=True)[2:]
    b0 = all_m[0]["covidqa_bootstrap"]["test_full"]
    L += ["", "## CovidQA bootstrap 95% CIs (per seed)", "",
          f"Percentile bootstrap over group_id clusters, B = {b0['n_boot']}, RNG seed {b0['seed']}, thresholds fixed. "
          "Resamples with one class only are skipped.", ""] + bootstrap_table(all_m)
    L += ["", "## Selection check: full vs clean test", "",
          "Clean test drops rows that share a question or an 8+ word passage with clinical TRAIN (hard rule 6). "
          "G never saw clinical train, so a difference here reflects which rows were removed, not leakage.", ""
          ] + selection_table(all_m)
    par = [m["scoring"].get("parity") for m in all_m]
    L += ["", "## Notes", "",
          f"- test_full has {n_dup} extra copies of identical CovidQA rows (same group_id, answer and context, label 0). "
          "All copies are kept and scored (row key = sha1(source|group_id|answer) + occurrence number).",
          "- Threshold (b) tuning rows: " + "; ".join(
              f"{NAMES[s]} on {', '.join(NAMES[t] for t in tb[s]['tuned_on'])} ({tb[s]['n_pos']} / {tb[s]['n']} positives)"
              for s in SOURCES) + ". Values per seed are in the Runs table.",
          "- MedHallu validation is pqa_artificial; MedHallu test is pqa_labeled.",
          "- Clean-test predictions are taken from the test_full predictions by row key (no row scored twice).",
          "- Parity (MPS re-score vs Phase 2's stored probabilities on RAGTruth validation rows): "
          + ("; ".join(f"seed {m['seed']}: max {p['max_abs_diff']:.2e}, mean {p['mean_abs_diff']:.2e}, "
                       f"crossings {p['crossings']} of {p['n']}, {'PASS' if p['passed'] else 'FAIL'}"
                       for m, p in zip(all_m, par) if p) or "not run") + ".",
          f"- Scoring: torch {all_m[0]['scoring']['torch_version']}, transformers "
          f"{all_m[0]['scoring']['transformers_version']}; Phase 2 trained on torch "
          "2.11.0+cu130 (T4).", ""]
    return "\n".join(L)


# ---- main -------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rq1-root", default="results/rq1")
    ap.add_argument("--runs-root", default="runs", help="Phase 2 run folders (metrics.json)")
    ap.add_argument("--data-dir", default="data/clinical")
    ap.add_argument("--out-md", default="results/rq1_G.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--allow-partial", action="store_true", help="accept --limit (smoke) predictions; never for reporting")
    args = ap.parse_args(argv)

    runs = find_scored_runs(args.rq1_root)
    if not runs:
        print(f"no {args.rq1_root}/run_seed*/predictions.jsonl found", file=sys.stderr)
        return 1
    clean_rows = read_jsonl(os.path.join(args.data_dir, "test.jsonl"))
    all_m = []
    for run in runs:
        sd = load_seed(run, args.rq1_root, args.runs_root, args.allow_partial)
        m = compute_seed_metrics(sd, clean_rows, args.n_boot, args.boot_seed)
        with open(os.path.join(sd["dir"], "metrics.json"), "w") as f:
            json.dump(m, f, indent=2)
        with open(os.path.join(args.rq1_root, f"{run}.json"), "w") as f:
            json.dump(build_m3_summary(m), f, indent=2)
        print(f"  {run}: wrote metrics.json and {run}.json")
        all_m.append(m)
    text = format_report(all_m)
    os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
    with open(args.out_md, "w") as f:
        f.write(text)
    print(text)
    print(f"wrote {args.out_md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
