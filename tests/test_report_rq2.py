"""Phase 5 (RQ2) report: G, C and G+C side by side (plumbing; the metric logic is src/rq1.py and src/metrics.py).

Toy end to end: two seeds per experiment, written in the shapes the real pipeline writes:
    G:      runs/<run>/{config_snapshot,metrics}.json + predictions.jsonl (Phase 2, splits val/test)
            results/rq1/<run>/metrics.json (scripts/report_rq1.py output; clinical val + test_full)
    C, G+C: runs/<run>/... with splits val (clinical val then general val), test, test_full

Run: python -m pytest tests/test_report_rq2.py
"""

import copy
import json
import os

import numpy as np
import pytest
import yaml

from scripts.report_rq1 import compute_seed_metrics
from scripts.report_rq2 import attach_row_info, main as report_main
from src.metrics import best_threshold, per_source_breakdown, ranking_metrics, thresholded_metrics
from src.rq1 import assign_row_keys
from src.train import build_metrics_summary, resolved_training_values
from utils.io import write_jsonl

CLIN = ["ragbench_pubmedqa", "ragbench_covidqa", "medhallu"]
GEN = ["ragtruth", "halueval"]
SELECT = ["ragbench_pubmedqa", "ragbench_covidqa"]
SEEDS = (42, 43)


def _load(path):
    with open(path) as f:
        return yaml.safe_load(f)


BASE, C_CFG, GC_CFG = _load("configs/base.yaml"), _load("configs/c.yaml"), _load("configs/gc.yaml")


# ---- toy data ---------------------------------------------------------------------------------

def _clin_rows(split, n):
    rows = []
    for s in CLIN:
        for i in range(n):
            y = int(i % 4 == 0) if s != "medhallu" else i % 2
            rows.append({"query": f"q{i}", "context": "c", "answer": f"{split}-{s}-{i}", "label": y, "source": s,
                         "task": "QA", "group_id": f"{s}-{split}-g{i // 2 if s == 'medhallu' else i}",
                         "difficulty": ["easy", "medium", "hard"][(i // 2) % 3] if s == "medhallu" else None,
                         "category": ["Incomplete Information", "Misinterpretation of #Question#"][(i // 2) % 2]
                         if s == "medhallu" else None})
    return rows


def _gen_rows(split, n):
    return [{"query": f"q{i}", "context": "c", "answer": f"{split}-{s}-{i}", "label": int(i % 3 == 0) if s == "ragtruth" else i % 2,
             "source": s, "task": "QA", "group_id": f"{s}-{split}-g{i}"} for s in GEN for i in range(n)]


def _preds(rng, split, rows, offset=0):
    return [{"split": split, "idx": offset + i, "source": r["source"], "task": r["task"], "group_id": r["group_id"],
             "label": r["label"], "prob": float(np.clip(rng.normal(0.35 + 0.3 * r["label"], 0.15), 0.001, 0.999))}
            for i, r in enumerate(rows)]


def _arrays(preds):
    return (np.array([p["label"] for p in preds], dtype=float), np.array([p["prob"] for p in preds]),
            [p["source"] for p in preds])


def _set_metrics(preds, thr):
    y, p, s = _arrays(preds)
    overall = ranking_metrics(y, p)
    overall.update(thresholded_metrics(y, p, thr))
    return {"overall": overall, "per_source": per_source_breakdown(y, p, s, thr)}


def _thr(preds, sources):
    y, p, _ = _arrays([q for q in preds if q["source"] in sources])
    return best_threshold(y, p)


@pytest.fixture
def toy(tmp_path):
    d = tmp_path / "data"
    files = {
        "gen_train": d / "general" / "train.jsonl", "gen_val": d / "general" / "val.jsonl",
        "gen_test": d / "general" / "test.jsonl", "clin_train": d / "clinical" / "train.jsonl.gz",
        "clin_val": d / "clinical" / "val.jsonl", "clin_full": d / "clinical" / "test_full.jsonl",
        "clin_clean": d / "clinical" / "test.jsonl",
    }
    rows = {"gen_train": _gen_rows("train", 30), "gen_val": _gen_rows("val", 24), "gen_test": _gen_rows("test", 30),
            "clin_train": _clin_rows("train", 20), "clin_val": _clin_rows("val", 40), "clin_full": _clin_rows("test", 48)}
    rows["clin_clean"] = [r for i, r in enumerate(rows["clin_full"]) if i % 3 != 0]
    for k, p in files.items():
        write_jsonl(rows[k], str(p))
    P = {k: str(v) for k, v in files.items()}
    runs, rq1 = tmp_path / "runs", tmp_path / "rq1"
    rng = np.random.default_rng(0)

    def write_run(name, cfg, preds, metrics, resolved=True):
        rd = runs / name
        os.makedirs(rd)
        snap = {"config": cfg, "seed": cfg["train"]["seed"], "experiment": cfg["experiment"], "git_sha": "abc1234",
                "git_dirty": False, "input_format": "Question: {query} Context: {context}", "truncation": "fit_first",
                "torch_version": "t", "transformers_version": "tf", "device_name": "Tesla T4"}
        if resolved:
            snap["resolved"] = resolved_training_values(cfg, 100, 13)
        (rd / "config_snapshot.json").write_text(json.dumps(snap))
        (rd / "metrics.json").write_text(json.dumps(metrics))
        write_jsonl(preds, str(rd / "predictions.jsonl"))

    for seed in SEEDS:
        # G: Phase 2 run + RQ1 output
        g = copy.deepcopy(BASE)
        g["train"]["seed"] = seed
        g["data"].update(train_path=P["gen_train"], val_path=P["gen_val"], test_path=P["gen_test"])
        gv, gt = _preds(rng, "val", rows["gen_val"]), _preds(rng, "test", rows["gen_test"])
        thr = _thr(gv, ["ragtruth"])
        sets = {"test": _set_metrics(gt, thr)}
        gm = build_metrics_summary(thr, ["ragtruth"], sets, sets["test"]["per_source"], g)
        name = f"run_seed{seed}_g"
        write_run(name, g, gv + gt, gm, resolved=False)          # the locked G runs predate "resolved"
        cv, cf = _preds(rng, "val", rows["clin_val"]), _preds(rng, "test_full", rows["clin_full"])
        for split_preds, split_rows in ((cv, rows["clin_val"]), (cf, rows["clin_full"])):
            for p, r, k in zip(split_preds, split_rows, assign_row_keys(split_rows)):
                p.update(row_key=k, difficulty=r.get("difficulty"), category=r.get("category"))
        rq1_m = compute_seed_metrics({"run": name, "preds": cv + cf, "phase2": gm,
                                      "scoring": {"seed": seed, "experiment": "G", "truncation": "fit_first"}},
                                     rows["clin_clean"], n_boot=30,
                                     report_git={"report_git_sha": "abc1234", "report_git_dirty": False})
        os.makedirs(rq1 / name)
        (rq1 / name / "metrics.json").write_text(json.dumps(rq1_m))

        # C and G+C
        for exp, base in (("C", C_CFG), ("G+C", GC_CFG)):
            cfg = copy.deepcopy(base)
            cfg["train"]["seed"] = seed
            cfg["data"].update(train_path=P["clin_train"] if exp == "C" else [P["gen_train"], P["clin_train"]],
                               val_path=[P["clin_val"], P["gen_val"]],
                               test_sets={"test": P["gen_test"], "test_full": P["clin_full"]})
            val = _preds(rng, "val", rows["clin_val"]) + _preds(rng, "val", rows["gen_val"], offset=len(rows["clin_val"]))
            t, tf = _preds(rng, "test", rows["gen_test"]), _preds(rng, "test_full", rows["clin_full"])
            thr = _thr(val, SELECT)
            sets = {"test": _set_metrics(t, thr), "test_full": _set_metrics(tf, thr)}
            merged = {**sets["test"]["per_source"], **sets["test_full"]["per_source"]}
            write_run(f"run_seed{seed}_{'c' if exp == 'C' else 'gc'}", cfg, val + t + tf,
                      build_metrics_summary(thr, SELECT, sets, merged, cfg))
    out_root, md = tmp_path / "rq2", tmp_path / "rq2.md"
    args = ["--runs-root", str(runs), "--rq1-root", str(rq1), "--clean-test", P["clin_clean"],
            "--out-root", str(out_root), "--out-md", str(md), "--n-boot", "30"]
    return {"runs": runs, "rq1": rq1, "out_root": out_root, "md": md, "args": args, "rows": rows, "P": P}


def _record(toy, run):
    return json.loads((toy["out_root"] / run / "metrics.json").read_text())


def _run_preds(toy, run):
    with open(toy["runs"] / run / "predictions.jsonl") as f:
        return [json.loads(l) for l in f]


# ---- end to end -------------------------------------------------------------------------------

def test_report_end_to_end(toy):
    assert report_main(toy["args"]) == 0
    n_clean = len(toy["rows"]["clin_clean"]) // 3
    for seed in SEEDS:
        for suffix, exp in (("g", "G"), ("c", "C"), ("gc", "G+C")):
            run = f"run_seed{seed}_{suffix}"
            r = _record(toy, run)
            assert r["experiment"] == exp and r["seed"] == seed
            assert set(r["test_clean"]) == set(CLIN) and set(r["test_full"]) == set(CLIN)
            assert r["test_clean"]["ragbench_pubmedqa"]["n"] == n_clean          # clean derived from full
            assert r["test_full"]["ragbench_pubmedqa"]["n"] == 48
            assert "at_own_thr" in r["test_clean"]["medhallu"]
            assert "at_ragtruth_thr" not in json.dumps(r)                        # relabelled everywhere
            assert set(r["general"]) == set(GEN)
            assert set(r["medhallu_by_difficulty"]["test_clean"]) == {"easy", "medium", "hard"}
            assert set(r["covidqa_bootstrap"]) == {"test_full", "test_clean"}
            summ = json.loads((toy["out_root"] / f"{run}.json").read_text())
            assert "per_source" not in summ and set(summ["test"]) == set(CLIN)


def test_own_threshold_and_tuned_on(toy):
    report_main(toy["args"])
    for seed in SEEDS:
        c = _record(toy, f"run_seed{seed}_c")
        m = json.loads((toy["runs"] / f"run_seed{seed}_c" / "metrics.json").read_text())
        assert c["thresholds"]["own"] == {"threshold": m["threshold"], "tuned_on": SELECT}
        assert c["test_clean"]["ragbench_covidqa"]["at_own_thr"]["threshold"] == m["threshold"]
        assert c["thresholds"]["clinical_val"]["ragbench_covidqa"]["tuned_on"] == sorted(SELECT)
        g = _record(toy, f"run_seed{seed}_g")
        assert g["thresholds"]["own"]["tuned_on"] == ["ragtruth"]
        assert g["thresholds"]["ragtruth_val"] == pytest.approx(g["thresholds"]["own"]["threshold"])


def test_general_block_both_thresholds(toy):
    report_main(toy["args"])
    for seed in SEEDS:
        run = f"run_seed{seed}_gc"
        r, preds = _record(toy, run), _run_preds(toy, run)
        rt_val = [p for p in preds if p["split"] == "val" and p["source"] == "ragtruth"]
        thr_rt = _thr(rt_val, ["ragtruth"])
        assert r["thresholds"]["ragtruth_val"] == pytest.approx(thr_rt)
        rt = r["general"]["ragtruth"]
        y, p, _ = _arrays([q for q in preds if q["split"] == "test" and q["source"] == "ragtruth"])
        assert rt["pr_auc"] == pytest.approx(ranking_metrics(y, p)["pr_auc"])
        assert rt["at_ragtruth_val_thr"]["f1"] == pytest.approx(thresholded_metrics(y, p, thr_rt)["f1"])
        assert rt["at_own_thr"]["threshold"] == r["thresholds"]["own"]["threshold"]
        assert (rt["n"], rt["n_pos"]) == (30, 10)


def test_training_data_sizes(toy):
    report_main(toy["args"])
    n_gen, n_clin = len(toy["rows"]["gen_train"]), len(toy["rows"]["clin_train"])
    assert _record(toy, "run_seed42_g")["training_data"]["n"] == n_gen
    assert _record(toy, "run_seed42_c")["training_data"]["n"] == n_clin
    gc = _record(toy, "run_seed42_gc")["training_data"]
    assert gc["n"] == n_gen + n_clin
    assert set(gc["per_source"]) == set(GEN) | set(CLIN)
    assert gc["n_pos"] == sum(r["label"] for r in toy["rows"]["gen_train"] + toy["rows"]["clin_train"])


def test_markdown_layout(toy):
    report_main(toy["args"])
    text = toy["md"].read_text()
    for heading in ("## Training data", "## Runs", "## General test", "## Clean clinical test (headline)",
                    "## Full clinical test (contamination contrast)", "## MedHallu by difficulty (clean test)",
                    "## MedHallu by category (clean test)", "## CovidQA bootstrap", "## Notes"):
        assert heading in text, heading
    assert text.index("## Clean clinical test (headline)") < text.index("## Full clinical test (contamination contrast)")
    assert "| G | C | G+C | C - G | G+C - G |" in text
    assert "F1 @ RAGTruth-val threshold" in text
    assert "reporting only" in text                          # threshold (b) label (decision D3)
    assert "0.05" in text and "not resolvable" in text       # decision D4
    assert "—" not in text                                   # no em dashes in reports
    train_sec = text.split("## Training data")[1].split("## Runs")[0]
    assert f"| G+C |" in train_sec and str(len(toy["rows"]["gen_train"]) + len(toy["rows"]["clin_train"])) in train_sec


# ---- refusals ---------------------------------------------------------------------------------

def test_refuses_smoke_runs(toy):
    snap_p = toy["runs"] / "run_seed42_c" / "config_snapshot.json"
    snap = json.loads(snap_p.read_text())
    snap["config"]["data"]["train_path"] = "/kaggle/working/smoke/clinical_tiny.jsonl"
    snap_p.write_text(json.dumps(snap))
    with pytest.raises(ValueError, match="tiny"):
        report_main(toy["args"])


def test_refuses_g_without_rq1_metrics(toy):
    os.remove(toy["rq1"] / "run_seed43_g" / "metrics.json")
    with pytest.raises(ValueError, match="score_rq1"):
        report_main(toy["args"])


def test_refuses_wrong_truncation(toy):
    snap_p = toy["runs"] / "run_seed43_gc" / "config_snapshot.json"
    snap = json.loads(snap_p.read_text())
    snap["truncation"] = "legacy"
    snap_p.write_text(json.dumps(snap))
    with pytest.raises(ValueError, match="truncation"):
        report_main(toy["args"])


def test_attach_row_info_checks_labels_and_counts():
    rows = _clin_rows("test", 4)
    preds = _preds(np.random.default_rng(1), "test_full", rows)
    out = attach_row_info(preds, {"test_full": rows})
    assert [p["row_key"] for p in out] == assign_row_keys(rows)
    assert out[8]["difficulty"] == rows[8]["difficulty"]
    assert "row_key" not in preds[0]                         # input not modified
    bad = copy.deepcopy(preds)
    bad[0]["label"] = 1 - bad[0]["label"]
    with pytest.raises(ValueError):
        attach_row_info(bad, {"test_full": rows})
    with pytest.raises(ValueError):
        attach_row_info(preds[:-1], {"test_full": rows})
