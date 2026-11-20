"""Phase 4 plumbing: parity gate (scripts/score_rq1.py) and outputs of scripts/report_rq1.py.

The parity and M3-shape tests run now. The end-to-end report test also calls the src/rq1.py templates,
so it passes only once those bodies are written.

Run: python -m pytest tests/test_rq1_plumbing.py
"""

import json
import os

import numpy as np
import pytest

import scripts.mvp_check as mvp                 # module import: pytest must not collect mvp.test_block as a test
from scripts.report_rq1 import all_flagged_f1, build_m3_summary, main as report_main, runs_table
from scripts.score_rq1 import parity_report


# ---- parity gate ------------------------------------------------------------------

def test_parity_identical_passes():
    p = np.array([0.1, 0.8, 0.3, 0.6])
    y = np.array([0, 1, 0, 1])
    r = parity_report(p, p.copy(), y, threshold=0.5)
    assert r["passed"] and r["max_abs_diff"] == 0.0 and r["crossings"] == 0
    assert r["auroc_phase2"] == r["auroc_new"] == 1.0


def test_parity_small_noise_passes_large_diff_fails():
    rng = np.random.default_rng(0)
    y = (rng.random(200) < 0.3).astype(int)
    a = np.clip(rng.normal(0.3 + 0.4 * y, 0.1), 0, 1)
    ok = parity_report(a, a + 1e-4, y, 0.5)
    assert ok["passed"]
    bad = parity_report(a, a + 0.02, y, 0.5)
    assert not bad["checks"]["max_abs_diff_lt_0.01"] and not bad["passed"]


def test_parity_crossings_over_one_percent_fail():
    a = np.full(100, 0.49)
    b = a.copy()
    b[:2] = 0.505                                   # 2 of 100 rows cross 0.5 (the max-diff check also fails; not tested here)
    y = np.array([1, 0] * 50)
    r = parity_report(a, b, y, 0.5)
    assert r["crossings"] == 2 and not r["checks"]["crossings_le_1pct"]
    b2 = a.copy()
    b2[0] = 0.505
    assert parity_report(a, b2, y, 0.5)["checks"]["crossings_le_1pct"]   # 1 of 100 is allowed


def test_parity_auroc_must_match_to_3dp():
    y = np.array([0, 1, 0, 1, 0, 1])
    a = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60])
    b = np.array([0.10, 0.20, 0.30, 0.40, 0.605, 0.60])   # swaps the order of the last pair: AUROC changes
    r = parity_report(a, b, y, 0.9)
    assert not r["checks"]["auroc_equal_3dp"]


def test_parity_length_mismatch_raises():
    with pytest.raises(ValueError):
        parity_report([0.1, 0.2], [0.1], [0, 1], 0.5)


# ---- M3 summary shape ---------------------------------------------------------------

def _block(src_names):
    m = {"pr_auc": 0.5, "roc_auc": 0.7, "prevalence": 0.2, "n": 10, "n_pos": 2,
         "at_ragtruth_thr": {"threshold": 0.1, "precision": 0.5, "recall": 0.5, "f1": 0.5, "faithful_precision": 0.9},
         "at_clinical_thr": {"threshold": 0.3, "precision": 0.5, "recall": 0.5, "f1": 0.4, "faithful_precision": 0.9}}
    return {s: dict(m) for s in src_names}


def test_m3_summary_is_found_by_mvp_check(tmp_path):
    srcs = ["ragbench_pubmedqa", "ragbench_covidqa", "medhallu"]
    metrics = {"seed": 42, "experiment": "G", "truncation": "fit_first", "run": "run_seed42_x",
               "test_full": _block(srcs), "test_clean": _block(srcs)}
    summ = build_m3_summary(metrics)
    p = tmp_path / "run_seed42_x.json"
    p.write_text(json.dumps(summ))
    d = json.loads(p.read_text())
    assert d["seed"] == 42 and "per_source" not in d
    assert d["truncation"] == "fit_first"
    found = {mvp.clinical_key(k) for k in mvp.test_block(d)}
    assert found == set(mvp.CLINICAL_KEYS)
    assert d["test"]["ragbench_covidqa"]["f1"] == 0.5          # F1 at threshold (a)
    assert d["test"]["medhallu"]["pr_auc"] == 0.5


# ---- report formatting: all-flagged F1 baseline and grid-floor flag ------------------------

def test_all_flagged_f1_by_hand():
    # flag every row: P = p, R = 1, F1 = 2p / (1 + p)
    assert all_flagged_f1(0.25) == pytest.approx(0.4)
    assert all_flagged_f1(0.5) == pytest.approx(2 / 3)
    assert all_flagged_f1(1.0) == pytest.approx(1.0)
    assert all_flagged_f1(0.0) == 0.0


def _runs_m(thr_a, thr_b):
    return {"run": "run_seed42_x", "seed": 42, "truncation": "fit_first",
            "thresholds": {"ragtruth_val": thr_a,
                           "clinical_val": {s: {"threshold": t} for s, t in zip(SRCS, thr_b)}},
            "scoring": {"git_sha": "abc", "git_dirty": False, "checkpoint": "/x/best.pt", "device": "cpu",
                        "device_name": "cpu", "precision": {"logits_dtype": "torch.float32", "autocast": False},
                        "parity": None}}


def test_runs_table_flags_grid_floor_only():
    grid = np.linspace(0.05, 0.95, 19)              # the best_threshold grid: its floor is grid[0]
    row = runs_table([_runs_m(float(grid[0]), [0.1, float(grid[0]), 0.15])])[2]
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[2] == "fit_first"                  # truncation column
    thr_cells = cells[8:12]                         # thr (a), thr (b) PubMedQA, CovidQA, MedHallu
    assert thr_cells == ["0.05 (grid floor)", "0.10", "0.05 (grid floor)", "0.15"]


# ---- end to end on toy files (needs the src/rq1.py bodies) -----------------------------

SRCS = ["ragbench_pubmedqa", "ragbench_covidqa", "medhallu"]


def _toy_rows(rng, split, n_per_src):
    rows = []
    for s in SRCS:
        for i in range(n_per_src):
            y = int(i % 4 == 0) if s != "medhallu" else i % 2
            r = {"query": f"q{i}", "context": "c", "answer": f"{split}-{s}-{i}", "label": y, "source": s,
                 "task": "QA", "group_id": f"{s}-{split}-g{i // 2 if s == 'medhallu' else i}",
                 "difficulty": ["easy", "medium", "hard"][(i // 2) % 3] if s == "medhallu" else None,
                 "category": ["Incomplete Information", "Misinterpretation of #Question#"][(i // 2) % 2] if s == "medhallu" else None}
            rows.append(r)
    return rows


def _write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _make_toy(tmp_path, seeds=(42, 43)):
    from src.rq1 import assign_row_keys
    rng = np.random.default_rng(0)
    data = tmp_path / "clinical"
    val = _toy_rows(rng, "val", 40)
    full = _toy_rows(rng, "test", 48)
    clean = [r for i, r in enumerate(full) if i % 3 != 0]
    _write_jsonl(data / "val.jsonl", val)
    _write_jsonl(data / "test_full.jsonl", full)
    _write_jsonl(data / "test.jsonl", clean)
    rq1, runs = tmp_path / "rq1", tmp_path / "runs"
    for seed in seeds:
        run = f"run_seed{seed}_x"
        preds = []
        for split, rows in (("val", val), ("test_full", full)):
            for i, (r, k) in enumerate(zip(rows, assign_row_keys(rows))):
                prob = float(np.clip(rng.normal(0.35 + 0.3 * r["label"], 0.15), 0, 1))
                preds.append({"split": split, "idx": i, "source": r["source"], "task": "QA", "group_id": r["group_id"],
                              "label": r["label"], "prob": prob, "row_key": k,
                              "difficulty": r["difficulty"], "category": r["category"]})
        _write_jsonl(rq1 / run / "predictions.jsonl", preds)
        scoring = {"run": run, "seed": seed, "experiment": "G", "truncation": "fit_first", "checkpoint": "/x/best.pt", "git_sha": "abc",
                   "git_dirty": False, "device": "cpu", "device_name": "cpu",
                   "precision": {"param_dtype": "torch.float32", "logits_dtype": "torch.float32", "autocast": False},
                   "torch_version": "t", "transformers_version": "tf", "limit": None,
                   "data_files": {"val": {"path": "v", "rows_in_file": len(val)},
                                  "test_full": {"path": "f", "rows_in_file": len(full)}},
                   "parity": {"n": 100, "max_abs_diff": 1e-5, "mean_abs_diff": 1e-6, "auroc_phase2": 0.8,
                              "auroc_new": 0.8, "crossings": 0, "passed": True}}
        (rq1 / run / "scoring.json").write_text(json.dumps(scoring))
        os.makedirs(runs / run)
        (runs / run / "metrics.json").write_text(json.dumps({
            "threshold": 0.4, "per_source": {"ragtruth": {"pr_auc": 0.5, "roc_auc": 0.77, "f1": 0.46, "n": 1800, "n_pos": 364}}}))
    return data, rq1, runs


def test_report_end_to_end(tmp_path):
    data, rq1, runs = _make_toy(tmp_path)
    md = tmp_path / "rq1_G.md"
    rc = report_main(["--rq1-root", str(rq1), "--runs-root", str(runs), "--data-dir", str(data),
                      "--out-md", str(md), "--n-boot", "50"])
    assert rc == 0
    for seed in (42, 43):
        m = json.loads((rq1 / f"run_seed{seed}_x" / "metrics.json").read_text())
        assert set(m["test_full"]) == set(SRCS) and set(m["test_clean"]) == set(SRCS)
        assert m["test_full"]["ragbench_pubmedqa"]["n"] == 48
        assert m["test_clean"]["ragbench_pubmedqa"]["n"] == 32
        assert set(m["covidqa_bootstrap"]) == {"test_full", "test_clean"}
        assert set(m["covidqa_bootstrap"]["test_full"]["ci"]) == {"pr_auc", "roc_auc", "f1_at_ragtruth_thr", "f1_at_clinical_thr"}
        assert m["thresholds"]["ragtruth_val"] == 0.4
        assert m["thresholds"]["clinical_val"]["ragbench_covidqa"]["tuned_on"] == ["ragbench_covidqa", "ragbench_pubmedqa"]
        assert set(m["medhallu_by_difficulty"]["test_full"]) == {"easy", "medium", "hard"}
        assert m["truncation"] == "fit_first"
        summ = json.loads((rq1 / f"run_seed{seed}_x.json").read_text())
        assert summ["truncation"] == "fit_first"
        assert {mvp.clinical_key(k) for k in mvp.test_block(summ)} == set(mvp.CLINICAL_KEYS)
    text = md.read_text()
    for heading in ("## Headline: full clinical test", "## Clean clinical test", "## Drop from RAGTruth test",
                    "## MedHallu by difficulty", "## MedHallu by category", "## CovidQA bootstrap",
                    "## Selection check: full vs clean test", "## Notes"):
        assert heading in text
    assert "—" not in text                      # no em dashes in reports
    # the all-flagged F1 column is in the full and clean tables only
    assert text.count("| F1 if every row flagged |") == 2
    full_sec = text.split("## Headline: full clinical test")[1].split("## Clean clinical test")[0]
    pub_row = next(l for l in full_sec.splitlines() if l.startswith("| PubMedQA |"))
    assert pub_row.rstrip().endswith("| 0.4000 |")   # toy PubMedQA full: 12 / 48 positives, 2(0.25)/1.25 = 0.4


def test_report_refuses_smoke_predictions(tmp_path):
    data, rq1, runs = _make_toy(tmp_path, seeds=(42,))
    sc = json.loads((rq1 / "run_seed42_x" / "scoring.json").read_text())
    sc["limit"] = 100
    (rq1 / "run_seed42_x" / "scoring.json").write_text(json.dumps(sc))
    with pytest.raises(ValueError):
        report_main(["--rq1-root", str(rq1), "--runs-root", str(runs), "--data-dir", str(data),
                     "--out-md", str(tmp_path / "x.md"), "--n-boot", "10"])


def _set_scoring(rq1, run, **changes):
    p = rq1 / run / "scoring.json"
    sc = json.loads(p.read_text())
    for k, v in changes.items():
        if v is None:
            sc.pop(k, None)
        else:
            sc[k] = v
    p.write_text(json.dumps(sc))


def test_report_refuses_scoring_without_truncation(tmp_path):
    """A scoring.json from before truncation was recorded (the pre-20 Nov G runs) is not reportable."""
    data, rq1, runs = _make_toy(tmp_path, seeds=(42,))
    _set_scoring(rq1, "run_seed42_x", truncation=None)
    with pytest.raises(ValueError, match="truncation"):
        report_main(["--rq1-root", str(rq1), "--runs-root", str(runs), "--data-dir", str(data),
                     "--out-md", str(tmp_path / "x.md"), "--n-boot", "10"])


def test_report_refuses_mixed_truncation_modes(tmp_path):
    data, rq1, runs = _make_toy(tmp_path)
    _set_scoring(rq1, "run_seed43_x", truncation="other")
    with pytest.raises(ValueError, match="different truncation modes"):
        report_main(["--rq1-root", str(rq1), "--runs-root", str(runs), "--data-dir", str(data),
                     "--out-md", str(tmp_path / "x.md"), "--n-boot", "10"])
    assert not (rq1 / "run_seed42_x.json").exists()     # refused before any file was written


# ---- score_rq1.make_eval_loader: truncation mode from the run's snapshot (encoder-fit-first, 20 Nov 2026) ----
class _SpecialTokensOnly:
    """Just enough tokenizer for Collator.__post_init__; the loader is never iterated, so nothing is encoded."""
    def num_special_tokens_to_add(self, pair=True):
        return 3


def _fake_snapshot(tmp_path, truncation):
    """A config_snapshot.json-like dict built from base.yaml; truncation=None leaves model.truncation out,
    as in every G snapshot written before 20 Nov 2026."""
    import copy
    import yaml
    with open("configs/base.yaml") as f:
        cfg = copy.deepcopy(yaml.safe_load(f))
    rows = [{"query": "q", "context": "c", "answer": "a", "label": 0, "source": "ragtruth"}]
    path = tmp_path / "val.jsonl"
    _write_jsonl(path, rows)
    cfg["data"]["val_path"] = str(path)
    if truncation is None:
        cfg["model"].pop("truncation", None)
    else:
        cfg["model"]["truncation"] = truncation
    return {"config": cfg, "experiment": "G"}


def test_make_eval_loader_takes_truncation_from_snapshot(tmp_path):
    from scripts.score_rq1 import make_eval_loader
    cfg = _fake_snapshot(tmp_path, "fit_first")["config"]
    loader = make_eval_loader(cfg["data"]["val_path"], cfg, _SpecialTokensOnly(), batch_size=2)
    coll = loader.collate_fn
    assert coll.truncation == "fit_first"
    assert (coll.max_length, coll.resp_max) == (cfg["model"]["max_length"], cfg["model"]["resp_max"])


def test_make_eval_loader_refuses_snapshot_without_truncation(tmp_path):
    """A pre-20 Nov G snapshot has no model.truncation (legacy encoding, removed): scoring it must fail."""
    from scripts.score_rq1 import make_eval_loader
    cfg = _fake_snapshot(tmp_path, None)["config"]
    assert "truncation" not in cfg["model"]                      # guard: the case is what it claims
    with pytest.raises(ValueError):
        make_eval_loader(cfg["data"]["val_path"], cfg, _SpecialTokensOnly(), batch_size=2)
