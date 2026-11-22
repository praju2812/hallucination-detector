"""Phase 5 (RQ2): train.py with several test sets and configurable selection sources.

Spec for the templates (you write the bodies):
    T1 selection_mask, T2 evaluate_test_sets, T3 merge_per_source (src/train.py)
and for your two edits:
    validate_and_save_best_model and evaluate_on_test call T1; evaluate_on_test takes test_loaders (a dict).
Plumbing tested here too: select_sources_from_cfg, named_test_sets, build_metrics_summary,
resolved_training_values vs the real scheduler, and what scripts/mvp_check.py M4 sees.

No ModernBERT download: a one-parameter model reads its logit straight from input_ids.

Run: python -m pytest tests/test_train_multi.py
"""

import copy
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from torch import nn

import scripts.mvp_check as mvp                 # module import: pytest must not collect mvp.test_block as a test
from src.train import (
    build_metrics_summary,
    build_optimizer_and_scheduler,
    build_run_snapshot,
    evaluate_on_test,
    evaluate_test_sets,
    merge_per_source,
    resolved_training_values,
    select_sources_from_cfg,
    selection_mask,
    named_test_sets,
    validate_and_save_best_model,
)


def _load(path):
    with open(path) as f:
        return yaml.safe_load(f)


BASE = _load("configs/base.yaml")
C_CFG = _load("configs/c.yaml")
GC_CFG = _load("configs/gc.yaml")
SELECT = ["ragbench_pubmedqa", "ragbench_covidqa"]
CLINICAL = ["ragbench_pubmedqa", "ragbench_covidqa", "medhallu"]
LEGACY_KEYS = {"threshold", "overall", "per_source", "git_sha", "seed"}           # G metrics.json today
MULTI_KEYS = {"threshold", "threshold_tuned_on", "per_source", "per_test_set", "git_sha", "seed"}

SCALE = 1000   # logits travel as round(logit * SCALE) inside input_ids


# ---- toy model and loaders --------------------------------------------------------------

class LogitFromIds(nn.Module):
    """logit = w * input_ids[:, 0] / SCALE, with w = 1. One parameter, so best.pt round-trips."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        return self.w * input_ids[:, 0].float() / SCALE


def _logit(p):
    return math.log(p / (1 - p))


class FakeLoader:
    """Iterates like the real DataLoader + Collator: {"enc": {..., "labels"}, "sources"}; .dataset.rows = file rows."""

    def __init__(self, rows, probs, batch_size=4):
        assert len(rows) == len(probs)
        self.dataset = SimpleNamespace(rows=rows)
        self.probs = probs
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.probs) / self.batch_size)

    def __iter__(self):
        for i in range(0, len(self.probs), self.batch_size):
            rows, probs = self.dataset.rows[i:i + self.batch_size], self.probs[i:i + self.batch_size]
            ids = torch.tensor([[round(_logit(p) * SCALE)] for p in probs], dtype=torch.long)
            yield {"enc": {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                           "labels": torch.tensor([float(r["label"]) for r in rows])},
                   "sources": [r["source"] for r in rows]}


def _row(source, label, i):
    return {"query": "q", "context": "c", "answer": f"{source}-{i}", "label": label, "source": source,
            "task": "QA", "group_id": f"{source}-g{i}"}


# Validation rows chosen so each selection rule gives a different threshold (grid 0.05..0.95, step 0.05):
#   PubMedQA + CovidQA only -> 0.60 (F1 = 1 only there)     RAGTruth only -> 0.25
#   MedHallu is inverted: including it would change both the threshold and the PR-AUC.
VAL = [("ragbench_pubmedqa", 1, 0.62), ("ragbench_pubmedqa", 0, 0.57),
       ("ragbench_covidqa", 1, 0.72), ("ragbench_covidqa", 0, 0.33),
       ("medhallu", 1, 0.12), ("medhallu", 0, 0.92),
       ("ragtruth", 1, 0.27), ("ragtruth", 0, 0.22),
       ("halueval", 1, 0.97), ("halueval", 0, 0.03)]
TEST_GENERAL = [("ragtruth", 1, 0.81), ("ragtruth", 1, 0.41), ("ragtruth", 0, 0.55), ("ragtruth", 0, 0.07),
                ("halueval", 1, 0.90), ("halueval", 0, 0.10)]
TEST_FULL = [("ragbench_pubmedqa", 1, 0.66), ("ragbench_pubmedqa", 0, 0.61),
             ("ragbench_covidqa", 1, 0.70), ("ragbench_covidqa", 0, 0.20),
             ("medhallu", 1, 0.65), ("medhallu", 0, 0.30)]


def _loader(spec):
    return FakeLoader([_row(s, y, i) for i, (s, y, _) in enumerate(spec)], [p for _, _, p in spec])


def _cfg(base, tmp_path):
    cfg = copy.deepcopy(base)
    cfg["train"]["out_dir"] = str(tmp_path)
    return cfg


def _model_with_best(run_dir):
    model = LogitFromIds()
    torch.save(model.state_dict(), run_dir / "best.pt")
    return model


def _read_preds(run_dir):
    with open(run_dir / "predictions.jsonl") as f:
        return [json.loads(l) for l in f]


# ---- plumbing: config readers -------------------------------------------------------------

def test_select_sources_default_is_ragtruth():
    assert select_sources_from_cfg(BASE) == ["ragtruth"]
    assert select_sources_from_cfg(C_CFG) == SELECT
    assert select_sources_from_cfg(GC_CFG) == SELECT


def test_named_test_sets():
    assert named_test_sets(BASE["data"]) == {"test": "data/unified_v2/test.jsonl"}
    assert list(named_test_sets(C_CFG["data"])) == ["test", "test_full"]      # order kept
    both = {**C_CFG["data"], "test_path": "x.jsonl"}
    neither = {k: v for k, v in BASE["data"].items() if k != "test_path"}
    for bad in (both, neither, {**C_CFG["data"], "test_sets": {}}):
        with pytest.raises(ValueError):
            named_test_sets(bad)


# ---- plumbing: resolved values match what the scheduler actually does ---------------------------

def test_resolved_warmup_is_the_scheduler_warmup():
    cfg = copy.deepcopy(BASE)
    steps_per_epoch = 50                                     # total 100 -> warmup int(6.0) = 6
    r = resolved_training_values(cfg, n_train_rows=400, steps_per_epoch=steps_per_epoch)
    assert r["warmup_steps"] == 6 and r["total_steps"] == 100
    model = nn.Linear(2, 1)
    optim, sched = build_optimizer_and_scheduler(model, cfg, steps_per_epoch)
    lrs = []
    for _ in range(r["warmup_steps"]):
        optim.step()
        sched.step()
        lrs.append(sched.get_last_lr()[0])
    assert lrs[-2] < cfg["train"]["lr"]                      # still warming up one step before
    assert lrs[-1] == pytest.approx(cfg["train"]["lr"])      # peak reached exactly at warmup_steps


def test_snapshot_still_validates_experiment_for_new_configs():
    for cfg, exp in ((C_CFG, "C"), (GC_CFG, "G+C")):
        snap = build_run_snapshot(copy.deepcopy(cfg))
        assert snap["experiment"] == exp
        assert snap["config"]["train"]["select_sources"] == SELECT
        assert snap["truncation"] == "fit_first"


# ---- T1 selection_mask -----------------------------------------------------------------------

def test_mask_one_source():
    m = selection_mask(["ragtruth", "halueval", "ragtruth"], np.array([1.0, 0.0, 0.0]), ["ragtruth"])
    assert m.dtype == bool and m.tolist() == [True, False, True]


def test_mask_two_sources_pooled():
    srcs = ["ragbench_pubmedqa", "medhallu", "ragbench_covidqa", "ragtruth"]
    m = selection_mask(srcs, np.array([1.0, 1.0, 0.0, 1.0]), SELECT)
    assert m.tolist() == [True, False, True, False]


def test_mask_no_matching_row_raises():
    with pytest.raises(ValueError):
        selection_mask(["halueval", "medhallu"], np.array([1.0, 0.0]), ["ragtruth"])


def test_mask_unknown_source_name_raises():
    """A typo must not silently select on the other source alone."""
    with pytest.raises(ValueError):
        selection_mask(["ragbench_pubmedqa", "ragbench_covidqa"], np.array([1.0, 0.0]),
                       ["ragbench_pubmedqa", "ragbench_covidq"])


def test_mask_without_positives_raises():
    with pytest.raises(ValueError):
        selection_mask(["ragtruth", "ragtruth", "medhallu"], np.array([0.0, 0.0, 1.0]), ["ragtruth"])


# ---- T3 merge_per_source ---------------------------------------------------------------------

def test_merge_union_in_order():
    pts = {"test": {"overall": {}, "per_source": {"ragtruth": {"n": 4}, "halueval": {"n": 2}}},
           "test_full": {"overall": {}, "per_source": {"medhallu": {"n": 2}}}}
    merged = merge_per_source(pts)
    assert list(merged) == ["ragtruth", "halueval", "medhallu"]
    assert merged["medhallu"] == {"n": 2}


def test_merge_repeated_source_raises():
    pts = {"test_full": {"overall": {}, "per_source": {"medhallu": {"n": 2}}},
           "test_clean": {"overall": {}, "per_source": {"medhallu": {"n": 1}}}}
    with pytest.raises(ValueError):
        merge_per_source(pts)


def test_merge_does_not_modify_input():
    pts = {"test": {"overall": {}, "per_source": {"ragtruth": {"n": 4}}}}
    before = copy.deepcopy(pts)
    merge_per_source(pts)
    assert pts == before


# ---- T2 evaluate_test_sets -------------------------------------------------------------------

def test_evaluate_test_sets_per_set_metrics_and_records(tmp_path):
    cfg = _cfg(C_CFG, tmp_path)
    model = LogitFromIds()
    loaders = {"test": _loader(TEST_GENERAL), "test_full": _loader(TEST_FULL)}
    per_set, records = evaluate_test_sets(model, loaders, 0.6, cfg, "cpu")
    assert list(per_set) == ["test", "test_full"]
    assert set(per_set["test"]) == {"overall", "per_source"}
    assert per_set["test"]["overall"]["threshold"] == 0.6
    assert set(per_set["test"]["per_source"]) == {"ragtruth", "halueval"}
    assert set(per_set["test_full"]["per_source"]) == set(CLINICAL)
    rt = per_set["test"]["per_source"]["ragtruth"]
    assert (rt["n"], rt["n_pos"]) == (4, 2)
    assert rt["f1"] == pytest.approx(2 / 3)        # at 0.6: only 0.81 flagged -> P 1, R 0.5
    pub = per_set["test_full"]["per_source"]["ragbench_pubmedqa"]
    assert pub["f1"] == pytest.approx(2 / 3)       # at 0.6: 0.66 and 0.61 flagged -> P 0.5, R 1
    assert [r["split"] for r in records] == ["test"] * 6 + ["test_full"] * 6
    assert [r["idx"] for r in records if r["split"] == "test_full"] == list(range(6))
    assert records[0]["prob"] == pytest.approx(0.81, abs=1e-3)


# ---- your edits: validate_and_save_best_model ------------------------------------------------------

def test_validation_selects_on_pooled_ragbench_only(tmp_path):
    cfg = _cfg(C_CFG, tmp_path)
    model = LogitFromIds()
    # RAGTruth rows inverted here, so selecting on RAGTruth (today's rule) gives PR-AUC 0.5, not 1
    val_spec = [(s, y, (0.22 if y else 0.27) if s == "ragtruth" else p) for s, y, p in VAL]
    score, val = validate_and_save_best_model(cfg, "cpu", _loader(val_spec), model, -1.0, 0, str(tmp_path))
    # PubMedQA + CovidQA are perfectly separated; the inverted MedHallu and RAGTruth rows would pull PR-AUC below 1
    assert val["pr_auc"] == pytest.approx(1.0)
    assert score == pytest.approx(1.0)
    assert (tmp_path / "best.pt").exists()


def test_validation_g_still_selects_on_ragtruth(tmp_path):
    cfg = _cfg(BASE, tmp_path)
    score, val = validate_and_save_best_model(cfg, "cpu", _loader(VAL), LogitFromIds(), -1.0, 0, str(tmp_path))
    assert val["pr_auc"] == pytest.approx(1.0) and val["roc_auc"] == pytest.approx(1.0)


def test_validation_unknown_select_source_raises(tmp_path):
    cfg = _cfg(C_CFG, tmp_path)
    cfg["train"]["select_sources"] = ["ragbench_pubmedqa", "ragbench_covidq"]
    with pytest.raises(ValueError):
        validate_and_save_best_model(cfg, "cpu", _loader(VAL), LogitFromIds(), -1.0, 0, str(tmp_path))


# ---- your edits: evaluate_on_test ------------------------------------------------------------------

def test_g_single_test_set_keeps_todays_outputs(tmp_path):
    cfg = _cfg(BASE, tmp_path)
    model = _model_with_best(tmp_path)
    evaluate_on_test(model, _loader(VAL), {"test": _loader(TEST_GENERAL)}, cfg, "cpu", str(tmp_path))
    m = json.loads((tmp_path / "metrics.json").read_text())
    assert set(m) == LEGACY_KEYS
    assert m["threshold"] == pytest.approx(0.25)                       # RAGTruth val rows only
    assert set(m["per_source"]) == {"ragtruth", "halueval"}
    assert m["per_source"]["ragtruth"]["f1"] == pytest.approx(0.8)    # at 0.25: 0.81, 0.41, 0.55 flagged
    preds = _read_preds(tmp_path)
    assert [p["split"] for p in preds] == ["val"] * len(VAL) + ["test"] * len(TEST_GENERAL)


def test_c_two_test_sets(tmp_path):
    cfg = _cfg(C_CFG, tmp_path)
    model = _model_with_best(tmp_path)
    loaders = {"test": _loader(TEST_GENERAL), "test_full": _loader(TEST_FULL)}
    evaluate_on_test(model, _loader(VAL), loaders, cfg, "cpu", str(tmp_path))
    m = json.loads((tmp_path / "metrics.json").read_text())
    assert set(m) == MULTI_KEYS
    assert m["threshold"] == pytest.approx(0.60)                        # PubMedQA + CovidQA val rows only
    assert m["threshold_tuned_on"] == SELECT
    assert list(m["per_test_set"]) == ["test", "test_full"]
    assert set(m["per_source"]) == {"ragtruth", "halueval"} | set(CLINICAL)
    assert m["per_source"]["ragtruth"]["f1"] == pytest.approx(2 / 3)
    assert m["per_source"] == {**m["per_test_set"]["test"]["per_source"], **m["per_test_set"]["test_full"]["per_source"]}
    preds = _read_preds(tmp_path)
    counts = {s: sum(p["split"] == s for p in preds) for s in ("val", "test", "test_full")}
    assert counts == {"val": len(VAL), "test": len(TEST_GENERAL), "test_full": len(TEST_FULL)}
    assert len(preds) == sum(counts.values())
    # MedHallu val rows are scored and written, just never used for the threshold
    assert sum(p["split"] == "val" and p["source"] == "medhallu" for p in preds) == 2


# ---- plumbing: metrics.json shape ------------------------------------------------------------------

def _block(n=10, n_pos=2):
    return {"pr_auc": 0.5, "roc_auc": 0.7, "precision": 0.5, "recall": 0.5, "f1": 0.5, "threshold": 0.4,
            "n": n, "n_pos": n_pos}


def test_metrics_summary_single_set_is_legacy_shape():
    pts = {"test": {"overall": {"pr_auc": 0.9}, "per_source": {"ragtruth": _block()}}}
    s = build_metrics_summary(0.25, ["ragtruth"], pts, pts["test"]["per_source"], BASE)
    assert set(s) == LEGACY_KEYS
    assert s["overall"] == {"pr_auc": 0.9} and s["threshold"] == 0.25 and s["seed"] == BASE["train"]["seed"]


def test_metrics_summary_multi_set_shape():
    pts = {"test": {"overall": {}, "per_source": {"ragtruth": _block()}},
           "test_full": {"overall": {}, "per_source": {"medhallu": _block()}}}
    merged = {"ragtruth": _block(), "medhallu": _block()}
    s = build_metrics_summary(0.6, SELECT, pts, merged, C_CFG)
    assert set(s) == MULTI_KEYS
    assert s["threshold_tuned_on"] == SELECT and s["per_test_set"] == pts and s["per_source"] == merged


# ---- what mvp_check M4 sees ----------------------------------------------------------------------

def _write_run(root, cfg, seed, per_source):
    cfg = copy.deepcopy(cfg)
    cfg["train"]["seed"] = seed
    d = root / f"run_seed{seed}_{cfg['experiment'].replace('+', 'p')}"
    d.mkdir(parents=True)
    snap = build_run_snapshot(cfg)
    snap["resolved"] = resolved_training_values(cfg, 100, 13)
    (d / "config_snapshot.json").write_text(json.dumps(snap))
    sets = {"test": {"overall": {}, "per_source": {k: v for k, v in per_source.items() if k in ("ragtruth", "halueval")}},
            "test_full": {"overall": {}, "per_source": {k: v for k, v in per_source.items() if k in CLINICAL}}}
    (d / "metrics.json").write_text(json.dumps(build_metrics_summary(0.6, SELECT, sets, per_source, cfg)))
    return d


@pytest.fixture
def isolated_mvp(monkeypatch, tmp_path):
    monkeypatch.setattr(mvp, "RUN_ROOTS", [])                     # ignore real runs/ in the checkout
    monkeypatch.setattr(mvp, "RQ1_DIR", tmp_path / "no_rq1")
    mvp.RESULTS.clear()
    mvp.STATE.clear()
    return mvp


def test_mvp_m4_passes_on_three_seeds_each(isolated_mvp, tmp_path):
    full = {s: _block() for s in ["ragtruth", "halueval"] + CLINICAL}
    root = tmp_path / "runs"
    for cfg in (C_CFG, GC_CFG):
        for seed in (42, 43, 44):
            _write_run(root, cfg, seed, full)
    isolated_mvp.check_runs([str(root)], True)
    isolated_mvp.check_m3_m4()
    assert isolated_mvp.RESULTS["M4"] == "PASS"


def test_mvp_m4_ignores_a_run_missing_a_clinical_source(isolated_mvp, tmp_path):
    full = {s: _block() for s in ["ragtruth", "halueval"] + CLINICAL}
    no_mh = {k: v for k, v in full.items() if k != "medhallu"}
    root = tmp_path / "runs"
    for seed in (42, 43, 44):
        _write_run(root, GC_CFG, seed, full)
    for seed, ps in ((42, full), (43, full), (44, no_mh)):
        _write_run(root, C_CFG, seed, ps)
    isolated_mvp.check_runs([str(root)], True)
    isolated_mvp.check_m3_m4()
    assert isolated_mvp.RESULTS["M4"] == "NOT YET"


def test_mvp_marks_smoke_runs_invalid():
    cfg = copy.deepcopy(GC_CFG)
    cfg["data"]["train_path"] = ["data/unified_v2/tiny.jsonl", "/kaggle/working/smoke/clinical_tiny.jsonl"]
    snap = build_run_snapshot(cfg)
    assert any("tiny" in p for p in mvp.run_valid(snap, True))
    assert mvp.classify(snap)[0] == "G+C"
