"""Spec for Phase 4 (RQ1: G zero-shot on clinical data).

Templates (Prajakta writes the bodies, in src/rq1.py):
  check_g_snapshot(snapshot)                       -> raises if the run is not a reportable G run in the current format
  load_g_checkpoint(run_dir, device, build_model)  -> (model in eval mode, snapshot)
  assign_row_keys(rows)                            -> stable per-row key, duplicates numbered
  derive_clean_predictions(full_preds, clean_rows) -> clean-test predictions taken from test_full predictions
  tune_clinical_thresholds(val_preds, pooled)      -> threshold (b) per source; CovidQA tuned on pooled RAGBench val
  faithful_class_precision(labels, probs, thr)     -> TN / (TN + FN)
  subset_metrics(preds, thr_ragtruth, thr_clinical)-> n, n_pos, prevalence, PR-AUC, AUROC, P/R/F1 at both thresholds
  cluster_bootstrap_ci(labels, probs, groups, ...) -> 95% percentile CIs, resampling whole groups
  spread(values)                                   -> mean, sample SD, min, max across seeds
Plumbing (Claude Code): scripts/score_rq1.py (parity_report), scripts/report_rq1.py (build_m3_summary, group_by, report).

Decisions used (17 Nov): headline = test_full; duplicates kept and numbered; bootstrap per seed, B=2000,
F1 at both thresholds; threshold (b) on clean clinical val with CovidQA pooled with PubMedQA.

Run: python -m pytest tests/test_rq1.py
"""

import hashlib
import json
import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.encoding import INPUT_FORMAT
from src.metrics import best_threshold
from src.rq1 import (
    POOLED_VAL,
    assign_row_keys,
    check_g_snapshot,
    cluster_bootstrap_ci,
    derive_clean_predictions,
    faithful_class_precision,
    load_g_checkpoint,
    spread,
    subset_metrics,
    tune_clinical_thresholds,
)


# ---- toy data -------------------------------------------------------------------

def row(source, group_id, answer, label, **extra):
    return {"query": "q", "context": "c", "answer": answer, "label": label,
            "source": source, "task": "QA", "group_id": group_id, **extra}


def pred(source, group_id, label, prob, row_key=None, idx=0, split="test_full", **extra):
    return {"split": split, "idx": idx, "source": source, "task": "QA", "group_id": group_id,
            "label": label, "prob": prob, "row_key": row_key, **extra}


def good_snapshot():
    return {
        "seed": 42,
        "experiment": "G",
        "input_format": INPUT_FORMAT,
        "git_sha": "6294bb2",
        "config": {"model": {"name": "answerdotai/ModernBERT-base", "max_length": 1024,
                             "resp_max": 384, "dropout": 0.1},
                   "data": {"query_col": "query", "context_col": "context", "response_col": "answer",
                            "label_col": "label", "dataset_col": "source"}},
    }


# ---- check_g_snapshot / load_g_checkpoint -------------------------------------------

def test_snapshot_good_passes():
    check_g_snapshot(good_snapshot())   # no exception


@pytest.mark.parametrize("path, bad", [
    (("experiment",), "C"),
    (("input_format",), "{context}"),
    (("config", "model", "max_length"), 256),
    (("config", "model", "resp_max"), 128),
])
def test_snapshot_wrong_field_raises(path, bad):
    snap = good_snapshot()
    d = snap
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = bad
    with pytest.raises(ValueError):
        check_g_snapshot(snap)


def test_snapshot_missing_field_raises():
    snap = good_snapshot()
    del snap["input_format"]
    with pytest.raises(ValueError):
        check_g_snapshot(snap)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 1)
        self.drop = nn.Dropout(0.5)


def _write_run(tmp_path, snapshot, state_dict):
    run = tmp_path / "run_seed42_x"
    run.mkdir()
    (run / "config_snapshot.json").write_text(json.dumps(snapshot))
    torch.save(state_dict, run / "best.pt")
    return run


def test_load_checkpoint_restores_weights_and_eval_mode(tmp_path):
    src = Tiny()
    with torch.no_grad():
        src.lin.weight.fill_(0.25)
    run = _write_run(tmp_path, good_snapshot(), src.state_dict())
    model, snap = load_g_checkpoint(str(run), "cpu", build_model=lambda cfg: Tiny())
    assert snap["seed"] == 42
    assert not model.training                          # eval mode: dropout off
    assert torch.allclose(model.lin.weight, torch.full((1, 3), 0.25))


def test_load_checkpoint_strict_on_missing_keys(tmp_path):
    sd = Tiny().state_dict()
    del sd["lin.bias"]
    run = _write_run(tmp_path, good_snapshot(), sd)
    # match the key name: NotImplementedError is also a RuntimeError, so a bare RuntimeError check would pass vacuously
    with pytest.raises(RuntimeError, match="lin.bias"):
        load_g_checkpoint(str(run), "cpu", build_model=lambda cfg: Tiny())


def test_load_checkpoint_checks_snapshot_before_loading(tmp_path):
    snap = good_snapshot()
    snap["experiment"] = "G+C"
    run = _write_run(tmp_path, snap, Tiny().state_dict())
    with pytest.raises(ValueError):
        load_g_checkpoint(str(run), "cpu", build_model=lambda cfg: Tiny())


# ---- assign_row_keys --------------------------------------------------------------

def test_row_key_format_is_sha1_of_source_group_answer_plus_occurrence():
    r = row("ragbench_covidqa", "g1", "the answer", 0)
    base = hashlib.sha1("ragbench_covidqa|g1|the answer".encode("utf-8")).hexdigest()
    assert assign_row_keys([r]) == [base + "#0"]


def test_row_keys_number_duplicates_in_file_order():
    a = row("ragbench_covidqa", "g1", "same", 0)
    b = row("ragbench_covidqa", "g2", "other", 1)
    keys = assign_row_keys([a, b, dict(a), dict(a)])
    assert len(set(keys)) == 4
    assert [k.split("#")[1] for k in keys] == ["0", "0", "1", "2"]
    assert keys[0].split("#")[0] == keys[2].split("#")[0] == keys[3].split("#")[0]


def test_row_keys_depend_on_source_group_and_answer_only():
    a = row("medhallu", "g1", "ans", 0)
    a2 = dict(a, context="different context", query="other q", label=1)
    assert assign_row_keys([a]) == assign_row_keys([a2])
    assert assign_row_keys([a]) != assign_row_keys([dict(a, source="ragbench_pubmedqa")])
    assert assign_row_keys([a]) != assign_row_keys([dict(a, group_id="g2")])
    assert assign_row_keys([a]) != assign_row_keys([dict(a, answer="ans.")])


def test_row_keys_stable_across_files():
    """A row has the same key in test_full and in clean test when no duplicate of it is removed."""
    rows = [row("medhallu", f"g{i}", f"a{i}", i % 2) for i in range(6)]
    full_keys = assign_row_keys(rows)
    clean = [rows[1], rows[4], rows[5]]
    assert assign_row_keys(clean) == [full_keys[1], full_keys[4], full_keys[5]]


# ---- derive_clean_predictions ---------------------------------------------------------

def _full_and_clean():
    full_rows = [row("ragbench_pubmedqa", f"g{i}", f"a{i}", i % 2) for i in range(5)]
    keys = assign_row_keys(full_rows)
    full_preds = [pred(r["source"], r["group_id"], r["label"], 0.1 * (i + 1), row_key=k, idx=i)
                  for i, (r, k) in enumerate(zip(full_rows, keys))]
    clean_rows = [full_rows[3], full_rows[0], full_rows[4]]   # any order; output follows clean order
    return full_rows, full_preds, clean_rows


def test_clean_predictions_follow_clean_file_order():
    _, full_preds, clean_rows = _full_and_clean()
    out = derive_clean_predictions(full_preds, clean_rows)
    assert [p["group_id"] for p in out] == ["g3", "g0", "g4"]
    assert [p["prob"] for p in out] == pytest.approx([0.4, 0.1, 0.5])
    assert [p["idx"] for p in out] == [0, 1, 2]          # position in the clean file
    assert all(p["split"] == "test_clean" for p in out)


def test_clean_predictions_do_not_mutate_full():
    _, full_preds, clean_rows = _full_and_clean()
    before = json.dumps(full_preds, sort_keys=True)
    derive_clean_predictions(full_preds, clean_rows)
    assert json.dumps(full_preds, sort_keys=True) == before


def test_clean_row_missing_from_full_raises():
    _, full_preds, clean_rows = _full_and_clean()
    clean_rows = clean_rows + [row("ragbench_pubmedqa", "g99", "new", 0)]
    with pytest.raises(ValueError):
        derive_clean_predictions(full_preds, clean_rows)


def test_clean_label_mismatch_raises():
    _, full_preds, clean_rows = _full_and_clean()
    clean_rows = [dict(clean_rows[0], label=1 - clean_rows[0]["label"])] + clean_rows[1:]
    with pytest.raises(ValueError):
        derive_clean_predictions(full_preds, clean_rows)


def test_clean_row_that_is_duplicated_in_full_raises():
    """Ambiguous: which copy's prediction belongs to the clean row? (none in today's data)"""
    full_rows = [row("ragbench_covidqa", "g1", "dup", 0), row("ragbench_covidqa", "g1", "dup", 0),
                 row("ragbench_covidqa", "g2", "x", 1)]
    keys = assign_row_keys(full_rows)
    full_preds = [pred(r["source"], r["group_id"], r["label"], 0.3, row_key=k, idx=i)
                  for i, (r, k) in enumerate(zip(full_rows, keys))]
    with pytest.raises(ValueError):
        derive_clean_predictions(full_preds, [full_rows[0], full_rows[2]])


def test_full_predictions_with_duplicate_row_key_raise():
    _, full_preds, clean_rows = _full_and_clean()
    with pytest.raises(ValueError):
        derive_clean_predictions(full_preds + [dict(full_preds[0])], clean_rows)


# ---- tune_clinical_thresholds -------------------------------------------------------------

def _val_preds():
    rng = np.random.default_rng(0)
    out = []
    # pubmedqa: positives high-ish, many rows; covidqa: few rows, different spread; medhallu separate
    for src, n, shift in [("ragbench_pubmedqa", 120, 0.35), ("ragbench_covidqa", 30, 0.15), ("medhallu", 80, 0.25)]:
        for i in range(n):
            y = int(i % 3 == 0)
            p = float(np.clip(rng.normal(0.3 + shift * y, 0.15), 0.0, 1.0))
            out.append(pred(src, f"{src}{i}", y, p, split="val"))
    return out


def _lp(preds, sources):
    sel = [p for p in preds if p["source"] in sources]
    return np.array([p["label"] for p in sel]), np.array([p["prob"] for p in sel])


def test_pooled_val_constant():
    assert set(POOLED_VAL["ragbench_covidqa"]) == {"ragbench_pubmedqa", "ragbench_covidqa"}
    assert "ragbench_pubmedqa" not in POOLED_VAL and "medhallu" not in POOLED_VAL


def test_thresholds_own_source_except_covidqa_pooled():
    val = _val_preds()
    out = tune_clinical_thresholds(val)
    assert set(out) == {"ragbench_pubmedqa", "ragbench_covidqa", "medhallu"}
    for src in ("ragbench_pubmedqa", "medhallu"):
        y, p = _lp(val, {src})
        assert out[src]["threshold"] == pytest.approx(best_threshold(y, p))
        assert out[src]["tuned_on"] == [src]
        assert out[src]["n"] == len(y) and out[src]["n_pos"] == int(y.sum())
    y, p = _lp(val, {"ragbench_pubmedqa", "ragbench_covidqa"})
    cov = out["ragbench_covidqa"]
    assert cov["threshold"] == pytest.approx(best_threshold(y, p))
    assert cov["tuned_on"] == ["ragbench_covidqa", "ragbench_pubmedqa"]   # sorted
    assert cov["n"] == 150 and cov["n_pos"] == int(y.sum())


def test_thresholds_are_plain_floats_on_the_training_grid():
    out = tune_clinical_thresholds(_val_preds())
    grid = np.linspace(0.05, 0.95, 19)
    for v in out.values():
        assert type(v["threshold"]) is float
        assert np.isclose(grid, v["threshold"]).any()


def test_pooled_source_missing_from_val_raises():
    val = [p for p in _val_preds() if p["source"] != "ragbench_pubmedqa"]
    with pytest.raises(ValueError):
        tune_clinical_thresholds(val)


def test_source_without_positives_raises():
    val = [dict(p, label=0) if p["source"] == "medhallu" else p for p in _val_preds()]
    with pytest.raises(ValueError):
        tune_clinical_thresholds(val)


# ---- faithful_class_precision ---------------------------------------------------------------

def test_faithful_precision_hand_count():
    labels = np.array([1, 1, 0, 0, 0])
    probs = np.array([0.9, 0.4, 0.6, 0.2, 0.1])
    # thr 0.5: predicted faithful (prob < 0.5) = rows 1, 3, 4 -> TN = 2 (rows 3, 4), FN = 1 (row 1)
    assert faithful_class_precision(labels, probs, 0.5) == pytest.approx(2 / 3)
    # thr 0.3: predicted faithful = rows 3, 4 -> both truly faithful
    assert faithful_class_precision(labels, probs, 0.3) == pytest.approx(1.0)


def test_faithful_precision_prob_equal_to_threshold_is_hallucinated():
    """Same rule as thresholded_metrics: prob >= threshold -> predicted hallucinated."""
    labels = np.array([0, 1])
    probs = np.array([0.5, 0.2])
    assert faithful_class_precision(labels, probs, 0.5) == pytest.approx(0.0)   # only row 1 predicted faithful, it is a FN


def test_faithful_precision_none_when_nothing_predicted_faithful():
    assert faithful_class_precision(np.array([0, 1]), np.array([0.8, 0.9]), 0.5) is None


# ---- subset_metrics -----------------------------------------------------------------------

def _hand_preds():
    labels = [1, 1, 0, 0, 0]
    probs = [0.9, 0.4, 0.6, 0.2, 0.1]
    return [pred("medhallu", f"g{i}", y, p) for i, (y, p) in enumerate(zip(labels, probs))]


def test_subset_metrics_hand_values():
    m = subset_metrics(_hand_preds(), thr_ragtruth=0.5, thr_clinical=0.3)
    assert m["n"] == 5 and m["n_pos"] == 2
    assert m["prevalence"] == pytest.approx(0.4)
    assert m["pr_auc"] == pytest.approx(0.5 * 1 + 0.5 * (2 / 3))      # AP: P=1 at R=.5, P=2/3 at R=1
    assert m["roc_auc"] == pytest.approx(5 / 6)
    a = m["at_ragtruth_thr"]
    assert a["threshold"] == 0.5
    assert (a["precision"], a["recall"], a["f1"]) == pytest.approx((0.5, 0.5, 0.5))
    assert a["faithful_precision"] == pytest.approx(2 / 3)
    b = m["at_clinical_thr"]
    assert b["threshold"] == 0.3
    assert (b["precision"], b["recall"], b["f1"]) == pytest.approx((2 / 3, 1.0, 0.8))
    assert b["faithful_precision"] == pytest.approx(1.0)


def test_subset_metrics_json_serialisable_plain_types():
    m = subset_metrics(_hand_preds(), 0.5, 0.3)
    json.dumps(m)
    assert type(m["n"]) is int and type(m["pr_auc"]) is float


def test_subset_metrics_single_class_gives_none_ranking_metrics():
    preds = [pred("medhallu", "g", 0, p) for p in (0.1, 0.7)]
    m = subset_metrics(preds, 0.5, 0.5)
    assert m["pr_auc"] is None and m["roc_auc"] is None
    assert m["n_pos"] == 0 and m["prevalence"] == 0.0
    assert m["at_ragtruth_thr"]["faithful_precision"] == pytest.approx(1.0)


def test_subset_metrics_empty_raises():
    with pytest.raises(ValueError):
        subset_metrics([], 0.5, 0.5)


def test_subset_metrics_non_binary_label_raises():
    preds = _hand_preds()
    preds[0]["label"] = 2
    with pytest.raises(ValueError):
        subset_metrics(preds, 0.5, 0.5)


# ---- cluster_bootstrap_ci -------------------------------------------------------------------

def _boot_data(n_groups=200, seed=1):
    rng = np.random.default_rng(seed)
    labels = (rng.random(n_groups) < 0.2).astype(int)
    probs = np.clip(rng.normal(0.3 + 0.3 * labels, 0.15), 0, 1)
    groups = np.array([f"g{i}" for i in range(n_groups)])
    return labels, probs, groups


def _mean_label(y, p):
    return float(np.mean(y))


def test_bootstrap_reproducible_with_seed_and_changes_with_seed():
    from sklearn.metrics import roc_auc_score
    y, p, g = _boot_data()
    fns = {"roc_auc": lambda yy, pp: float(roc_auc_score(yy, pp))}
    a = cluster_bootstrap_ci(y, p, g, fns, n_boot=300, seed=7)
    b = cluster_bootstrap_ci(y, p, g, fns, n_boot=300, seed=7)
    c = cluster_bootstrap_ci(y, p, g, fns, n_boot=300, seed=8)
    assert a == b
    assert (a["roc_auc"]["lo"], a["roc_auc"]["hi"]) != (c["roc_auc"]["lo"], c["roc_auc"]["hi"])


def test_bootstrap_ci_brackets_point_estimate():
    from sklearn.metrics import average_precision_score, roc_auc_score
    y, p, g = _boot_data()
    fns = {"pr_auc": lambda yy, pp: float(average_precision_score(yy, pp)),
           "roc_auc": lambda yy, pp: float(roc_auc_score(yy, pp))}
    out = cluster_bootstrap_ci(y, p, g, fns, n_boot=500, seed=0)
    for name in fns:
        r = out[name]
        assert set(r) == {"point", "lo", "hi", "n_valid"}
        assert r["point"] == pytest.approx(fns[name](y, p))
        assert r["lo"] <= r["point"] <= r["hi"]
        assert r["n_valid"] == 500
        assert all(type(r[k]) is float for k in ("point", "lo", "hi"))


def test_bootstrap_resamples_whole_groups():
    """Every group = one faithful + one hallucinated row, so any group resample has prevalence exactly 0.5.
    A row-level resample would vary."""
    n = 50
    labels = np.array([1, 0] * n)
    probs = np.linspace(0, 1, 2 * n)
    groups = np.repeat([f"g{i}" for i in range(n)], 2)
    out = cluster_bootstrap_ci(labels, probs, groups, {"prev": _mean_label}, n_boot=200, seed=0)
    assert out["prev"]["lo"] == pytest.approx(0.5) and out["prev"]["hi"] == pytest.approx(0.5)


def test_bootstrap_skips_single_class_resamples():
    labels = np.array([1, 0, 0])
    probs = np.array([0.9, 0.2, 0.1])
    groups = np.array(["a", "b", "c"])
    out = cluster_bootstrap_ci(labels, probs, groups, {"prev": _mean_label}, n_boot=400, seed=0)
    # P(no positive group drawn) = (2/3)^3 ~ 0.30, so roughly 120 of 400 resamples are skipped
    assert 200 < out["prev"]["n_valid"] < 400
    assert out["prev"]["lo"] > 0.0          # skipped resamples never enter the CI


def test_bootstrap_ci_none_when_no_valid_resample():
    labels = np.array([0, 0])
    out = cluster_bootstrap_ci(labels, np.array([0.1, 0.2]), np.array(["a", "b"]),
                               {"prev": _mean_label}, n_boot=50, seed=0)
    assert out["prev"]["n_valid"] == 0
    assert out["prev"]["lo"] is None and out["prev"]["hi"] is None


def test_bootstrap_length_mismatch_raises():
    y, p, g = _boot_data()
    with pytest.raises(ValueError):
        cluster_bootstrap_ci(y, p[:-1], g, {"prev": _mean_label}, n_boot=10, seed=0)


# ---- spread ---------------------------------------------------------------------------------

def test_spread_sample_sd_min_max():
    s = spread([0.4, 0.5, 0.6])
    assert s["mean"] == pytest.approx(0.5)
    assert s["std"] == pytest.approx(0.1)          # sample SD (n-1), not population (0.0816)
    assert (s["min"], s["max"]) == (0.4, 0.6)


def test_spread_one_value_std_none():
    assert spread([0.7]) == {"mean": 0.7, "std": None, "min": 0.7, "max": 0.7}


@pytest.mark.parametrize("bad", [[], [0.5, None]])
def test_spread_rejects_empty_or_none(bad):
    with pytest.raises(ValueError):
        spread(bad)
