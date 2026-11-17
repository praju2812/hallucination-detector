"""Spec for scripts/summarize_runs.py (Phase 2).

Templates (Prajakta writes the bodies):
  summarize_source(runs, source)              -> per-source mean / sample std / min / max across seeds, prevalence
  peak_epoch(val_history, select_on)          -> epoch with the best validation metric
  truncation_split_metrics(preds, flags, thr) -> RAGTruth test metrics for truncated vs untruncated rows
Plumbing (Claude Code): load_runs, filter_runs, format_report.

Decisions used (9 Nov): D5 spread = sample std (n-1) plus min and max.

Run: python -m pytest tests/test_summarize_runs.py
"""

import json
import math

import numpy as np
import pytest

from scripts.summarize_runs import (
    filter_runs,
    format_report,
    load_runs,
    peak_epoch,
    summarize_source,
    truncation_split_metrics,
)
from src.metrics import ranking_metrics, thresholded_metrics


# ---- toy runs -----------------------------------------------------------------

RAG = {"n": 1800, "n_pos": 364}
HALU = {"n": 2000, "n_pos": 1000}
# per seed: (pr_auc, f1, roc_auc) for ragtruth
RAG_VALUES = {42: (0.60, 0.50, 0.80), 43: (0.70, 0.55, 0.82), 44: (0.80, 0.60, 0.84)}


def _metrics(seed):
    pr, f1, roc = RAG_VALUES[seed]
    return {
        "threshold": 0.5,
        "overall": {"pr_auc": 0.99, "roc_auc": 0.99, "f1": 0.99},   # blended: must never be reported
        "per_source": {
            "ragtruth": {"pr_auc": pr, "f1": f1, "roc_auc": roc, **RAG},
            "halueval": {"pr_auc": 0.95, "f1": 0.90, "roc_auc": 0.97, **HALU},
        },
        "seed": seed,
        "git_sha": "abc1234",
    }


def _run(seed):
    return {"name": f"run_seed{seed}_20261110_000000", "metrics": _metrics(seed)}


RUNS = [_run(s) for s in (42, 43, 44)]


# ---- summarize_source (TEMPLATE) ---------------------------------------------

def test_summarize_source_mean_sample_std_min_max():
    out = summarize_source(RUNS, "ragtruth")
    assert out["source"] == "ragtruth"
    assert out["n_runs"] == 3
    assert out["pr_auc"]["mean"] == pytest.approx(0.70)
    assert out["pr_auc"]["std"] == pytest.approx(0.10)       # sample std (n-1), not population (0.0816)
    assert out["pr_auc"]["min"] == pytest.approx(0.60)
    assert out["pr_auc"]["max"] == pytest.approx(0.80)
    assert out["f1"]["mean"] == pytest.approx(0.55)
    assert out["f1"]["std"] == pytest.approx(0.05)
    assert out["roc_auc"]["mean"] == pytest.approx(0.82)
    assert out["roc_auc"]["std"] == pytest.approx(0.02)


def test_summarize_source_prevalence():
    out = summarize_source(RUNS, "ragtruth")
    assert out["n"] == 1800 and out["n_pos"] == 364
    assert out["prevalence"] == pytest.approx(364 / 1800)
    assert summarize_source(RUNS, "halueval")["prevalence"] == pytest.approx(0.5)


def test_summarize_source_single_run_std_is_none():
    out = summarize_source(RUNS[:1], "ragtruth")
    assert out["n_runs"] == 1
    assert out["pr_auc"]["std"] is None
    assert out["pr_auc"]["mean"] == out["pr_auc"]["min"] == out["pr_auc"]["max"] == pytest.approx(0.60)


def test_summarize_source_run_missing_source_raises():
    """A seed that silently lacks the source would shrink n without anyone noticing."""
    broken = {"name": "run_seed45_20261110_000000", "metrics": _metrics(42)}
    del broken["metrics"]["per_source"]["ragtruth"]
    with pytest.raises(ValueError):
        summarize_source(RUNS + [broken], "ragtruth")


def test_summarize_source_prevalence_mismatch_raises():
    """Every seed is tested on the same test file, so n and n_pos must agree across runs."""
    other = _run(44)
    other["metrics"] = _metrics(44)
    other["metrics"]["per_source"]["ragtruth"]["n_pos"] = 300
    with pytest.raises(ValueError):
        summarize_source(RUNS[:2] + [other], "ragtruth")


def test_summarize_source_no_runs_raises():
    with pytest.raises(ValueError):
        summarize_source([], "ragtruth")


# ---- peak_epoch (TEMPLATE) ----------------------------------------------------

def test_peak_epoch_last():
    assert peak_epoch([{"epoch": 0, "pr_auc": 0.6}, {"epoch": 1, "pr_auc": 0.7}]) == 1


def test_peak_epoch_first():
    assert peak_epoch([{"epoch": 0, "pr_auc": 0.7}, {"epoch": 1, "pr_auc": 0.6}]) == 0


def test_peak_epoch_tie_is_earliest():
    """train.py only saves best.pt on a strict improvement, so on a tie the earlier epoch is the checkpoint."""
    assert peak_epoch([{"epoch": 0, "pr_auc": 0.7}, {"epoch": 1, "pr_auc": 0.7}]) == 0


def test_peak_epoch_uses_select_on():
    hist = [{"epoch": 0, "pr_auc": 0.6, "roc_auc": 0.9}, {"epoch": 1, "pr_auc": 0.7, "roc_auc": 0.8}]
    assert peak_epoch(hist, select_on="roc_auc") == 0


def test_peak_epoch_returns_epoch_field_not_list_position():
    assert peak_epoch([{"epoch": 3, "pr_auc": 0.1}, {"epoch": 7, "pr_auc": 0.9}]) == 7


def test_peak_epoch_empty_raises():
    with pytest.raises(ValueError):
        peak_epoch([])


# ---- truncation_split_metrics (TEMPLATE) --------------------------------------

# RAGTruth test rows idx 0-3 truncated, 4-7 untruncated. group_id repeats in pairs (not unique, like RAGTruth).
T_LABELS, T_PROBS = [1, 0, 1, 0], [0.9, 0.2, 0.4, 0.6]
U_LABELS, U_PROBS = [1, 0, 0, 0], [0.8, 0.3, 0.7, 0.1]


def _preds():
    preds = []
    for i, (y, p) in enumerate(zip(T_LABELS + U_LABELS, T_PROBS + U_PROBS)):
        preds.append({"split": "test", "idx": i, "source": "ragtruth", "task": "QA",
                      "group_id": f"g{i // 2}", "label": y, "prob": p})
    # rows that must be ignored: a HaluEval test row and a RAGTruth VALIDATION row
    preds.append({"split": "test", "idx": 8, "source": "halueval", "task": "QA", "group_id": "h0", "label": 1, "prob": 0.9})
    preds.append({"split": "val", "idx": 0, "source": "ragtruth", "task": "QA", "group_id": "v0", "label": 0, "prob": 0.95})
    return preds


def _flags():
    flags = [{"split": "test", "idx": i, "source": "ragtruth", "group_id": f"g{i // 2}", "ctx_truncated": i < 4}
             for i in range(8)]
    flags.append({"split": "test", "idx": 8, "source": "halueval", "group_id": "h0", "ctx_truncated": False})
    # same idx as test row 0 but a different split: a join on idx alone would pick this up
    flags.append({"split": "val", "idx": 0, "source": "ragtruth", "group_id": "v0", "ctx_truncated": False})
    return flags


def _expected(labels, probs, thr):
    y, p = np.array(labels), np.array(probs)
    out = ranking_metrics(y, p)
    out.update(thresholded_metrics(y, p, thr))
    return out


def test_truncation_split_values():
    out = truncation_split_metrics(_preds(), _flags(), threshold=0.5)
    assert set(out) == {"truncated", "untruncated"}
    for name, (labels, probs) in {"truncated": (T_LABELS, T_PROBS), "untruncated": (U_LABELS, U_PROBS)}.items():
        exp = _expected(labels, probs, 0.5)
        got = out[name]
        for k in ("pr_auc", "roc_auc", "f1"):
            assert got[k] == pytest.approx(exp[k]), (name, k)
        assert got["n"] == 4
        assert got["n_pos"] == sum(labels)
        assert got["prevalence"] == pytest.approx(sum(labels) / 4)


def test_truncation_split_uses_given_threshold():
    out = truncation_split_metrics(_preds(), _flags(), threshold=0.35)
    assert out["truncated"]["f1"] == pytest.approx(_expected(T_LABELS, T_PROBS, 0.35)["f1"])


def test_truncation_split_ignores_other_sources_and_splits():
    out = truncation_split_metrics(_preds(), _flags(), threshold=0.5)
    assert out["truncated"]["n"] + out["untruncated"]["n"] == 8


def test_truncation_split_missing_flag_raises():
    flags = [f for f in _flags() if not (f["split"] == "test" and f["idx"] == 5)]
    with pytest.raises(ValueError):
        truncation_split_metrics(_preds(), flags, threshold=0.5)


def test_truncation_split_missing_prediction_raises():
    """A RAGTruth test row has a flag but no prediction (e.g. a partly downloaded predictions.jsonl).
    idx 5 is chosen so both subsets still have both classes: the raise must come from the coverage
    check, not from the no-positives check."""
    preds = [p for p in _preds() if not (p["split"] == "test" and p["source"] == "ragtruth" and p["idx"] == 5)]
    with pytest.raises(ValueError):
        truncation_split_metrics(preds, _flags(), threshold=0.5)


def test_truncation_split_shifted_idx_raises():
    """Flags shifted by one row: (split, idx) still joins, but group_id no longer agrees."""
    flags = []
    for f in _flags():
        if f["split"] == "test" and f["source"] == "ragtruth":
            f = {**f, "group_id": f"g{(f['idx'] - 1) // 2}" if f["idx"] > 0 else "g_none"}
        flags.append(f)
    with pytest.raises(ValueError):
        truncation_split_metrics(_preds(), flags, threshold=0.5)


def test_truncation_split_source_mismatch_raises():
    flags = [{**f, "source": "halueval"} if (f["split"] == "test" and f["idx"] == 2) else f for f in _flags()]
    with pytest.raises(ValueError):
        truncation_split_metrics(_preds(), flags, threshold=0.5)


def test_truncation_split_subset_without_positives_raises():
    preds = [{**p, "label": 0} if (p["split"] == "test" and p["source"] == "ragtruth" and p["idx"] < 4) else p
             for p in _preds()]
    with pytest.raises(ValueError):
        truncation_split_metrics(preds, _flags(), threshold=0.5)


# ---- plumbing: load_runs / filter_runs / format_report -----------------------

def _snapshot(seed, experiment="G", max_length=1024, data="data/unified_v2"):
    return {
        "config": {"experiment": experiment, "model": {"max_length": max_length},
                   "data": {"train_path": f"{data}/train.jsonl", "val_path": f"{data}/val.jsonl",
                            "test_path": f"{data}/test.jsonl"},
                   "train": {"seed": seed, "select_on": "pr_auc", "out_dir": "/kaggle/working/runs"}},
        "seed": seed, "git_sha": "abc1234", "git_dirty": False, "device_name": "Tesla T4",
        "experiment": experiment, "input_format": "Question: {query} Context: {context}",
    }


def _write_run(root, name, seed, snapshot=None, with_metrics=True):
    d = root / name
    d.mkdir(parents=True)
    (d / "config_snapshot.json").write_text(json.dumps(snapshot or _snapshot(seed)))
    if with_metrics:
        (d / "metrics.json").write_text(json.dumps(_metrics(seed)))
    (d / "val_history.json").write_text(json.dumps([{"epoch": 0, "pr_auc": 0.6}, {"epoch": 1, "pr_auc": 0.7}]))
    return d


@pytest.fixture
def root(tmp_path):
    for s in (42, 43, 44):
        _write_run(tmp_path, f"run_seed{s}_20261110_000000", s)
    return tmp_path


def test_load_runs_reads_all_files(root):
    runs = load_runs(str(root))
    assert [r["name"] for r in runs] == [f"run_seed{s}_20261110_000000" for s in (42, 43, 44)]
    for r in runs:
        assert r["metrics"]["per_source"]["ragtruth"]["n"] == 1800
        assert r["snapshot"]["experiment"] == "G"
        assert len(r["val_history"]) == 2


def test_load_runs_skips_folders_without_metrics(root):
    _write_run(root, "run_seed45_crashed", 45, with_metrics=False)
    assert len(load_runs(str(root))) == 3


def test_filter_runs_drops_invalid(root):
    smoke = _snapshot(42)
    for k in ("train_path", "val_path", "test_path"):
        smoke["config"]["data"][k] = "data/unified_v2/tiny.jsonl"
    _write_run(root, "run_seed42_smoke", 42, snapshot=smoke)
    _write_run(root, "run_seed42_short", 42, snapshot=_snapshot(42, max_length=256))
    _write_run(root, "run_seed42_C", 42, snapshot=_snapshot(42, experiment="C"))
    _write_run(root, "run_seed42_v1", 42, snapshot=_snapshot(42, data="data/unified"))
    kept, dropped = filter_runs(load_runs(str(root)), experiment="G")
    assert sorted(r["name"] for r in kept) == [f"run_seed{s}_20261110_000000" for s in (42, 43, 44)]
    assert {name for name, _ in dropped} == {"run_seed42_smoke", "run_seed42_short", "run_seed42_C", "run_seed42_v1"}
    assert all(reasons for _, reasons in dropped)


def test_format_report_lists_runs_and_never_overall(root):
    runs = load_runs(str(root))
    text = format_report(runs, sources=["ragtruth", "halueval"])
    for r in runs:
        assert r["name"] in text
    assert "overall" not in text.lower()
    assert "0.99" not in text                         # the blended number
    assert "20.2%" in text                            # RAGTruth prevalence shown beside PR-AUC
