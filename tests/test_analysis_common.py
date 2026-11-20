"""Output folder and default inputs of the supervisor-analysis scripts (scripts/analysis/, plumbing).

The 11 Nov output (results/analysis_2026-11-11/) is a dated record: no analysis script may write into it again.

Run: python -m pytest tests/test_analysis_common.py
"""

import json
from datetime import date

import pytest

import scripts.analysis.a_lengths as a_lengths
import scripts.analysis.c_summary_length_cue as c_cue
from scripts.analysis.common import FROZEN_OUT_DIR, REPO, RUNS_GLOB, check_out_dir, default_out_dir, write_json


def test_default_out_dir_is_dated_today():
    assert default_out_dir() == REPO / "results" / f"analysis_{date.today().isoformat()}"
    assert check_out_dir(None) == default_out_dir().resolve()


@pytest.mark.parametrize("bad", [
    FROZEN_OUT_DIR,
    FROZEN_OUT_DIR / "sub",
    "results/analysis_2026-11-11",            # relative to the repo root (tests run from there)
    "results/analysis_2026-11-11/",
    "results/../results/analysis_2026-11-11",
])
def test_frozen_out_dir_refused(bad):
    with pytest.raises(ValueError):
        check_out_dir(bad)


def test_write_json_refuses_frozen_dir_and_writes_elsewhere(tmp_path):
    with pytest.raises(ValueError):
        write_json({"a": 1}, "x.json", FROZEN_OUT_DIR)
    assert not (FROZEN_OUT_DIR / "x.json").exists()
    path = write_json({"a": 1}, "x.json", tmp_path / "out")
    assert path == (tmp_path / "out" / "x.json").resolve()
    assert json.loads(path.read_text()) == {"a": 1}


def test_runs_glob_points_at_repo_runs():
    assert RUNS_GLOB == str(REPO / "runs" / "run_seed*" / "predictions.jsonl")
    assert "hd-phase2" not in RUNS_GLOB


def test_flags_default_points_at_repo_results():
    assert a_lengths.DEFAULT_FLAGS == str(REPO / "results" / "truncation_flags_v2.jsonl")


@pytest.mark.parametrize("module", [a_lengths, c_cue])
def test_scripts_refuse_frozen_out_dir_before_any_work(module, monkeypatch):
    """--out-dir results/analysis_2026-11-11 stops at argument checking, before any data is read."""
    monkeypatch.setattr("sys.argv", ["x", "--out-dir", str(FROZEN_OUT_DIR)])

    def boom(*a, **k):
        raise AssertionError("work started before the out-dir check")
    monkeypatch.setattr(module, "load_unified_v2", boom)
    if hasattr(module, "load_config"):
        monkeypatch.setattr(module, "load_config", boom)
    with pytest.raises(ValueError):
        module.main()
