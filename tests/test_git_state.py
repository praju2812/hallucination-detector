"""Git sha / dirty flag recorded by the report and scoring scripts: read ONCE, before anything is written.

The bug: a script that writes a tracked output (e.g. results/rq1/<run>/metrics.json) and only then reads
`git status` records dirty=True for everything after the first write, even from a clean checkout.
Here the fake `get_git_dirty` returns True as soon as any "tracked" file differs from its committed content,
which is exactly what rewriting a tracked file does; the fake `get_git_sha` counts how often git is read.

Run: python -m pytest tests/test_git_state.py
"""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import scripts.report_nli as report_nli
import scripts.report_rq1 as report_rq1
import scripts.report_rq2 as report_rq2
import scripts.score_nli as score_nli
import scripts.score_rq1 as score_rq1
import src.train as train
from src.train import git_state

SHA = "abc1234"
ORIGINAL = "committed content\n"


class FakeGit:
    """dirty() is True once any tracked file no longer holds ORIGINAL; sha() counts reads."""

    def __init__(self, tracked):
        self.tracked = [str(p) for p in tracked]
        for p in self.tracked:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                f.write(ORIGINAL)
        self.sha_calls = 0
        self.dirty_calls = 0

    def sha(self):
        self.sha_calls += 1
        return SHA

    def dirty(self):
        self.dirty_calls += 1
        for p in self.tracked:
            with open(p) as f:
                if f.read() != ORIGINAL:
                    return True
        return False


@pytest.fixture
def fake_git(monkeypatch):
    """Install a FakeGit everywhere a script could read git from (src.train and any name a script imported)."""
    def install(tracked):
        g = FakeGit(tracked)
        for mod in (train, report_rq1, report_rq2, report_nli, score_rq1, score_nli):
            monkeypatch.setattr(mod, "get_git_sha", g.sha, raising=False)
            monkeypatch.setattr(mod, "get_git_dirty", g.dirty, raising=False)
        return g
    return install


def test_git_state_reads_both(fake_git, tmp_path):
    g = fake_git([tmp_path / "t.json"])
    assert git_state() == {"git_sha": SHA, "git_dirty": False}
    assert (g.sha_calls, g.dirty_calls) == (1, 1)


def test_compute_seed_metrics_requires_report_git():
    with pytest.raises(TypeError):
        report_rq1.compute_seed_metrics({}, [], 10, 0)


# ---- report_rq1: several seeds, each writing tracked files ---------------------------------------

def test_report_rq1_one_reading_for_every_seed(fake_git, tmp_path):
    from tests.test_rq1_plumbing import _make_toy
    seeds = (42, 43, 44)
    data, rq1, runs = _make_toy(tmp_path, seeds=seeds)
    md = tmp_path / "rq1_G.md"
    tracked = [rq1 / f"run_seed{s}_x" / "metrics.json" for s in seeds] + [rq1 / f"run_seed{s}_x.json" for s in seeds]
    g = fake_git(tracked + [md])
    rc = report_rq1.main(["--rq1-root", str(rq1), "--runs-root", str(runs), "--data-dir", str(data),
                          "--out-md", str(md), "--n-boot", "20"])
    assert rc == 0
    for s in seeds:
        m = json.loads((rq1 / f"run_seed{s}_x" / "metrics.json").read_text())
        assert m["report_git_dirty"] is False, f"seed {s} recorded dirty after earlier seeds were written"
        assert m["report_git_sha"] == SHA
    assert (g.sha_calls, g.dirty_calls) == (1, 1)


# ---- report_rq2: one reading shared by every run ----------------------------------------------------

def test_report_rq2_one_reading_for_every_run(fake_git, tmp_path):
    from tests.test_report_rq2 import toy as toy_fixture
    toy = toy_fixture.__wrapped__(tmp_path)
    runs = sorted(p.name for p in toy["runs"].iterdir())
    tracked = [toy["out_root"] / r / "metrics.json" for r in runs] + [toy["md"]]
    g = fake_git(tracked)
    assert report_rq2.main(toy["args"]) == 0
    for r in runs:
        rec = json.loads((toy["out_root"] / r / "metrics.json").read_text())
        assert (rec["report_git_sha"], rec["report_git_dirty"]) == (SHA, False)
    assert (g.sha_calls, g.dirty_calls) == (1, 1)


# ---- score_rq1: predictions.jsonl is rewritten before scoring.json ------------------------------------

def _clin_rows(n=4):
    return [{"query": "q", "context": "c", "answer": f"a{i}", "label": i % 2, "source": "ragbench_pubmedqa",
             "task": "QA", "group_id": f"g{i}"} for i in range(n)]


def test_score_rq1_scoring_json_uses_the_start_reading(fake_git, tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    for fname in score_rq1.CLINICAL_SPLITS.values():
        (data / fname).write_text("".join(json.dumps(r) + "\n" for r in _clin_rows()))
    run_dir = tmp_path / "runs" / "run_seed42_x"
    run_dir.mkdir(parents=True)
    (run_dir / "best.pt").write_bytes(b"not a real checkpoint")
    snapshot = {"seed": 42, "experiment": "G", "git_sha": SHA,
                "config": {"model": {"name": "m", "truncation": "fit_first"}, "train": {"batch_size": 2}}}

    monkeypatch.setattr(score_rq1, "load_g_checkpoint", lambda d, dev: (nn.Linear(1, 1), snapshot))
    monkeypatch.setattr(score_rq1, "build_tokenizer", lambda name: None)
    monkeypatch.setattr(score_rq1, "make_eval_loader",
                        lambda path, cfg, tok, bs, keep=None: SimpleNamespace(dataset=SimpleNamespace(rows=_clin_rows())))
    monkeypatch.setattr(score_rq1, "probe_precision", lambda m, l, d: {"param_dtype": "torch.float32"})
    monkeypatch.setattr(score_rq1, "score_loader", lambda m, loader, split, cfg, dev: (
        [{"split": split, "idx": i, "source": r["source"], "label": r["label"], "prob": 0.5}
         for i, r in enumerate(loader.dataset.rows)], 0.1))

    out = tmp_path / "rq1"
    pred_path = out / "run_seed42_x" / "predictions.jsonl"
    g = fake_git([pred_path])                                   # a tracked predictions file, about to be overwritten
    rc = score_rq1.main(["--run-dir", str(run_dir), "--data-dir", str(data), "--out-root", str(out),
                         "--device", "cpu", "--overwrite"])
    assert rc == 0
    assert pred_path.read_text() != ORIGINAL                    # it really was rewritten
    sc = json.loads((out / "run_seed42_x" / "scoring.json").read_text())
    assert (sc["git_sha"], sc["git_dirty"]) == (SHA, False)
    assert (g.sha_calls, g.dirty_calls) == (1, 1)


# ---- score_nli: same order (predictions first, scoring.json second) ----------------------------------------

class FakeNLI(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(1, 1)
        self.config = SimpleNamespace(id2label={0: "entailment", 1: "neutral", 2: "contradiction"}, type_vocab_size=0)


def test_score_nli_scoring_json_uses_the_start_reading(fake_git, tmp_path, monkeypatch):
    import transformers
    split = next(iter(score_nli.SPLITS))
    data_file = tmp_path / "split.jsonl"
    data_file.write_text("x\n")
    rows = _clin_rows(3)

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(transformers.AutoModelForSequenceClassification, "from_pretrained", lambda *a, **k: FakeNLI())
    monkeypatch.setattr(score_nli, "cached_commit", lambda *a, **k: "rev")
    monkeypatch.setattr(score_nli, "resolve_revision", lambda *a, **k: "rev")
    monkeypatch.setattr(score_nli, "PairBuilder", lambda tok: SimpleNamespace(verify=lambda: {"layout": "ok"}))
    monkeypatch.setattr(score_nli, "load_split", lambda name, g, c: {
        "rows": rows, "idx": list(range(3)), "keys": [f"k{i}" for i in range(3)], "path": str(data_file),
        "source_filter": None, "rows_in_file": 3})
    monkeypatch.setattr(score_nli, "prepare_row", lambda *a, **k: {"spans": [[(0, 1)]]})
    monkeypatch.setattr(score_nli, "split_counts", lambda sp, prepared: {})
    monkeypatch.setattr(score_nli, "score_prepared", lambda prepared, *a, **k: (
        [{"prob": 0.5} for _ in prepared], {"pairs": len(prepared), "seconds": 0.1}))
    monkeypatch.setattr(score_nli, "make_records", lambda sp, results: [
        {"split": split, "idx": i, "prob": r["prob"]} for i, r in enumerate(results)])
    monkeypatch.setattr(score_nli, "determinism_check", lambda *a, **k: {
        "n": 1, "max_abs_diff": 0.0, "tolerance": 1e-4, "ranking_unchanged": True, "passed": True})

    out = tmp_path / "nli"
    pred_path = out / score_nli.model_dir_name(score_nli.MODEL_ID) / "predictions.jsonl"
    g = fake_git([pred_path])
    rc = score_nli.main(["--splits", split, "--out-root", str(out), "--device", "cpu", "--overwrite"])
    assert rc == 0
    assert pred_path.read_text() != ORIGINAL
    sc = json.loads((pred_path.parent / "scoring.json").read_text())
    assert (sc["git_sha"], sc["git_dirty"]) == (SHA, False)
    assert (g.sha_calls, g.dirty_calls) == (1, 1)
