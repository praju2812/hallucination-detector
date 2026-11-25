"""Phase 7: latency and cost (scripts/measure_latency.py).

Sample, hardware and output plumbing run now. Tests marked "needs body" call the templates
time_requests / latency_stats / summarize_latency / cost_per_1000 / monthly_cost.

Run: python -m pytest tests/test_measure_latency.py
"""

import hashlib
import json
import time

import numpy as np
import pytest

import scripts.measure_latency as ml
from scripts.measure_latency import (
    SAMPLE_STRATA,
    build_latency_sample,
    cost_per_1000,
    hardware_info,
    latency_stats,
    load_or_create_sample,
    monthly_cost,
    summarize_latency,
    time_requests,
)
from src.rq1 import assign_row_keys


# ---- toy data -----------------------------------------------------------------------------------

def _rows(source, task, n, start=0):
    return [{"source": source, "task": task, "group_id": f"{source}-{task}-{i}", "query": f"q {i}",
             "context": f"context words for row {i}", "answer": f"answer {source} {task} {i}", "label": i % 2}
            for i in range(start, start + n)]


def toy_general():
    return _rows("ragtruth", "qa", 60) + _rows("halueval", "qa", 30) + _rows("ragtruth", "summary", 70)


def toy_clinical():
    return (_rows("ragbench_pubmedqa", "qa", 120) + _rows("ragbench_covidqa", "qa", 105)
            + _rows("medhallu", "qa", 110))


def stratum_of(r):
    return (r["source"], r["task"]) if r["source"] == "ragtruth" else (r["source"], None)


# ---- sample (runs now) --------------------------------------------------------------------------

def test_strata_are_the_decided_design():
    assert SAMPLE_STRATA == {("ragtruth", "qa"): 50, ("ragtruth", "summary"): 50, ("ragbench_pubmedqa", None): 100,
                             ("ragbench_covidqa", None): 100, ("medhallu", None): 100}


def test_sample_counts_exact_and_no_halueval():
    s = build_latency_sample(toy_general(), toy_clinical(), seed=42)
    counts = {}
    for r in s:
        counts[stratum_of(r)] = counts.get(stratum_of(r), 0) + 1
    assert counts == SAMPLE_STRATA
    assert not any(r["source"] == "halueval" for r in s)
    assert len({r["row_key"] for r in s}) == len(s) == 400


def test_sample_rows_keep_the_text_and_label():
    s = build_latency_sample(toy_general(), toy_clinical(), seed=42)
    for r in s:
        for k in ("row_key", "source", "task", "group_id", "query", "context", "answer", "label", "split_file"):
            assert k in r, k


def test_sample_is_sha1_ordered_and_deterministic():
    g, c = toy_general(), toy_clinical()
    a = build_latency_sample(g, c, seed=42)
    assert [r["row_key"] for r in a] == [r["row_key"] for r in build_latency_sample(g, c, seed=42)]
    # within a stratum: the rows with the smallest sha1("seed:row_key"), in that order
    keys = dict(zip([id(r) for r in g], assign_row_keys(g)))
    qa_keys = [keys[id(r)] for r in g if (r["source"], r["task"]) == ("ragtruth", "qa")]
    expected = sorted(qa_keys, key=lambda k: hashlib.sha1(f"42:{k}".encode("utf-8")).hexdigest())[:50]
    got = [r["row_key"] for r in a if stratum_of(r) == ("ragtruth", "qa")]
    assert got == expected


def test_sample_depends_on_seed():
    g, c = toy_general(), toy_clinical()
    a = {r["row_key"] for r in build_latency_sample(g, c, seed=42)}
    b = {r["row_key"] for r in build_latency_sample(g, c, seed=7)}
    assert a != b


def test_sample_short_stratum_raises():
    with pytest.raises(ValueError):
        build_latency_sample(toy_general(), _rows("ragbench_pubmedqa", "qa", 120) + _rows("medhallu", "qa", 110),
                             seed=42)   # no CovidQA rows


def test_sample_is_created_once_then_reused(tmp_path):
    g, c = tmp_path / "g.jsonl", tmp_path / "c.jsonl"
    g.write_text("".join(json.dumps(r) + "\n" for r in toy_general()))
    c.write_text("".join(json.dumps(r) + "\n" for r in toy_clinical()))
    path = tmp_path / "latency" / "sample_seed42.jsonl"
    rows1, info1 = load_or_create_sample(str(path), str(g), str(c), seed=42)
    assert info1["created"] and path.exists()
    # data files change: the saved sample is still used as it is (never resampled)
    c.write_text("".join(json.dumps(r) + "\n" for r in toy_clinical()[::-1]))
    rows2, info2 = load_or_create_sample(str(path), str(g), str(c), seed=42)
    assert not info2["created"]
    assert [r["row_key"] for r in rows1] == [r["row_key"] for r in rows2]
    assert info1["sha1"] == info2["sha1"] == hashlib.sha1(path.read_bytes()).hexdigest()


def test_saved_sample_with_other_seed_is_refused(tmp_path):
    g, c = tmp_path / "g.jsonl", tmp_path / "c.jsonl"
    g.write_text("".join(json.dumps(r) + "\n" for r in toy_general()))
    c.write_text("".join(json.dumps(r) + "\n" for r in toy_clinical()))
    path = tmp_path / "sample.jsonl"
    load_or_create_sample(str(path), str(g), str(c), seed=42)
    with pytest.raises(ValueError):
        load_or_create_sample(str(path), str(g), str(c), seed=43)


# ---- hardware (runs now) ------------------------------------------------------------------------

def test_hardware_info_fields():
    hw = hardware_info("cpu")
    for k in ("device", "platform", "machine", "cpu_model", "logical_cores", "physical_cores",
              "torch_threads", "torch_version", "transformers_version", "python_version", "gpu"):
        assert k in hw, k
    assert hw["device"] == "cpu" and hw["logical_cores"] >= 1


# ---- time_requests (needs body) -----------------------------------------------------------------

def _timed_rows(n):
    return [{"row_key": f"k{i}", "source": "ragtruth" if i % 2 else "medhallu",
             "task": "qa" if i % 2 else "qa", "pause": 0.01 * (1 + i % 3)} for i in range(n)]


def test_time_requests_records_every_row_once_after_warmup():
    rows = _timed_rows(6)
    events = []

    def fn(r):
        events.append(("fn", r["row_key"]))
        time.sleep(r["pause"])
        return {"prob": 0.5, "n_pairs": 3}

    def sync():
        events.append(("sync", None))

    recs = time_requests(fn, rows, warmup=2, sync=sync)
    assert [r["row_key"] for r in recs] == [r["row_key"] for r in rows]        # every row timed once, in order
    assert len([e for e in events if e[0] == "fn"]) == len(rows) + 2              # + 2 untimed warm-up calls
    timed = events[-2 * len(rows):]                                               # after warm-up: fn, sync, fn, sync
    assert [e[0] for e in timed] == ["fn", "sync"] * len(rows)
    for rec, row in zip(recs, rows):
        assert rec["source"] == row["source"] and rec["task"] == row["task"]
        assert row["pause"] <= rec["seconds"] < row["pause"] + 0.05
        assert rec["n_pairs"] == 3


def test_time_requests_n_pairs_none_for_g():
    recs = time_requests(lambda r: {"prob": 0.1}, _timed_rows(3), warmup=1, sync=lambda: None)
    assert all(r["n_pairs"] is None for r in recs)


def test_time_requests_rejects_bad_warmup():
    with pytest.raises(ValueError):
        time_requests(lambda r: {}, _timed_rows(3), warmup=-1, sync=lambda: None)


# ---- latency_stats / summarize_latency (needs body) ---------------------------------------------

def test_latency_stats_by_hand():
    s = latency_stats([1.0, 2.0, 3.0, 4.0])
    assert s["n"] == 4
    assert s["median_s"] == pytest.approx(2.5)
    assert s["p95_s"] == pytest.approx(3.85)            # numpy.percentile, linear interpolation
    assert s["mean_s"] == pytest.approx(2.5)
    assert s["total_s"] == pytest.approx(10.0)
    assert s["throughput_rps"] == pytest.approx(0.4)   # n / total seconds, batch size 1, one request at a time
    with pytest.raises(ValueError):
        latency_stats([])


def test_summarize_latency_groups():
    recs = ([{"row_key": f"a{i}", "source": "ragtruth", "task": "qa", "seconds": 0.1 * (i + 1), "n_pairs": None}
             for i in range(3)]
            + [{"row_key": f"b{i}", "source": "ragtruth", "task": "summary", "seconds": 0.5, "n_pairs": None}
               for i in range(2)]
            + [{"row_key": f"c{i}", "source": "medhallu", "task": "qa", "seconds": 0.2, "n_pairs": None}
               for i in range(4)])
    out = summarize_latency(recs)
    assert set(out) >= {"pooled", "per_source", "per_task", "p95_method"}
    assert out["pooled"] == latency_stats([r["seconds"] for r in recs])
    assert set(out["per_source"]) == {"ragtruth", "medhallu"}
    assert out["per_source"]["medhallu"] == latency_stats([0.2] * 4)
    assert set(out["per_task"]) == {"ragtruth_qa", "ragtruth_summary"}       # RAGTruth only
    assert out["per_task"]["ragtruth_summary"] == latency_stats([0.5, 0.5])


# ---- cost (needs body) --------------------------------------------------------------------------

def test_cost_per_1000_by_hand():
    # $0.35/h, 10 requests/s -> 36,000 requests/h -> $0.35 / 36 per 1,000
    assert cost_per_1000(0.35, 10.0) == pytest.approx(0.35 / 36.0)
    assert cost_per_1000(0.0, 5.0) == 0.0
    for bad in [(0.35, 0.0), (0.35, -1.0), (-0.1, 5.0)]:
        with pytest.raises(ValueError):
            cost_per_1000(*bad)


def test_monthly_cost_by_hand():
    assert monthly_cost(250.0, target_qps=20.0, qps_per_instance=10.0) == pytest.approx(500.0)
    assert monthly_cost(250.0, target_qps=None, qps_per_instance=10.0) is None   # only with a target qps
    with pytest.raises(ValueError):
        monthly_cost(250.0, target_qps=5.0, qps_per_instance=0.0)


# ---- end to end with a fake method (needs bodies) -----------------------------------------------

def test_main_writes_results(tmp_path, monkeypatch):
    g, c = tmp_path / "g.jsonl", tmp_path / "c.jsonl"
    g.write_text("".join(json.dumps(r) + "\n" for r in toy_general()))
    c.write_text("".join(json.dumps(r) + "\n" for r in toy_clinical()))

    def fake_load(method, args):
        return ml.MethodHandle(fn=lambda r: {"prob": 0.2, "label": 0, "threshold": 0.3, "model_id": "fake"},
                               meta={"model_id": "fake", "threshold": 0.3},
                               lengths=lambda r: {"n_tokens": len(r["answer"].split())})

    monkeypatch.setattr(ml, "load_method", fake_load)
    out = tmp_path / "latency"
    rc = ml.main(["--method", "g", "--device", "cpu", "--warmup", "2", "--tag", "test",
                  "--general-test", str(g), "--clinical-test", str(c), "--out", str(out),
                  "--price-per-hour", "0.36", "--price-source", "toy price"])
    assert rc == 0
    res = json.loads((out / "g_test.json").read_text())
    for k in ("method", "hardware", "startup_s", "sample", "warmup", "batch_size", "summary", "cost", "git_sha"):
        assert k in res, k
    assert res["batch_size"] == 1 and res["warmup"] == 2
    assert res["sample"]["n_rows"] == 400
    assert res["cost"]["price_per_hour"] == 0.36 and res["cost"]["price_source"] == "toy price"
    assert res["cost"]["monthly_cost"] is None                                   # no --target-qps given
    recs = [json.loads(l) for l in (out / "g_test_requests.jsonl").read_text().splitlines()]
    assert len(recs) == 400 and all("n_tokens" in r for r in recs)
