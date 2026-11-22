"""Smoke config (Phase 2, decision D1): base.yaml with ONLY the data paths, epochs and out_dir changed.

Run: python -m pytest tests/test_smoke_config.py
"""

import copy

import yaml

from scripts.make_smoke_config import make_smoke_config

with open("configs/base.yaml") as f:
    BASE = yaml.safe_load(f)

TINY = "data/unified_v2/tiny.jsonl"
OUT = "/kaggle/working/smoke"


def _without_changed_keys(cfg):
    c = copy.deepcopy(cfg)
    for k in ("train_path", "val_path", "test_path"):
        c["data"].pop(k)
    c["train"].pop("epochs")
    c["train"].pop("out_dir")
    return c


def test_only_five_keys_change():
    smoke = make_smoke_config(BASE, TINY, OUT)
    assert _without_changed_keys(smoke) == _without_changed_keys(BASE)


def test_changed_values():
    smoke = make_smoke_config(BASE, TINY, OUT)
    for k in ("train_path", "val_path", "test_path"):
        assert smoke["data"][k] == TINY
    assert smoke["train"]["epochs"] == 1
    assert smoke["train"]["out_dir"] == OUT
    # the real settings stay, so GPU memory is tested at reported-run size
    assert smoke["model"]["max_length"] == 1024
    assert smoke["train"]["batch_size"] == BASE["train"]["batch_size"]


def test_base_is_not_mutated():
    before = copy.deepcopy(BASE)
    make_smoke_config(BASE, TINY, OUT)
    assert BASE == before


def test_written_file_round_trips(tmp_path):
    from scripts.make_smoke_config import main
    out = tmp_path / "smoke.yaml"
    main(["--base", "configs/base.yaml", "--out", str(out), "--data", TINY, "--out-dir", OUT])
    with open(out) as f:
        assert yaml.safe_load(f) == make_smoke_config(BASE, TINY, OUT)


# ---- Phase 5: C and G+C smoke configs -----------------------------------------------------------
# Each path keeps its family: unified_v2 paths -> data/unified_v2/tiny.jsonl, clinical paths -> a clinical tiny file
# written at smoke time (10 positive + 10 negative rows per source from clinical val). Nothing goes into data/clinical/.

import collections
import json

import pytest

from scripts.make_smoke_config import write_clinical_tiny

with open("configs/c.yaml") as f:
    C = yaml.safe_load(f)
with open("configs/gc.yaml") as f:
    GC = yaml.safe_load(f)

CLIN_TINY = "/kaggle/working/smoke/clinical_tiny.jsonl"


def _without_paths(cfg):
    c = copy.deepcopy(cfg)
    for k in ("train_path", "val_path", "test_path", "test_sets"):
        c["data"].pop(k, None)
    c["train"].pop("epochs")
    c["train"].pop("out_dir")
    return c


def test_c_smoke_maps_each_path_to_its_tiny_file():
    s = make_smoke_config(C, TINY, OUT, clinical_tiny=CLIN_TINY)
    assert s["data"]["train_path"] == CLIN_TINY
    assert s["data"]["val_path"] == [CLIN_TINY, TINY]
    assert s["data"]["test_sets"] == {"test": TINY, "test_full": CLIN_TINY}
    assert "test_path" not in s["data"]
    assert s["train"]["epochs"] == 1 and s["train"]["out_dir"] == OUT
    assert _without_paths(s) == _without_paths(C)      # experiment, select_sources, model settings unchanged


def test_gc_smoke_keeps_both_training_files():
    s = make_smoke_config(GC, TINY, OUT, clinical_tiny=CLIN_TINY)
    assert s["data"]["train_path"] == [TINY, CLIN_TINY]
    assert s["experiment"] == "G+C"
    assert _without_paths(s) == _without_paths(GC)


def test_clinical_path_without_clinical_tiny_fails():
    with pytest.raises(ValueError):
        make_smoke_config(C, TINY, OUT)


def test_g_smoke_ignores_clinical_tiny():
    assert make_smoke_config(BASE, TINY, OUT, clinical_tiny=CLIN_TINY) == make_smoke_config(BASE, TINY, OUT)


def test_clinical_tiny_rows(tmp_path):
    out = tmp_path / "clinical_tiny.jsonl"
    rows = write_clinical_tiny("data/clinical/val.jsonl", str(out))
    assert len(rows) == 60
    counts = collections.Counter((r["source"], int(r["label"])) for r in rows)
    assert counts == {(s, y): 10 for s in ("medhallu", "ragbench_covidqa", "ragbench_pubmedqa") for y in (0, 1)}
    with open("data/clinical/val.jsonl") as f:
        val = [json.loads(l) for l in f]
    assert all(r in val for r in rows)                  # a subset of clinical val, rows unchanged
    with open(out) as f:
        assert [json.loads(l) for l in f] == rows


def test_clinical_tiny_is_deterministic(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_clinical_tiny("data/clinical/val.jsonl", str(a))
    write_clinical_tiny("data/clinical/val.jsonl", str(b))
    assert a.read_bytes() == b.read_bytes()


def test_main_writes_clinical_tiny_for_gc(tmp_path):
    from scripts.make_smoke_config import main
    out, out_dir = tmp_path / "smoke.yaml", tmp_path / "smoke"
    main(["--base", "configs/gc.yaml", "--out", str(out), "--out-dir", str(out_dir)])
    clin = str(out_dir / "clinical_tiny.jsonl")
    with open(out) as f:
        cfg = yaml.safe_load(f)
    assert cfg == make_smoke_config(GC, TINY, str(out_dir), clinical_tiny=clin)
    with open(clin) as f:
        assert sum(1 for _ in f) == 60


def test_main_writes_no_clinical_tiny_for_g(tmp_path):
    from scripts.make_smoke_config import main
    out, out_dir = tmp_path / "smoke.yaml", tmp_path / "smoke"
    main(["--base", "configs/base.yaml", "--out", str(out), "--out-dir", str(out_dir)])
    assert not (out_dir / "clinical_tiny.jsonl").exists()
