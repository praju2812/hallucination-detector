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
