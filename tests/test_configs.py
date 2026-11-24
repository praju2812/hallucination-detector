"""C and G+C configs (Phase 5, RQ2): exactly G's settings; only data, experiment and selection keys differ.

Two kinds of training setting, and which applies to each:
- pos_weight is a FIXED value. base.yaml's 2.3 goes straight to src.model.build_loss and is never recomputed
  from the neg/pos ratio of the training set. So G, C and G+C must hold the same number.
- Warmup is a RULE. warmup_steps = int(warmup_ratio * steps_per_epoch * epochs) (src.train.build_optimizer_and_scheduler).
  The ratio (0.06) must be identical; the step count then scales with the training-set size.
src.train.resolved_training_values records the resolved numbers in config_snapshot.json["resolved"].

Run: python -m pytest tests/test_configs.py
"""

import copy
import math
import os

import pytest
import yaml

from src.train import resolved_training_values


def _load(path):
    with open(path) as f:
        return yaml.safe_load(f)


BASE = _load("configs/base.yaml")
C = _load("configs/c.yaml")
GC = _load("configs/gc.yaml")

GEN_TRAIN = "data/unified_v2/train.jsonl"
CLIN_TRAIN = "data/clinical/train.jsonl.gz"
VAL = ["data/clinical/val.jsonl", "data/unified_v2/val.jsonl"]     # clinical first: its idx = file position
TEST_SETS = {"test": "data/unified_v2/test.jsonl", "test_full": "data/clinical/test_full.jsonl"}
SELECT = ["ragbench_pubmedqa", "ragbench_covidqa"]                 # pooled RAGBench clean val; never MedHallu


def _strip(cfg):
    """Everything except the keys Phase 5 is allowed to change."""
    c = copy.deepcopy(cfg)
    c.pop("experiment")
    for k in ("train_path", "val_path", "test_path", "test_sets"):
        c["data"].pop(k, None)
    c["train"].pop("select_sources", None)
    return c


@pytest.mark.parametrize("cfg", [C, GC], ids=["C", "G+C"])
def test_every_other_setting_equals_base(cfg):
    # model name, fit_first, max_length 1024, resp_max 384, dropout, columns, epochs, batch, lr, weight decay,
    # warmup_ratio, grad norm, pos_weight, seed default, select_on, out_dir
    assert _strip(cfg) == _strip(BASE)


def test_experiment_names():
    assert (BASE["experiment"], C["experiment"], GC["experiment"]) == ("G", "C", "G+C")


def test_c_data_paths():
    d = C["data"]
    assert d["train_path"] == CLIN_TRAIN
    assert d["val_path"] == VAL
    assert d["test_sets"] == TEST_SETS
    assert "test_path" not in d          # one way of naming the test files per config, never both


def test_gc_data_paths():
    d = GC["data"]
    assert d["train_path"] == [GEN_TRAIN, CLIN_TRAIN]   # general first, then clinical
    assert d["val_path"] == VAL
    assert d["test_sets"] == TEST_SETS
    assert "test_path" not in d


def test_selection_sources():
    assert C["train"]["select_sources"] == SELECT
    assert GC["train"]["select_sources"] == SELECT
    assert "select_sources" not in BASE["train"]          # G keeps the default, ["ragtruth"]
    assert "medhallu" not in SELECT


@pytest.mark.parametrize("cfg", [C, GC], ids=["C", "G+C"])
def test_every_path_exists(cfg):
    d = cfg["data"]
    paths = [d["train_path"]] if isinstance(d["train_path"], str) else d["train_path"]
    paths += d["val_path"] + list(d["test_sets"].values())
    for p in paths:
        assert os.path.exists(p), p


@pytest.mark.parametrize("cfg", [BASE, C, GC], ids=["G", "C", "G+C"])
def test_pos_weight_is_the_same_fixed_value(cfg):
    """FIXED value, not a rule: identical in all three configs even though the training prevalences differ."""
    assert cfg["train"]["pos_weight"] == 2.3


@pytest.mark.parametrize("cfg", [BASE, C, GC], ids=["G", "C", "G+C"])
def test_warmup_is_the_same_rule(cfg):
    """RULE, not a value: the same ratio and epochs; the number of warmup steps follows the data size."""
    assert cfg["train"]["warmup_ratio"] == 0.06
    assert cfg["train"]["epochs"] == 2
    assert cfg["train"]["batch_size"] == 8


@pytest.mark.parametrize("name,n_rows,warmup", [
    ("G", 25_008, 375),      # 3,126 steps/epoch -> 6,252 total -> int(375.12)
    ("C", 37_114, 556),      # 4,640 -> 9,280 -> int(556.8)
    ("G+C", 62_122, 931),    # 7,766 -> 15,532 -> int(931.92)
])
def test_resolved_values(name, n_rows, warmup):
    steps = math.ceil(n_rows / BASE["train"]["batch_size"])     # DataLoader without drop_last
    r = resolved_training_values(BASE, n_train_rows=n_rows, steps_per_epoch=steps)
    assert r["n_train_rows"] == n_rows
    assert r["steps_per_epoch"] == steps
    assert r["total_steps"] == 2 * steps
    assert r["warmup_steps"] == warmup
    assert r["pos_weight"] == 2.3
    assert r["warmup_ratio"] == 0.06
    assert "fixed" in r["pos_weight_rule"]
    assert "warmup_ratio" in r["warmup_rule"]
