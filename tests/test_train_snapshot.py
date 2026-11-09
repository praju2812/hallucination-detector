"""config_snapshot.json contents (hard rule 8, plumbing).

Run: python -m pytest tests/test_train_snapshot.py
"""

import copy
import json

import pytest
import torch
import transformers
import yaml

from src.encoding import INPUT_FORMAT
from src.train import build_run_snapshot

with open("configs/base.yaml") as f:
    BASE = yaml.safe_load(f)


def test_snapshot_records_required_fields():
    cfg = copy.deepcopy(BASE)
    snap = build_run_snapshot(cfg)
    assert snap["config"] == cfg                       # data paths included
    assert snap["seed"] == cfg["train"]["seed"]
    assert snap["experiment"] == cfg["experiment"]
    assert snap["input_format"] == INPUT_FORMAT
    assert snap["torch_version"] == torch.__version__
    assert snap["transformers_version"] == transformers.__version__
    assert "git_sha" in snap
    json.dumps(snap)  # must be serialisable


def test_snapshot_seed_follows_cli_override():
    cfg = copy.deepcopy(BASE)
    cfg["train"]["seed"] = 43
    assert build_run_snapshot(cfg)["seed"] == 43


def test_base_config_points_at_v2_with_query_col():
    assert BASE["experiment"] == "G"
    assert BASE["data"]["query_col"] == "query"
    for k in ("train_path", "val_path", "test_path"):
        assert BASE["data"][k].startswith("data/unified_v2/")
    assert BASE["model"]["max_length"] == 1024


@pytest.mark.parametrize("bad", [None, "", "g", "G+C+X"])
def test_snapshot_rejects_unknown_experiment(bad):
    cfg = copy.deepcopy(BASE)
    if bad is None:
        del cfg["experiment"]
    else:
        cfg["experiment"] = bad
    with pytest.raises(ValueError):
        build_run_snapshot(cfg)
