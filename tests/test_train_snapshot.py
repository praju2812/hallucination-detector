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
from src.train import apply_cli_overrides, build_run_snapshot, check_out_dir, is_kaggle, parse_args

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


# ---- Phase 2 plumbing -------------------------------------------------------

def test_snapshot_records_git_dirty_and_device_name():
    snap = build_run_snapshot(copy.deepcopy(BASE))
    assert snap["git_dirty"] in (True, False, None)    # None = git not available
    assert isinstance(snap["device_name"], str) and snap["device_name"]
    json.dumps(snap)


def test_cli_out_dir_and_seed_override_reach_snapshot():
    cfg = copy.deepcopy(BASE)
    args = parse_args(["--seed", "44", "--out-dir", "/kaggle/working/runs"])
    apply_cli_overrides(cfg, args)
    snap = build_run_snapshot(cfg)
    assert snap["seed"] == 44
    assert snap["config"]["train"]["out_dir"] == "/kaggle/working/runs"


def test_cli_without_flags_keeps_config_values():
    cfg = copy.deepcopy(BASE)
    apply_cli_overrides(cfg, parse_args([]))
    assert cfg == BASE


@pytest.mark.parametrize("out_dir", ["/kaggle/working/runs", "/kaggle/working/smoke", "/kaggle/working"])
def test_check_out_dir_on_kaggle_accepts_absolute_working_paths(out_dir):
    check_out_dir(out_dir, on_kaggle=True)


@pytest.mark.parametrize("out_dir", [
    "checkpoints",                  # base.yaml default: relative, the lost-checkpoint case
    "kaggle/working/runs",          # missing leading "/"
    "/kaggle/input/runs",           # absolute but read-only input area
    "/tmp/runs",                    # absolute but not persisted
    "/kaggle/working_old/runs",     # prefix trick: not under /kaggle/working/
    "/kaggle/working/../input/x",   # escapes /kaggle/working after normalisation
])
def test_check_out_dir_on_kaggle_rejects_other_paths(out_dir):
    with pytest.raises(ValueError):
        check_out_dir(out_dir, on_kaggle=True)


@pytest.mark.parametrize("out_dir", ["checkpoints", "runs/local", "/tmp/runs"])
def test_check_out_dir_off_kaggle_accepts_anything(out_dir):
    check_out_dir(out_dir, on_kaggle=False)


def test_is_kaggle_reads_env_var(monkeypatch):
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    assert is_kaggle() is False
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
    assert is_kaggle() is True
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    assert is_kaggle() is True
