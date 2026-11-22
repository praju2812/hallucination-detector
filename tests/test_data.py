"""HallucinationDataset passes the query column through (Phase 1).

Run: python -m pytest tests/test_data.py
"""

import json

import pytest

from src.data import HallucinationDataset

CFG_DATA = {"query_col": "query", "context_col": "context", "response_col": "answer",
            "label_col": "label", "dataset_col": "source"}


@pytest.fixture
def path(tmp_path):
    rows = [
        {"query": "who sat?", "context": "the cat sat", "answer": "a dog", "label": 1, "source": "ragtruth"},
        {"query": "boiling point", "context": "water boils at 100", "answer": "50", "label": 0, "source": "halueval"},
    ]
    p = tmp_path / "rows.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(p)


def test_item_contains_query(path):
    ds = HallucinationDataset(path, CFG_DATA)
    item = ds[0]
    assert item == {"query": "who sat?", "context": "the cat sat", "response": "a dog",
                    "label": 1.0, "source": "ragtruth"}


def test_query_col_is_configurable(tmp_path):
    p = tmp_path / "rows.jsonl"
    p.write_text(json.dumps({"question": "q?", "context": "c", "answer": "a", "label": 0, "source": "x"}) + "\n")
    ds = HallucinationDataset(str(p), {**CFG_DATA, "query_col": "question"})
    assert ds[0]["query"] == "q?"


def test_missing_query_col_in_config_fails(path):
    """No silent fallback to the old input format."""
    cfg = {k: v for k, v in CFG_DATA.items() if k != "query_col"}
    with pytest.raises(KeyError):
        HallucinationDataset(path, cfg)


def test_sources_filter_still_works(path):
    ds = HallucinationDataset(path, CFG_DATA, sources=["halueval"])
    assert len(ds) == 1 and ds[0]["query"] == "boiling point"


# ---- Phase 5: several files (G+C trains on unified_v2 train + clinical train.jsonl.gz) -------------

def _clinical_gz(tmp_path):
    from utils.io import write_jsonl
    rows = [
        {"query": "q-a", "context": "abstract a", "answer": "yes", "label": 0, "source": "ragbench_pubmedqa"},
        {"query": "q-b", "context": "abstract b", "answer": "no", "label": 1, "source": "medhallu"},
    ]
    p = tmp_path / "clinical" / "train.jsonl.gz"
    write_jsonl(rows, str(p))
    return str(p)


def test_list_of_paths_concatenates_in_order(path, tmp_path):
    gz = _clinical_gz(tmp_path)
    ds = HallucinationDataset([path, gz], CFG_DATA)
    assert len(ds) == 4
    assert [ds[i]["source"] for i in range(4)] == ["ragtruth", "halueval", "ragbench_pubmedqa", "medhallu"]
    assert ds.rows[2]["context"] == "abstract a"          # .gz read like plain jsonl


def test_list_order_is_kept(path, tmp_path):
    gz = _clinical_gz(tmp_path)
    ds = HallucinationDataset([gz, path], CFG_DATA)
    assert [r["source"] for r in ds.rows] == ["ragbench_pubmedqa", "medhallu", "ragtruth", "halueval"]


def test_single_path_string_unchanged(path):
    assert HallucinationDataset(path, CFG_DATA).rows == HallucinationDataset([path], CFG_DATA).rows


def test_sources_filter_applies_across_files(path, tmp_path):
    gz = _clinical_gz(tmp_path)
    ds = HallucinationDataset([path, gz], CFG_DATA, sources=["ragtruth", "medhallu"])
    assert [r["source"] for r in ds.rows] == ["ragtruth", "medhallu"]


def test_empty_path_list_fails():
    with pytest.raises(ValueError):
        HallucinationDataset([], CFG_DATA)
