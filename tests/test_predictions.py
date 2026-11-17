"""predictions.jsonl rows (Phase 2 plumbing, decision D3).

One line per val/test row: split, idx, source, task, group_id, label, prob.
idx = row position in the split file (group_id is not unique for RAGTruth).

Run: python -m pytest tests/test_predictions.py
"""

import json

import numpy as np
import pytest

from src.train import prediction_records
from utils.io import write_jsonl

CFG_DATA = {"query_col": "query", "context_col": "context", "response_col": "answer",
            "label_col": "label", "dataset_col": "source"}

ROWS = [
    {"query": "q0", "context": "c0", "answer": "a0", "label": 1, "source": "ragtruth", "task": "QA", "group_id": "g1"},
    {"query": "q1", "context": "c1", "answer": "a1", "label": 0, "source": "ragtruth", "task": "QA", "group_id": "g1"},
    {"query": "q2", "context": "c2", "answer": "a2", "label": 0, "source": "halueval", "task": "QA", "group_id": "h7"},
]
LABELS = np.array([1.0, 0.0, 0.0], dtype=np.float32)   # floats, as evaluate() returns them
PROBS = np.array([0.91, 0.12, 0.40], dtype=np.float32)

KEYS = {"split", "idx", "source", "task", "group_id", "label", "prob"}


def test_one_record_per_row_with_required_keys():
    recs = prediction_records("test", ROWS, LABELS, PROBS, CFG_DATA)
    assert len(recs) == len(ROWS)
    for i, r in enumerate(recs):
        assert set(r) == KEYS
        assert r["split"] == "test"
        assert r["idx"] == i
        assert r["source"] == ROWS[i]["source"]
        assert r["group_id"] == ROWS[i]["group_id"]
        assert r["task"] == ROWS[i]["task"]


def test_types_are_plain_python_and_json_serialisable(tmp_path):
    recs = prediction_records("val", ROWS, LABELS, PROBS, CFG_DATA)
    for r in recs:
        assert type(r["label"]) is int
        assert type(r["prob"]) is float
        assert type(r["idx"]) is int
    p = tmp_path / "run" / "predictions.jsonl"
    write_jsonl(recs, str(p))
    lines = p.read_text().splitlines()
    assert len(lines) == 3
    back = json.loads(lines[0])
    assert back["label"] == 1 and abs(back["prob"] - 0.91) < 1e-6


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        prediction_records("test", ROWS, LABELS[:2], PROBS[:2], CFG_DATA)


def test_label_mismatch_raises():
    """If the loader order and the file order disagree, labels will not line up."""
    shifted = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    with pytest.raises(ValueError):
        prediction_records("test", ROWS, shifted, PROBS, CFG_DATA)


def test_rows_without_task_or_group_id_get_none():
    rows = [{"query": "q", "context": "c", "answer": "a", "label": 0, "source": "x"}]
    rec = prediction_records("test", rows, np.array([0.0]), np.array([0.3]), CFG_DATA)[0]
    assert rec["task"] is None and rec["group_id"] is None
