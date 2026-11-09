"""Per-row truncation flags (Phase 2 approval addition 1): report_truncation.py --per-row.

Each line: {split, idx, source, group_id, ctx_truncated}; idx = row position in the split file,
the same definition predictions.jsonl uses, so the two files join on (split, idx).

Run: python -m pytest tests/test_report_truncation.py
"""

import json

from scripts.report_truncation import per_row_record
from utils.io import write_jsonl

CFG_DATA = {"query_col": "query", "context_col": "context", "response_col": "answer",
            "label_col": "label", "dataset_col": "source"}


def test_record_fields():
    row = {"query": "q", "context": "c", "answer": "a", "label": 1, "source": "ragtruth", "task": "QA", "group_id": "g9"}
    stats = {"ctx_trunc": True, "ctx_zero": False, "resp_capped": False, "resp_overflow": False, "q_bare": 3, "q_head": 7}
    rec = per_row_record("test", 12, row, stats, CFG_DATA)
    assert rec == {"split": "test", "idx": 12, "source": "ragtruth", "group_id": "g9", "ctx_truncated": True}


def test_types_and_writer(tmp_path):
    row = {"query": "q", "context": "c", "answer": "a", "label": 0, "source": "halueval", "group_id": "h1"}
    stats = {"ctx_trunc": False}
    rec = per_row_record("val", 0, row, stats, CFG_DATA)
    assert type(rec["ctx_truncated"]) is bool and type(rec["idx"]) is int
    p = tmp_path / "flags.jsonl"
    write_jsonl([rec], str(p))
    assert json.loads(p.read_text()) == rec
