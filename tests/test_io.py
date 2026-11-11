"""utils.io: plain and gzip JSONL (Phase 3 addition 1: files over ~90 MB are committed as .jsonl.gz).

Run: python -m pytest tests/test_io.py
"""

import gzip

from utils.io import read_jsonl, write_jsonl

ROWS = [{"query": "q1", "context": "passage 1:é", "label": 0}, {"query": "q2", "context": "c", "label": 1}]


def test_plain_round_trip(tmp_path):
    p = str(tmp_path / "a" / "x.jsonl")
    write_jsonl(ROWS, p)
    assert read_jsonl(p) == ROWS


def test_gzip_round_trip(tmp_path):
    p = str(tmp_path / "x.jsonl.gz")
    write_jsonl(ROWS, p)
    with open(p, "rb") as f:
        assert f.read(2) == b"\x1f\x8b", "a .gz path must be written gzip-compressed"
    assert read_jsonl(p) == ROWS


def test_gzip_same_text_as_plain(tmp_path):
    plain, gz = str(tmp_path / "x.jsonl"), str(tmp_path / "x.jsonl.gz")
    write_jsonl(ROWS, plain)
    write_jsonl(ROWS, gz)
    with open(plain, encoding="utf-8") as f, gzip.open(gz, "rt", encoding="utf-8") as g:
        assert f.read() == g.read()


def test_gzip_bytes_deterministic(tmp_path):
    """Same rows -> identical bytes (mtime fixed), so a rebuild does not change the committed file."""
    a, b = str(tmp_path / "a.jsonl.gz"), str(tmp_path / "b.jsonl.gz")
    write_jsonl(ROWS, a)
    write_jsonl(ROWS, b)
    assert open(a, "rb").read() == open(b, "rb").read()


def test_dataset_reads_gzip(tmp_path):
    from src.data import HallucinationDataset

    p = str(tmp_path / "x.jsonl.gz")
    write_jsonl([{"query": "q", "context": "c", "answer": "a", "label": 1, "source": "medhallu"}], p)
    cfg = {"query_col": "query", "context_col": "context", "response_col": "answer",
           "label_col": "label", "dataset_col": "source"}
    ds = HallucinationDataset(p, cfg)
    assert len(ds) == 1 and ds[0]["label"] == 1.0 and ds[0]["source"] == "medhallu"
