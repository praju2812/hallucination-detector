"""Spec for the Phase 1 additions to scripts/build_dataset.py.

Decisions (Phase 1):
  - Summary query = first line of `prompt` with " within N words" and the trailing ":" removed
    -> "Summarize the following news".
  - QA query = bare source_info["question"] (RAGTruth) / record["question"] (HaluEval).
  - Queries are stripped of leading/trailing whitespace; an empty query is an error.

Run: python -m pytest tests/test_build_dataset.py
"""

import json

import pytest

from scripts.build_dataset import (
    _halueval_query,
    _ragtruth_query,
    build_tiny,
    check_against_v1,
    refuse_protected_out_dir,
)

# ---------------------------------------------------------------- toy raw records
QA_RECORD = {
    "source_id": "qa1",
    "task_type": "QA",
    "source": "MARCO",
    "source_info": {
        "question": "butcher shop phone number",
        "passages": "passage 1:Butcher Shop - Hayward 826 B Street\n\npassage 2:Lakefield 212 Main Street",
    },
    "prompt": "Briefly answer the following question:\nbutcher shop phone number\nBear in mind ...\n\noutput:",
}


def summary_record(n_words: int, passage: str = "Seventy years ago, Anne Frank died of typhus.") -> dict:
    return {
        "source_id": f"sum{n_words}",
        "task_type": "Summary",
        "source": "CNN/DM",
        "source_info": passage,
        "prompt": f"Summarize the following news within {n_words} words:\n{passage}\n\noutput:",
    }


# ---------------------------------------------------------------- HaluEval query
def test_halueval_query_is_question():
    rec = {"knowledge": "k", "question": "Which magazine was started first?",
           "right_answer": "a", "hallucinated_answer": "b"}
    assert _halueval_query(rec) == "Which magazine was started first?"


def test_halueval_query_strips_whitespace():
    rec = {"knowledge": "k", "question": "  Who wrote it?\n", "right_answer": "a", "hallucinated_answer": "b"}
    assert _halueval_query(rec) == "Who wrote it?"


@pytest.mark.parametrize("q", ["", "   ", None])
def test_halueval_empty_question_raises(q):
    rec = {"knowledge": "k", "question": q, "right_answer": "a", "hallucinated_answer": "b"}
    with pytest.raises(ValueError):
        _halueval_query(rec)


# ---------------------------------------------------------------- RAGTruth query
def test_ragtruth_qa_query_is_bare_question():
    """Not the prompt's instruction line, not the passages."""
    assert _ragtruth_query(QA_RECORD) == "butcher shop phone number"


def test_ragtruth_qa_query_strips_whitespace():
    rec = json.loads(json.dumps(QA_RECORD))
    rec["source_info"]["question"] = "  butcher shop phone number \n"
    assert _ragtruth_query(rec) == "butcher shop phone number"


def test_ragtruth_qa_empty_question_raises():
    rec = json.loads(json.dumps(QA_RECORD))
    rec["source_info"]["question"] = " "
    with pytest.raises(ValueError):
        _ragtruth_query(rec)


@pytest.mark.parametrize("n", [25, 103, 116, 200])
def test_ragtruth_summary_query_drops_word_limit(n):
    """Decision 1: the word limit N is removed, so every Summary row gets the same query."""
    assert _ragtruth_query(summary_record(n)) == "Summarize the following news"


def test_ragtruth_summary_query_never_contains_passage_text():
    passage = "Summarize this: the passage itself starts with an instruction-like phrase."
    q = _ragtruth_query(summary_record(80, passage))
    assert q == "Summarize the following news"
    assert "passage" not in q


def test_ragtruth_summary_unexpected_instruction_raises():
    """If the first line does not match the known template, fail loudly instead of
    guessing (a bad split rule could leak passage text into the query)."""
    rec = summary_record(80)
    rec["prompt"] = "Write a short poem about this news:\n" + rec["source_info"] + "\n\noutput:"
    with pytest.raises(ValueError):
        _ragtruth_query(rec)


def test_ragtruth_data2txt_raises():
    """Data2txt rows are dropped from the dataset; asking for their query is a bug."""
    rec = {"source_id": "d1", "task_type": "Data2txt", "source": "Yelp",
           "source_info": {"name": "x"}, "prompt": "Write an objective overview:\n..."}
    with pytest.raises(ValueError):
        _ragtruth_query(rec)


# ---------------------------------------------------------------- check_against_v1
V1_COLS = ("context", "answer", "label", "source", "task", "group_id")


def v1_rows(prefix: str, n: int) -> list[dict]:
    return [
        {"context": f"{prefix} context {i}", "answer": f"{prefix} answer {i}", "label": i % 2,
         "source": "ragtruth" if i % 2 else "halueval", "task": "qa", "group_id": f"{prefix}-g{i}"}
        for i in range(n)
    ]


def with_query(rows: list[dict]) -> list[dict]:
    return [{**r, "query": f"question for {r['group_id']}"} for r in rows]


@pytest.fixture
def v1_dir(tmp_path):
    """Writes toy v1 split files (no query column) and returns (dir, {split: rows})."""
    splits = {"train": v1_rows("tr", 6), "val": v1_rows("va", 3), "test": v1_rows("te", 4)}
    for name, rows in splits.items():
        with open(tmp_path / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return tmp_path, splits


def test_check_against_v1_passes_on_identical_rows_plus_query(v1_dir):
    d, splits = v1_dir
    v2 = {name: with_query(rows) for name, rows in splits.items()}
    check_against_v1(v2, str(d))  # must not raise


def test_check_against_v1_ignores_key_order(v1_dir):
    """JSON key order is not content."""
    d, splits = v1_dir
    v2 = {name: [dict(reversed(list(r.items()))) for r in with_query(rows)] for name, rows in splits.items()}
    check_against_v1(v2, str(d))


def test_check_against_v1_fails_on_reordered_rows(v1_dir):
    d, splits = v1_dir
    v2 = {name: with_query(rows) for name, rows in splits.items()}
    v2["train"][0], v2["train"][1] = v2["train"][1], v2["train"][0]
    with pytest.raises(AssertionError, match="train"):
        check_against_v1(v2, str(d))


@pytest.mark.parametrize("field,value", [
    ("context", "edited context"),
    ("answer", "edited answer"),
    ("label", 1),            # row 2 of val has label 0
    ("source", "covidqa"),
    ("task", "summary"),
    ("group_id", "va-g999"),
])
def test_check_against_v1_fails_on_changed_field(v1_dir, field, value):
    d, splits = v1_dir
    v2 = {name: with_query(rows) for name, rows in splits.items()}
    v2["val"][2][field] = value
    with pytest.raises(AssertionError, match="val"):
        check_against_v1(v2, str(d))


@pytest.mark.parametrize("bad", ["", "   ", None])
def test_check_against_v1_fails_on_empty_query(v1_dir, bad):
    d, splits = v1_dir
    v2 = {name: with_query(rows) for name, rows in splits.items()}
    v2["test"][3]["query"] = bad
    with pytest.raises(AssertionError, match="test"):
        check_against_v1(v2, str(d))


def test_check_against_v1_fails_on_missing_query_key(v1_dir):
    d, splits = v1_dir
    v2 = {name: with_query(rows) for name, rows in splits.items()}
    del v2["test"][0]["query"]
    with pytest.raises(AssertionError, match="test"):
        check_against_v1(v2, str(d))


def test_check_against_v1_fails_on_length_mismatch(v1_dir):
    d, splits = v1_dir
    v2 = {name: with_query(rows) for name, rows in splits.items()}
    v2["train"] = v2["train"][:-1]
    with pytest.raises(AssertionError, match="train"):
        check_against_v1(v2, str(d))


def test_check_against_v1_fails_on_missing_split(v1_dir):
    d, splits = v1_dir
    v2 = {name: with_query(rows) for name, rows in splits.items() if name != "val"}
    with pytest.raises(AssertionError, match="val"):
        check_against_v1(v2, str(d))


# ---------------------------------------------------------------- plumbing (written in full)
@pytest.mark.parametrize("path", ["data/unified", "data/unified/", "./data/unified",
                                  "data/unified/sub", "data/raw", "data/raw/RAGTruth"])
def test_refuse_protected_out_dir(path):
    """Hard rule 1: never write into data/unified/ or data/raw/."""
    with pytest.raises(ValueError):
        refuse_protected_out_dir(path)


@pytest.mark.parametrize("path", ["data/unified_v2", "data/clinical"])
def test_allowed_out_dirs(path):
    refuse_protected_out_dir(path)  # must not raise (note: data/unified_v2 starts with "data/unified")


def test_build_tiny_keeps_v1_order_and_adds_query():
    rows = with_query(v1_rows("tr", 6))
    tiny_v1 = [{k: rows[i][k] for k in V1_COLS} for i in (4, 1, 3)]
    tiny = build_tiny(tiny_v1, rows)
    assert [r["group_id"] for r in tiny] == ["tr-g4", "tr-g1", "tr-g3"]
    assert all(r["query"] == f"question for {r['group_id']}" for r in tiny)


def test_build_tiny_raises_if_row_missing():
    rows = with_query(v1_rows("tr", 3))
    missing = {**{k: rows[0][k] for k in V1_COLS}, "answer": "not in v2"}
    with pytest.raises(ValueError):
        build_tiny([missing], rows)
