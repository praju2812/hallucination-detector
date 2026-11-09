"""Build the unified dataset the loader reads.

Turns RAGTruth and HaluEval into one schema and writes group-split train/val/test
files.

Unified row schema (every row from both sources ends up exactly like this):
    query     : str   the user's question / instruction 
    context   : str   the passage(s) the answer is meant to be grounded in
    answer    : str   the response being judged
    label     : int   1 = hallucinated / unfaithful, 0 = faithful
    source    : str   "ragtruth" | "halueval"
    task      : str   "qa" | "summary" | "data2txt"
    group_id  : str   shared-context key; rows sharing it must not split apart

Run:  python -m scripts.build_dataset \
          --ragtruth-response data/raw/RAGTruth/dataset/response.jsonl \
          --ragtruth-source   data/raw/RAGTruth/dataset/source_info.jsonl \
          --halueval          data/raw/HaluEval/data/qa_data.json \
          --out-dir           data/unified_v2 \
          --v1-dir            data/unified \
          --seed 42

v2 (Phase 1) adds `query`. Rows, order, splits and all other columns must be identical
to v1: check_against_v1 runs before anything is written. data/unified/ and data/raw/
are never written (refuse_protected_out_dir).
"""

from __future__ import annotations
import argparse
import os

import sys
from pathlib import Path

import re

sys.path.append(str(Path(__file__).parent.parent))
from sklearn.model_selection import GroupShuffleSplit
from utils.io import read_jsonl, write_jsonl

PROTECTED_DIRS = ("data/unified", "data/raw")  
KEEP = ("query", "context", "answer", "label", "source", "task", "group_id")   # column order in output files
V1_COLS = ("context", "answer", "label", "source", "task", "group_id")         # must match v1 exactly


# ------------------------------- query extraction -------------------------------

def _halueval_query(record: dict) -> str:
    """Return the query for one HaluEval QA record.

    Input:  one record from qa_data.json (keys: knowledge, question, right_answer, hallucinated_answer).
    Output: the question, stripped of leading/trailing whitespace.

    """
    # read the question from the record.
    question = record.get("question")
    
    if not question:
        raise ValueError("Question is missing or empty")

    # normalise whitespace at the ends.
    question = question.strip()

    # check if the question is still empty after stripping.
    if not question:
        raise ValueError("Question is missing or empty")
    return question

def _ragtruth_query(source_record: dict) -> str:
    """Return the query for one RAGTruth source_info.jsonl record (QA or Summary).

    Input:  the whole source_info.jsonl record (keys: source_id, task_type, source, source_info, prompt).
            The whole record is needed because the Summary instruction lives in `prompt`, not in `source_info`.
    Output :
      - QA:      the bare question from source_info, stripped.
      - Summary: the prompt's first line with the word limit removed: "Summarize the following news".
    Raises ValueError for any other task_type, an empty query, or a Summary first line that does not
    match the known template "Summarize the following news within N words:".

    Data facts : QA source_info is always a dict with `question` and `passages`;
    all 943 Summary prompts start with that template on line 1, then the passage, then "output:".

    """
    # branch on task_type.
    task_type = source_record.get("task_type")

    if task_type == "QA":
        # extract the question from source_info and strip it for whitespace
        question = source_record.get("source_info", {}).get("question", "")

        # check if the question is empty or missing after stripping
        if not question or not question.strip():
            raise ValueError("Question is missing or empty in RAGTruth QA record")
        return question.strip()
            
    elif task_type == "Summary":
        # isolate the instruction from the prompt without touching the passage.
        prompt = source_record.get("prompt", "")
        lines = prompt.split("\n")

        instruction = lines[0]
        # check the instruction matches the expected template 
        if re.fullmatch(r"Summarize the following news within \d+ words:", instruction):
            return "Summarize the following news"
        else:
            raise ValueError(f"Unexpected Summary instruction: {instruction}")
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def check_against_v1(v2_splits: dict[str, list[dict]], v1_dir: str) -> None:
    """Assert that v2 is v1 plus a query column, and nothing else changed.

    Input:  v2_splits = {"train": rows, "val": rows, "test": rows} as they are about to be written;
            v1_dir    = directory holding v1 {split}.jsonl files (read with utils.io.read_jsonl).
    Checks, for every split present in v1_dir as train/val/test:
      - the split exists in v2_splits;
      - same number of rows;
      - row i of v2 equals row i of v1 on every column in V1_COLS (same order, identical values);
      - every v2 row has a `query` that is a non-empty string after stripping.
    Raises AssertionError whose message names the split (and the row index and column where relevant).
    Returns None when everything matches. Never writes anything.

    """
    #for each split name, load the v1 rows.
    v1_rows = {}
    for split in ["train", "val", "test"]:
        if os.path.exists(os.path.join(v1_dir, f"{split}.jsonl")):
            v1_rows[split] = read_jsonl(os.path.join(v1_dir, f"{split}.jsonl"))

    # check if v1 is empty
    if not v1_rows:
        raise AssertionError(f"No v1 splits found in {v1_dir}")
    
    # compare lengths first (a zip over unequal lists silently stops early)
    for split in v1_rows.keys():
        v1 = v1_rows[split]
        v2 = v2_splits.get(split)
        if v2 is None:
            raise AssertionError(f"v2 missing split {split}")
        if len(v1) != len(v2):
            raise AssertionError(f"v1 {split} has {len(v1)} rows, v2 has {len(v2)} rows")
        
    # walk the two lists in parallel; compare the V1_COLS columns.
    for split in v1_rows.keys():
        v1 = v1_rows[split]
        v2 = v2_splits[split]
        for i, (r1, r2) in enumerate(zip(v1, v2)):
            for k in V1_COLS:
                if r1[k] != r2[k]:
                    # only report the first 80 characters of each mismatch per row, to avoid a flood of messages
                    raise AssertionError(f"v1 {split} row {i}, column {k}: {r1[k]!r:.80} != {r2[k]!r:.80}")
                
    # check the query column on every v2 row.
    for split in v2_splits.keys():
        v2 = v2_splits[split]
        for i, r in enumerate(v2):
            query = r.get("query")
            if not isinstance(query, str) or not query.strip():
                raise AssertionError(f"v2 {split} row {i} has empty or non-string query: {query!r}")


def refuse_protected_out_dir(out_dir: str) -> None:
    """Raise ValueError if out_dir is (inside) data/unified or data/raw.
    Compares resolved paths, so data/unified_v2 is allowed (a plain startswith would not be)."""
    target = Path(out_dir).resolve()
    for p in PROTECTED_DIRS:
        protected = Path(p).resolve()
        if target == protected or protected in target.parents:
            raise ValueError(f"refusing to write into {p}/ (hard rule 1): got --out-dir {out_dir}")


def build_tiny(tiny_v1: list[dict], rows: list[dict]) -> list[dict]:
    """The same rows as v1 tiny.jsonl, in the same order, taken from the new rows (so they carry `query`).
    Rows are matched on all V1_COLS. Raises ValueError if a v1 tiny row is not found."""
    index = {tuple(r[k] for k in V1_COLS): r for r in rows}
    out = []
    for i, t in enumerate(tiny_v1):
        key = tuple(t[k] for k in V1_COLS)
        if key not in index:
            raise ValueError(f"tiny row {i} (group_id {t.get('group_id')}) not found in the new build")
        out.append({k: index[key][k] for k in KEEP})
    return out


def load_halueval(path: str) -> list[dict]:
    """Convert HaluEval QA records into unified rows.

    Each record has: knowledge, question, right_answer, hallucinated_answer.

      - For each record emit TWO rows: one with right_answer (label 0) and one
        with hallucinated_answer (label 1). That is where the 50/50 balance comes
        from.
      - Build `context` from `knowledge`
      - group_id: both rows from one record share a context, so they must share a
        group_id (record index). If they
        don't, the faithful and hallucinated halves can leak across the split.
      - task = "qa" for HaluEval QA. source = "halueval".
    """
    rows = []
    for i, r in enumerate(read_jsonl(path)):
        knowledge = r["knowledge"]
        right_answer = r["right_answer"]
        hallucinated_answer = r["hallucinated_answer"]
        context = knowledge
        query = _halueval_query(r)

        # hash the knowledge to create a stable group_id for both rows. This ensures that
        # both the faithful and hallucinated answers for the same knowledge/question pair
        # are kept together in the same split (train/val/test) and prevents data leakage
        group_id = f"halueval-{i}"
        rows.append(
            {
                "query": query,
                "context": context,
                "answer": right_answer,
                "label": 0,
                "source": "halueval",
                "task": "qa",
                "group_id": group_id,
            }
        )
        rows.append(
            {
                "query": query,
                "context": context,
                "answer": hallucinated_answer,
                "label": 1,
                "source": "halueval",
                "task": "qa",
                "group_id": group_id
            }
        )
    return rows

def _ragtruth_context(source_info_value):
    # RAGTruth 'source_info' is a str for Summary, a dict for QA (question+passages).
    if isinstance(source_info_value, str):
        return source_info_value.strip()
    if isinstance(source_info_value, dict):
        # QA: use the retrieved passages as grounding evidence, drop the question
        # (evidence-only, to stay symmetric with HaluEval and Summary).
        passages = source_info_value.get("passages")
        if passages is not None:
            return str(passages).strip()
        # fallback: join whatever string values exist, so nothing silently vanishes
        return "\n\n".join(str(v) for v in source_info_value.values() if isinstance(v, str)).strip()
    return str(source_info_value).strip()

def load_ragtruth(response_path: str, source_info_path: str) -> list[dict]:
    """Join RAGTruth's two files and convert to unified rows.

    source_info.jsonl holds the context/prompt and task_type, keyed by source_id.
    response.jsonl holds each model response and its human-annotated spans, also
    keyed by source_id.

    - Read both files. Build a dict from source_id -> source_info record so you
    can look up context and task for each response.
    - context: the source passage(s)/prompt from source_info. 
    - task: map task_type to "qa" | "summary" | "data2txt".
    - group_id: use source_id.
    - source = "ragtruth".
    """
    TASK_MAP = {"QA": "qa", "Summary": "summary", "Data2txt": "data2txt"}
    # print(set(s["task_type"] for s in read_jsonl(source_info_path)))  # sanity check: all task_types are in TASK_MAP

    info = {}
    for s in read_jsonl(source_info_path):
        task = TASK_MAP[s["task_type"]]
        info[s["source_id"]] = {
            "context": s["source_info"],
            "task": task,
            # Data2txt rows are dropped below, so they never need a query
            "query": None if task == "data2txt" else _ragtruth_query(s),
        }

    rows = []
    for r in read_jsonl(response_path):
        sid = r["source_id"]
        if sid not in info or info[sid]["task"] == "data2txt":
            continue
        rows.append({
            "query":    info[sid]["query"],
            "context":  _ragtruth_context(info[sid]["context"]),
            "answer":   r["response"],
            "label":    1 if r["labels"] else 0,
            "source":   "ragtruth",
            "task":     info[sid]["task"],
            "group_id": sid,
            "split":    r.get("split")      # optional, for the split decision
        })
    return rows

def _grouped_split(rows, test_size, seed):
    """Split rows into (bigger, smaller) by group_id, no group shared."""

    # extract the group_id for each row to use in the split
    group_ids = [r["group_id"] for r in rows]

    # Group ShuffleSplit will split the data into train/test sets while ensuring that the same group_id does not appear in both sets.
    # n_splits=1 means we only want one split, test_size is the proportion of the dataset to include in the test split, and random_state ensures reproducibility.
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)

    # gss returns indices for the train/test split. We use next() to get the first (and only) split.
    # big_idx will contain the indices for the larger split (train), and small_idx will contain the indices for the smaller split (test).
    big_idx, small_idx = next(gss.split(rows, groups=group_ids))
    return [rows[i] for i in big_idx], [rows[i] for i in small_idx]


def assign_splits(rows, seed, val_frac=0.1, test_frac=0.1):
    """
    Write r['split'] in {'train','val','test'} on every row.

    RAGTruth: honour the file's train/test. Carve val out of its train pool so
    the published test set is untouched.
    HaluEval: no official split, so make a three-way grouped split.
    """
    ragtruth = [r for r in rows if r["source"] == "ragtruth"]
    halueval = [r for r in rows if r["source"] == "halueval"]

    # sanity: see what split values RAGTruth actually ships
    print("ragtruth split values:", {r.get("split") for r in ragtruth})

    # ---- RAGTruth: honour official train/test ----
    rt_test = [r for r in ragtruth if r.get("split") == "test"]
    rt_train_all = [r for r in ragtruth if r.get("split") == "train"]
    rt_train, rt_val = _grouped_split(rt_train_all, test_size=val_frac, seed=seed)

    # ---- HaluEval: three-way grouped split ----
    he_trainval, he_test = _grouped_split(halueval, test_size=test_frac, seed=seed)
    he_train, he_val = _grouped_split(he_trainval, test_size=val_frac, seed=seed)

    for r in rt_train + he_train:
        r["split"] = "train"
    for r in rt_val + he_val:
        r["split"] = "val"
    for r in rt_test + he_test:
        r["split"] = "test"

    return rows
# ------------------------- checkpoints (filled in) -------------------------

def assert_no_group_leakage(train, val, test) -> None:
    g_tr = {r["group_id"] for r in train}
    g_va = {r["group_id"] for r in val}
    g_te = {r["group_id"] for r in test}
    assert not (g_tr & g_va), "group leak: train/val share groups"
    assert not (g_tr & g_te), "group leak: train/test share groups"
    assert not (g_va & g_te), "group leak: val/test share groups"


def assert_schema(rows: list[dict]) -> None:
    fields = set(KEEP)
    for r in rows:
        assert isinstance(r["query"], str) and r["query"].strip(), f"bad query: {r['group_id']}"
        assert fields <= set(r), f"row missing fields: {fields - set(r)}"
        assert r["label"] in (0, 1), f"bad label: {r['label']!r}"
        assert r["context"] and r["answer"], "empty context or answer"
        assert isinstance(r["context"], str) and r["context"].strip(), f"bad context: {r['group_id']}"
        assert isinstance(r["answer"], str) and r["answer"].strip(), f"bad answer: {r['group_id']}"


def report(name: str, rows: list[dict]) -> None:
    n = len(rows)
    by_src = {}
    for r in rows:
        d = by_src.setdefault(r["source"], [0, 0])   # [count, positives]
        d[0] += 1
        d[1] += r["label"]
    parts = ", ".join(f"{s} {c} ({p/c:.0%} pos)" for s, (c, p) in by_src.items())
    print(f"[{name}] n={n}  {parts}")


# ------------------------------ orchestration ------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ragtruth-response", required=True)
    ap.add_argument("--ragtruth-source", required=True)
    ap.add_argument("--halueval", required=True)
    ap.add_argument("--out-dir", default="data/unified_v2")
    ap.add_argument("--v1-dir", default="data/unified", help="read-only: v1 splits + tiny.jsonl to check against")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    refuse_protected_out_dir(args.out_dir)

    rows = []
    rows += load_ragtruth(args.ragtruth_response, args.ragtruth_source)
    rows += load_halueval(args.halueval)

    assert_schema(rows)
    report("all", rows)

    rows = assign_splits(rows, seed=args.seed)
    train = []
    test = []
    val = []
    for r in rows:
        if r["split"] not in ("train", "val", "test"):
            raise ValueError(f"row missing split: {r}")
        if r["split"] == "train":
            train.append(r)
        elif r["split"] == "val":
            val.append(r)
        elif r["split"] == "test":
            test.append(r)
        else:
            raise ValueError(f"row has unknown split: {r}")

    assert_no_group_leakage(train, val, test)

    # keep only the fields in KEEP (a tuple, so the key order in the files is fixed)
    clean = {name: [{k: r[k] for k in KEEP} for r in split]
             for name, split in [("train", train), ("val", val), ("test", test)]}

    # v2 must be v1 plus query: check BEFORE anything is written
    check_against_v1(clean, args.v1_dir)
    print(f"check_against_v1: PASS against {args.v1_dir}")

    tiny_v1_path = os.path.join(args.v1_dir, "tiny.jsonl")
    tiny = build_tiny(read_jsonl(tiny_v1_path), clean["train"] + clean["val"] + clean["test"]) \
        if os.path.exists(tiny_v1_path) else None

    for name, split in clean.items():
        report(name, split)
        write_jsonl(split, os.path.join(args.out_dir, f"{name}.jsonl"))
    if tiny is not None:
        write_jsonl(tiny, os.path.join(args.out_dir, "tiny.jsonl"))
        print(f"tiny.jsonl: {len(tiny)} rows (same rows and order as {tiny_v1_path})")

    print(f"wrote splits to {args.out_dir}")


if __name__ == "__main__":
    main()