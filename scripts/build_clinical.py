"""Build the clinical dataset under data/clinical/:

    train.jsonl(.gz)    clinical training data, never reduced for overlap
    val.jsonl           validation rows that share nothing with the training data
    test.jsonl          CLEAN test: test rows that share nothing with the training data
    test_full.jsonl     FULL official test sets (RQ1 reports G on both full and clean test)
    removed_val.jsonl   validation rows removed for overlap, with `removed_by`
    removed_test.jsonl  test rows removed for overlap (test_full = test + removed_test), with `removed_by`
    handcheck/          D8 hand-check sample, drawn from the FULL test set
    build_report.json

Sources (revisions pinned in REVISIONS):
    ragbench_pubmedqa, ragbench_covidqa : RAGBench official train / validation / test
    medhallu                            : MedHallu pqa_labeled = test; pqa_artificial = train/val (90/10)

Row schema: the unified schema (scripts/build_dataset.KEEP) plus
    difficulty, category      : MedHallu only (null on RAGBench rows)
    shares_passage_with_train : test files only; True if any passage (of >= 8 words) also appears in the training
                                data (clinical train + data/unified_v2/train.jsonl). Always False in the clean test.
    removed_by                : removed_* files only; why the row was removed (see REMOVED_BY_VALUES)

Overlap keys (overlap_keys): the normalised question, plus each passage whose normalised text has at least
min_words words (default 8, used everywhere in build() and the audit), so boilerplate passages such as "None."
are not counted as overlap. RAGTruth Summary rows give no question key (their query is the constant
"Summarize the following news"); see training_keys.

Rule (supervisor decision, 10-11 Nov; CLAUDE.md hard rule 6): training data is never reduced for overlap.
Steps, in the order build() applies them:
    -  drop unlabelled rows only where expected (RAGBench PubMedQA validation: 1); a test row never
    -  contexts joined in RAGTruth QA style: "passage 1:{p1}\\n\\npassage 2:{p2}"
    -  group_id = sha1 of the normalised question, no source prefix
    -  pqa_artificial: 90/10 train/val by group_id (hashlib.sha1)
    -  training data = RAGBench PubMedQA train + RAGBench CovidQA train + pqa_artificial train
                       + data/unified_v2/train.jsonl
    -  validation and test: every GROUP with a row sharing a question or an 8+ word passage with the training data
       is removed (drop_overlap) and saved to removed_{split}.jsonl; the full test set is kept as test_full
    -  val x test passage sharing is reported only (no row removed); val x test question sharing must be 0 (rule 2)

Needs `datasets` (requirements-data.txt, local only). Training never does: it reads the JSONL files.
Run:  python -m scripts.build_clinical --out-dir data/clinical
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from scripts.build_dataset import KEEP, assert_no_group_leakage, assert_schema, refuse_protected_out_dir
from scripts.data_checks.common import norm  # same normalisation as the 22 Oct overlap checks
from utils.io import read_jsonl, write_jsonl

RAGBENCH_REPO = "rungalileo/ragbench"
MEDHALLU_REPO = "UTAustin-AIHealth/MedHallu"
REVISIONS = {  # dataset commit hashes; the 22 Oct data checks used these
    RAGBENCH_REPO: "97808f3e5fd16ede40bbff6c2949af8139b2eb7b",
    MEDHALLU_REPO: "515060458a945c633debc6fd5baac7764416b724",
}
RAGBENCH_SOURCES = {"pubmedqa": "ragbench_pubmedqa", "covidqa": "ragbench_covidqa"}
MEDHALLU = "medhallu"
HF_SPLITS = {"train": "train", "validation": "val", "test": "test"}
RAGBENCH_COLS = ["id", "question", "documents", "response", "adherence_score",
                 "documents_sentences", "all_relevant_sentence_keys"]
SPLITS = ("train", "val", "test")
COLUMNS = KEEP + ("difficulty", "category")
TEST_FLAG = "shares_passage_with_train"
REMOVED_BY = "removed_by"
# removed_by values: which key kind matched which part of the training data; "group" = only another row of the
# same group matched (drop_overlap removes whole groups)
REMOVED_BY_VALUES = ("question:clinical_train", "passage:clinical_train", "question:general_train",
                     "passage:general_train", "group")
OUT_SPLITS = ("train", "val", "test", "test_full")  # splits returned by build(); "test" is the clean test
EXPECTED_UNLABELLED = {("ragbench_pubmedqa", "val"): 1}  # measured 9 Nov; any other count raises
MAX_PLAIN_MB = 90  # GitHub limit is 100 MB; above this a split is written as .jsonl.gz

def question_group_id(question: str) -> str:
    """group_id for any clinical row: the same question gives the same id in every source.

    Input:  a question string (RAGBench `question`, MedHallu `Question`, or a row's `query`).
    Output: 40-character hex digest of hashlib.sha1 over the normalised question (UTF-8).
    Normalised = lowercase, whitespace runs collapsed to one space, ends stripped. No source prefix.
    Raises ValueError if the question is empty after normalisation.

    Invariant: question_group_id(q) == question_group_id(q.upper()) == question_group_id(" " + q + "\\n").
    """
    # normalise the question using the norm function
    normalised_question = norm(question)

    # check for an empty result
    if not normalised_question:
        raise ValueError("Question is empty after normalisation")
    
    # hash with hashlib.sha1 and return the hex digest
    return hashlib.sha1(normalised_question.encode("utf-8")).hexdigest()


def ragbench_label(adherence_score) -> int | None:
    """Map RAGBench adherence_score to the unified label.

    Input:  the raw value (datasets gives a Python bool, or None for the 1 unlabelled validation row).
    Output: 1 (hallucinated) for False, 0 (faithful) for True, None for None.
    Raises ValueError for anything else (a string such as "True", or a number).
    """
    # handle None first
    if adherence_score is None:
        return None
    
    # map the two bool values
    adherence_map = {True: 0, False: 1}

    # check if the input is a valid key in the map
    if not isinstance(adherence_score, bool) or adherence_score not in adherence_map:
        raise ValueError(f"Unexpected adherence_score value: {adherence_score}")

    # return the mapped value
    return adherence_map[adherence_score]


def join_passages(passages: list[str]) -> str:
    """Join retrieved documents into one context in the RAGTruth QA style .

    Input:  list of passage strings (RAGBench `documents`, MedHallu `Knowledge`).
    Output: "passage 1:{p1}\n\npassage 2:{p2}..." with each passage stripped; numbering from 1;
            no space after the colon (that is how the RAGTruth contexts in data/unified_v2 look).
    Raises ValueError for an empty list or a passage that is empty after stripping.
    """
    # strip each passage
    stripped_passages = [p.strip() for p in passages]

    # check for an empty list or any empty passage after stripping
    if not passages or any(not p for p in stripped_passages):
        raise ValueError("Passages contains an empty passage")

    # label each passage with its 1-based number, then join with a blank line.
    labeled_passages = [f"passage {i + 1}:{p}" for i, p in enumerate(stripped_passages)]
    return "\n\n".join(labeled_passages)

def split_passages(context: str) -> list[str]:
    """Inverse of join_passages: recover the passage texts from a context string.

    Input:  a context from any source: a clinical context (join_passages output), a RAGTruth QA context
            (same style), or a context with no passage labels (HaluEval knowledge, RAGTruth Summary).
    Output: list of passage texts without their "passage N:" labels, each stripped.
            A context that does not start with "passage 1:" is one passage: [context.strip()].
    Invariant: split_passages(join_passages(ps)) == [p.strip() for p in ps].
    """
    # decide whether the context uses passage labels at all
    if not context.startswith("passage 1:"):
        return [context.strip()]

    # split only where a blank line is followed by the next label ("passage N:")
    # use a regex to find the pattern of a blank line followed by "passage N:"
    pattern = r"\n\n(?=passage \d+:)"
    
    # split the context at the pattern and remove the label from each piece and strip it
    passages = re.split(pattern, context)
    stripped_passages = []
    for p in passages:
        # remove the "passage N:" label using regex
        match = re.match(r"passage \d+:(.*)", p, re.DOTALL)
        if not match:
            raise ValueError(f"Passage does not start with expected label: {p}")
        stripped_passages.append(match.group(1).strip())
    return stripped_passages

def ragbench_rows(records: list[dict], source: str) -> list[dict]:
    """Convert RAGBench records (one split of one subset) into unified rows, one row per record.

    Input:  records with keys id, question, documents (list[str]), response, adherence_score;
            source = "ragbench_pubmedqa" | "ragbench_covidqa".
    Output: rows in record order, each with
            query     = question, stripped
            context   = join_passages(documents)
            answer    = response
            label     = ragbench_label(adherence_score)   (None is kept: drop_unlabelled decides)
            source, task = "qa", group_id = question_group_id(question)
            difficulty = None, category = None
            _orig_id  = str(record["id"])   (private: hand-check key file; stripped before writing)
    Raises ValueError for an empty response.
    """
    # loop over the records; build one dict per record with the keys above
    rows = []
    for rec in records:
        # extract the fields
        query = rec["question"].strip()
        context = join_passages(rec["documents"])
        answer = rec["response"]
        label = ragbench_label(rec["adherence_score"])
        group_id = question_group_id(query)

        # check for an empty response
        if not answer or not answer.strip():
            raise ValueError(f"Record {rec['id']} has an empty response")

        # build the row dict
        row = {
            "query": query,
            "context": context,
            "answer": answer,
            "label": label,
            "source": source,
            "task": "qa",
            "group_id": group_id,
            "difficulty": None,
            "category": None,
            "_orig_id": str(rec["id"]),
        }
        rows.append(row)
    return rows

def medhallu_rows(records: list[dict]) -> list[dict]:
    """Convert MedHallu records (pqa_labeled or pqa_artificial) into unified rows.

    Input:  records with keys Question, Knowledge (list[str]), Ground Truth, Hallucinated Answer,
            Difficulty Level, Category of Hallucination.
    Output: TWO rows per record: the faithful row and the hallucinated row, in that order, each with
              Ground Truth        -> label 0
              Hallucinated Answer -> label 1
            Both rows: query = Question stripped, context = join_passages(Knowledge), source = "medhallu",
            task = "qa", group_id = question_group_id(Question), and the record's difficulty and category
            (the hallucinated answer's, copied onto the faithful row so each difficulty subset stays balanced).
    Raises ValueError for an empty Ground Truth or Hallucinated Answer.
    """
    # loop over the records; build two dicts per record with the keys above
    # per record, compute the fields both rows share
    rows = []
    for rec in records:
        query = rec["Question"].strip()
        context = join_passages(rec["Knowledge"])
        group_id = question_group_id(query)
        difficulty = rec["Difficulty Level"]
        category = rec["Category of Hallucination"]

        # check for empty answers
        if not rec["Ground Truth"] or not rec["Ground Truth"].strip():
            raise ValueError(f"Record with question '{query}' has an empty Ground Truth")
        if not rec["Hallucinated Answer"] or not rec["Hallucinated Answer"].strip():
            raise ValueError(f"Record with question '{query}' has an empty Hallucinated Answer")

        # create the two rows
        faithful_row = {
            "query": query,
            "context": context,
            "answer": rec["Ground Truth"],
            "label": 0,
            "source": MEDHALLU,
            "task": "qa",
            "group_id": group_id,
            "difficulty": difficulty,
            "category": category,
            "_orig_id": None,  # no original ID for MedHallu
        }
        hallucinated_row = {
            "query": query,
            "context": context,
            "answer": rec["Hallucinated Answer"],
            "label": 1,
            "source": MEDHALLU,
            "task": "qa",
            "group_id": group_id,
            "difficulty": difficulty,
            "category": category,
            "_orig_id": None,  # no original ID for MedHallu
        }

        rows.extend([faithful_row, hallucinated_row])

    return rows

def split_pqa_artificial(rows: list[dict], seed: int, val_frac: float = 0.1) -> tuple[list[dict], list[dict]]:
    """Split pqa_artificial rows into (train, val) by group_id, deterministically.

    Input:  medhallu_rows output for pqa_artificial; seed; val_frac (share of GROUPS sent to val).
    Output: (train, val); every row in exactly one; rows of one group always together;
            row order kept within each output.
    The assignment is a function of (seed, group_id) only, computed with hashlib.sha1, so it does not
    depend on row order, the number of rows, or PYTHONHASHSEED.
    """
    # for one group_id, turn sha1 of (seed, group_id) into a number in [0, 1)
    def group_hash(group_id: str) -> float:
        # concatenate seed and group_id, encode to bytes, compute sha1 hash
        hash_input = f"{seed}-{group_id}".encode("utf-8")
        hash_digest = hashlib.sha1(hash_input).hexdigest()

        # convert the hex digest to an integer
        hash_int = int(hash_digest, 16)

        # normalize to [0, 1) by dividing by the maximum possible value of a SHA-1 hash
        max_sha1_value = 2**160  # SHA-1 produces a 160-bit hash
        return hash_int / max_sha1_value
    
    # send the group to val if that number is below val_frac
    train_rows = []
    val_rows = []

    # walk the rows once and append each to the list its group belongs to.
    for row in rows:
        group_id = row["group_id"]
        if group_hash(group_id) < val_frac:
            val_rows.append(row)
        else:
            train_rows.append(row)
    return train_rows, val_rows

def overlap_keys(row: dict, min_words: int = 8) -> set[str]:
    """Keys that identify a row for overlap checks: its question and each of its passages.
    Passages whose normalised text has fewer than min_words words get no key (boilerplate such as `None` is ignored).
    min_words=0 keeps every non-empty passage

    Input:  a unified row (uses `query` and `context`).
    Output: {"q:" + question_group_id(query)} plus {"p:" + sha1 hex of the normalised passage text}
            for every passage in split_passages(context). Normalised = the same rule as for questions.
    Two rows overlap if their key sets intersect. The "q:" / "p:" prefixes let callers keep one kind.
    """
    # the question key
    q_key = f"q:{question_group_id(row['query'])}"
    # one key per passage
    # skip empty passages (split_passages returns them stripped, so they are empty strings)
    # skip passages with fewer than min_words words
    passage_keys = set()
    for passage in split_passages(row['context']):
        normed_passage = norm(passage)
        if normed_passage and len(normed_passage.split()) >= min_words:
            p_key = f"p:{hashlib.sha1(normed_passage.encode('utf-8')).hexdigest()}"
            passage_keys.add(p_key)
    return {q_key} | passage_keys

def drop_overlap(rows: list[dict], reference: list[dict], keys=overlap_keys) -> tuple[list[dict], list[dict]]:
    """Remove from `rows` every GROUP that overlaps `reference` (edit the training side only).

    Input:  rows = the side to clean (training/validation)
            reference = the side to protect(never modified)
            keys = function row -> set of keys (default overlap_keys)
    Output: (kept, removed), both in the original order of `rows`; len(kept) + len(removed) == len(rows).
            A row is removed if any row of its group shares a key with any reference row.
    """
    # collect every key of the reference side once
    ref_keys = set()

    # compute the keys for each reference row and add them to the set
    for r in reference:
        ref_keys |= keys(r)

    # find the group_ids in `rows` that have at least one row touching those keys.
    # create a set to hold the group_ids that need to be removed
    groups_to_remove = set()
    for r in rows:
        # check if the current row's keys intersect with the reference keys
        if keys(r) & ref_keys:
            groups_to_remove.add(r["group_id"])
        
    # split `rows` into kept / removed by group_id.
    kept = [r for r in rows if r["group_id"] not in groups_to_remove]
    removed = [r for r in rows if r["group_id"] in groups_to_remove]
    return kept, removed


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def q_keys(row: dict) -> set[str]:
    return {k for k in overlap_keys(row) if k.startswith("q:")}


def p_keys(row: dict) -> set[str]:
    return {k for k in overlap_keys(row) if k.startswith("p:")}


def is_constant_query(row: dict) -> bool:
    """RAGTruth Summary rows all have the query "Summarize the following news", so it identifies nothing."""
    return row.get("source") == "ragtruth" and row.get("task") == "summary"


def training_keys(row: dict) -> set[str]:
    """overlap_keys at the default min_words, without the question key of a constant-query row."""
    keys = overlap_keys(row)
    return {k for k in keys if not k.startswith("q:")} if is_constant_query(row) else keys


def key_union(rows: list[dict], keys=overlap_keys) -> set[str]:
    out = set()
    for r in rows:
        out |= keys(r)
    return out


def passage_key(text: str) -> str | None:
    """The "p:" key overlap_keys gives one passage (via a one-passage row, so the rule lives in one place),
    or None if overlap_keys skips it (fewer than its default min_words words)."""
    keys = p_keys({"query": "_", "context": text})
    if len(keys) > 1:
        raise ValueError(f"one passage gave {len(keys)} passage keys: {text[:80]!r}")
    return next(iter(keys), None)


def git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def load_hf(repo: str, config: str, revision: str):
    from datasets import load_dataset  # local only: requirements-data.txt

    return load_dataset(repo, config, revision=revision)


def load_raw() -> dict:
    """{"ragbench_pubmedqa": {"train": [...], "val": [...], "test": [...]}, "ragbench_covidqa": {...},
        "pqa_labeled": [...], "pqa_artificial": [...]}, records as plain dicts."""
    raw = {}
    for config, source in RAGBENCH_SOURCES.items():
        ds = load_hf(RAGBENCH_REPO, config, REVISIONS[RAGBENCH_REPO])
        if set(ds) != set(HF_SPLITS):
            raise ValueError(f"RAGBench {config}: splits {sorted(ds)}, expected {sorted(HF_SPLITS)}")
        raw[source] = {HF_SPLITS[s]: ds[s].select_columns(RAGBENCH_COLS).to_list() for s in ds}
    for config in ("pqa_labeled", "pqa_artificial"):
        raw[config] = load_hf(MEDHALLU_REPO, config, REVISIONS[MEDHALLU_REPO])["train"].to_list()
    return raw


def drop_unlabelled(rows: list[dict], source: str, split: str,
                    expected: dict = EXPECTED_UNLABELLED) -> tuple[list[dict], int]:
    """D7 guard: drop label-None rows only where the measured count says so; a test row never."""
    n = sum(r["label"] is None for r in rows)
    if split == "test" and n:
        raise ValueError(f"{source}/test: {n} unlabelled row(s); test rows are never dropped")
    want = expected.get((source, split), 0)
    if n != want:
        raise ValueError(f"{source}/{split}: {n} unlabelled row(s), expected {want}")
    return [r for r in rows if r["label"] is not None], n


def build(raw: dict, general_train: list[dict], seed: int, val_frac: float = 0.1,
          expected_unlabelled: dict = EXPECTED_UNLABELLED) -> tuple[dict, dict, dict]:
    """All rules in order; no file I/O.

    Returns (splits, removed, report):
        splits  = {"train", "val", "test" (clean), "test_full"} -> rows; train is never reduced for overlap
        removed = {"val", "test"} -> rows removed for overlap, each with `removed_by`, in original order
        report  = counts for build_report.json
    Invariants: test_full == test + removed["test"] as row sets; val + removed["val"] == validation after D7.
    """
    report = {"unlabelled": {}, "removals": [], "flag": {}, "medhallu_shared_passages": [],
              "pqa_artificial_duplicates": []}

    # rows + D7
    rb = {}
    for source in RAGBENCH_SOURCES.values():
        rb[source] = {}
        for split in SPLITS:
            rows, n = drop_unlabelled(ragbench_rows(raw[source][split], source), source, split, expected_unlabelled)
            report["unlabelled"][f"{source}/{split}"] = n
            rb[source][split] = rows
    mh_test = medhallu_rows(raw["pqa_labeled"])
    mh_art = medhallu_rows(raw["pqa_artificial"])

    # pqa_artificial split
    mh_train, mh_val = split_pqa_artificial(mh_art, seed, val_frac)
    per_group = Counter(r["group_id"] for r in mh_art)
    for gid, n in per_group.items():
        if n > 2:  # more than one record (two rows each) for one normalised question
            where = sorted({s for s, rows in (("train", mh_train), ("val", mh_val)) for r in rows if r["group_id"] == gid})
            q = next(r["query"] for r in mh_art if r["group_id"] == gid)
            report["pqa_artificial_duplicates"].append({"question": q, "records": n // 2, "splits_after_split": where})

    train = rb["ragbench_pubmedqa"]["train"] + rb["ragbench_covidqa"]["train"] + mh_train
    val_all = rb["ragbench_pubmedqa"]["val"] + rb["ragbench_covidqa"]["val"] + mh_val
    test_full = rb["ragbench_pubmedqa"]["test"] + rb["ragbench_covidqa"]["test"] + mh_test

    # val / test x training data (question OR 8+ word passage); whole groups, training never reduced
    clin_keys = key_union(train, training_keys)
    gen_keys = key_union(general_train, training_keys)
    training = train + general_train
    val, removed_val = drop_overlap(val_all, training, keys=training_keys)
    test, removed_test = drop_overlap(test_full, training, keys=training_keys)
    for split, before, removed in (("val", val_all, removed_val), ("test", test_full, removed_test)):
        for r in removed:
            r[REMOVED_BY] = removal_reasons(r, clin_keys, gen_keys)
        for source in sorted({r["source"] for r in before}):
            b = [r for r in before if r["source"] == source]
            rm = [r for r in removed if r["source"] == source]
            report["removals"].append({
                "split": split, "source": source, "rows_before": len(b), "positives_before": sum(r["label"] for r in b),
                "removed": len(rm), "positives_removed": sum(r["label"] for r in rm),
                **{reason: sum(reason in r[REMOVED_BY] for r in rm) for reason in REMOVED_BY_VALUES},
            })

    # passages behind MedHallu test removals (the old D5/D4a check: are they boilerplate?)
    seen = set()
    for r in removed_test:
        if r["source"] != MEDHALLU:
            continue
        for p in split_passages(r["context"]):
            k = passage_key(p)
            if k is not None and k in (clin_keys | gen_keys) and k not in seen:
                seen.add(k)
                report["medhallu_shared_passages"].append(p)

    splits = {"train": train, "val": val, "test": test, "test_full": test_full}
    for rows in splits.values():
        assert_schema(rows)
    assert_no_group_leakage(splits["train"], splits["val"], splits["test"])
    if len(test) + len(removed_test) != len(test_full) or len(val) + len(removed_val) != len(val_all):
        raise AssertionError("kept + removed must equal the rows before removal")

    for d in report["pqa_artificial_duplicates"]:
        gid = question_group_id(d["question"])
        d["splits_final"] = sorted({s for s, rows in splits.items() for r in rows if r["group_id"] == gid})

    # shares_passage_with_train on the full test (the same row objects appear in the clean test)
    clin_p = {k for k in clin_keys if k.startswith("p:")}
    gen_p = {k for k in gen_keys if k.startswith("p:")}
    flag = defaultdict(lambda: {"rows": 0, "flagged": 0, "clinical_train": 0, "general_train": 0, "in_clean_test": 0})
    for r in test_full:
        pk = p_keys(r)
        in_clin, in_gen = bool(pk & clin_p), bool(pk & gen_p)
        r[TEST_FLAG] = in_clin or in_gen
        f = flag[r["source"]]
        f["rows"] += 1
        f["flagged"] += r[TEST_FLAG]
        f["clinical_train"] += in_clin
        f["general_train"] += in_gen
    for r in test:
        flag[r["source"]]["in_clean_test"] += 1
        if r[TEST_FLAG]:
            raise AssertionError(f"clean test row flagged {TEST_FLAG}: {r['group_id']}")
    report["flag"] = dict(flag)
    return splits, {"val": removed_val, "test": removed_test}, report


def removal_reasons(row: dict, clin_keys: set[str], gen_keys: set[str]) -> list[str]:
    """removed_by for one removed row: which key kinds of this row matched which training part, or ["group"]."""
    keys = training_keys(row)
    out = [f"{kind}:{part}"
           for part, ref in (("clinical_train", clin_keys), ("general_train", gen_keys))
           for kind, prefix in (("question", "q:"), ("passage", "p:"))
           if any(k.startswith(prefix) and k in ref for k in keys)]
    return sorted(out, key=REMOVED_BY_VALUES.index) or ["group"]

# overlap_audit: check the final splits for any overlap that should have been removed, and report some that is allowed.
def overlap_audit(splits: dict, general_train: list[dict] | None = None, general_test: list[dict] | None = None,
                  verbose: bool = False) -> list[dict]:
    """Recount overlap on the final splits (keys = training_keys: question, or passage of >= 8 words).

    Asserted (must be 0), per source:
        val x clinical train, val x general train          (any key)
        test (clean) x clinical train, x general train     (any key)
        val x test and val x test_full                     (question; CLAUDE.md rule 2, groups)
    Reported only:
        test_full x clinical / general train               (any key; = the removed test rows)
        val x test and val x test_full                     (passage)
        clinical train x general test                      (question, passage; G+C)
    general_train=None leaves out the general-train lines (clinical train only).
    """
    sources = sorted({r["source"] for rows in splits.values() for r in rows})
    lines, failures = [], []
    refs = {"train/clinical": key_union(splits["train"], training_keys)}
    if general_train is not None:
        refs["train/general"] = key_union(general_train, training_keys)

    def kind_keys(row, kind):
        keys = training_keys(row)
        return keys if kind == "any" else {k for k in keys if k.startswith(kind + ":")}

    def add(a, b, kind, rows, ref, asserted):
        n = sum(bool(kind_keys(r, kind) & ref) for r in rows)
        lines.append({"a": a, "b": b, "kind": kind, "rows": n, "asserted": asserted})
        if asserted and n:
            failures.append(f"{a} x {b} ({kind}): {n} rows")

    for split, asserted in (("val", True), ("test", True), ("test_full", False)):
        for src in sources:
            rows = [r for r in splits.get(split, []) if r["source"] == src]
            for b, ref in refs.items():
                add(f"{split}/{src}", b, "any", rows, ref, asserted)
    for tsplit in ("test", "test_full"):
        if tsplit not in splits:
            continue
        tkeys = key_union(splits[tsplit], training_keys)
        for src in sources:
            vrows = [r for r in splits["val"] if r["source"] == src]
            add(f"val/{src}", tsplit, "q", vrows, tkeys, True)
            add(f"val/{src}", tsplit, "p", vrows, tkeys, False)
    if general_test is not None:
        gkeys = key_union(general_test, training_keys)
        for kind in ("q", "p"):
            add("train/clinical", "general test", kind, splits["train"], gkeys, False)
    if verbose:
        print("\n## Overlap audit (rows of A touching B)")
        print("| A | B | key | rows | asserted 0 |\n|---|---|---|---|---|")
        for x in lines:
            print(f"| {x['a']} | {x['b']} | {x['kind']} | {x['rows']} | {'yes' if x['asserted'] else 'no'} |")
    assert not failures, "overlap audit failed: " + "; ".join(failures)
    return lines


def shared_passage_relevance(raw: dict, splits: dict) -> dict:
    """D2(a): passages a RAGBench (full) test row shares with the train of the same subset:
    relevant (any of its sentence keys in the row's all_relevant_sentence_keys) or distractor."""
    out = {}
    for source in RAGBENCH_SOURCES.values():
        train_rows = [r for r in splits["train"] if r["source"] == source]
        train_ids = {r["_orig_id"] for r in train_rows}
        train_p = key_union(train_rows, p_keys)

        def doc_relevance(records):
            for rec in records:
                rel = {int(re.match(r"\d+", k).group()) for k in rec["all_relevant_sentence_keys"] or []}
                for i, d in enumerate(rec["documents"]):
                    k = passage_key(d)
                    if k is not None:  # short documents give no overlap key, so they are never "shared"
                        yield k, i in rel

        test_side = [(k, rel) for k, rel in doc_relevance(raw[source]["test"]) if k in train_p]
        test_shared = {k for k, _ in test_side}
        train_side = [(k, rel) for k, rel in doc_relevance([x for x in raw[source]["train"] if str(x["id"]) in train_ids])
                      if k in test_shared]
        rel_test = {k for k, rel in test_side if rel}
        rel_train = {k for k, rel in train_side if rel}
        out[source] = {
            "test_occurrences": len(test_side),
            "test_occ_relevant": sum(rel for _, rel in test_side),
            "distinct_passages": len(test_shared),
            "relevant_in_a_test_row": len(rel_test),
            "relevant_in_a_train_row": len(rel_train),
            "relevant_in_both": len(rel_test & rel_train),
            "distractor_everywhere": len(test_shared - rel_test - rel_train),
        }
    return out


def sample_handcheck(test_rows: list[dict], source: str, n_per_class: int = 25, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """D8: n_per_class hallucinated + n_per_class faithful test rows of one source, sha1-ranked, shuffled.
    Returns (labelling rows without label, key rows: row_id -> label and RAGBench id)."""
    rows = [r for r in test_rows if r["source"] == source]
    rank = lambda r, tag: sha1(f"{seed}:{tag}:{r['_orig_id']}:{r['answer']}")
    picked = []
    for label in (1, 0):
        pool = sorted((r for r in rows if r["label"] == label), key=lambda r: rank(r, "pick"))
        if len(pool) < n_per_class:
            raise ValueError(f"{source}: only {len(pool)} test rows with label {label}, need {n_per_class}")
        picked += pool[:n_per_class]
    picked.sort(key=lambda r: rank(r, "order"))
    short = source.replace("ragbench_", "")
    label_rows, key_rows = [], []
    for i, r in enumerate(picked, 1):
        rid = f"{short}-{i:03d}"
        label_rows.append({"row_id": rid, "question": r["query"], "context": r["context"], "response": r["answer"]})
        key_rows.append({"row_id": rid, "source": source, "ragbench_id": r["_orig_id"], "label": r["label"],
                         "answer_sha1": sha1(r["answer"])[:12]})
    return label_rows, key_rows


def public(r: dict, split: str) -> dict:
    """The columns written for `split` (train, val, test, test_full, removed_val, removed_test)."""
    cols = COLUMNS
    if split in ("test", "test_full", "removed_test"):
        cols += (TEST_FLAG,)
    if split.startswith("removed_"):
        cols += (REMOVED_BY,)
    return {k: r[k] for k in cols}


def write_split(rows: list[dict], out_dir: str, split: str) -> str:
    """{split}.jsonl, or {split}.jsonl.gz if the plain file would exceed MAX_PLAIN_MB."""
    clean = [public(r, split) for r in rows]
    size = sum(len(json.dumps(r, ensure_ascii=False).encode("utf-8")) + 1 for r in clean)
    path = os.path.join(out_dir, f"{split}.jsonl" + (".gz" if size > MAX_PLAIN_MB * 1e6 else ""))
    other = path[:-3] if path.endswith(".gz") else path + ".gz"
    if os.path.exists(other):
        print(f"WARNING: {other} also exists (stale from an earlier build?); not removed", file=sys.stderr)
    write_jsonl(clean, path)
    return path


def print_report(splits: dict, report: dict, relevance: dict) -> None:
    print("\n## Rows and prevalence (test = clean test; test_full = official test sets)")
    print("| split | source | rows | positives | prevalence |\n|---|---|---|---|---|")
    for split in OUT_SPLITS:
        c = Counter((r["source"], r["label"]) for r in splits[split])
        for src in sorted({s for s, _ in c}):
            n, pos = c[(src, 0)] + c[(src, 1)], c[(src, 1)]
            print(f"| {split} | {src} | {n} | {pos} | {pos / n:.1%} |")
    print("\n## Unlabelled rows dropped (D7)")
    for k, n in report["unlabelled"].items():
        if n:
            print(f"- {k}: {n}" + (" (PubMedQA validation = 2,449 rows; official 2,450)" if k == "ragbench_pubmedqa/val" else ""))
    print("\n## Removed from val / test for overlap with the training data (training never reduced)")
    print("A row can have several reasons; 'group' = removed only because another row of its group matched.")
    print("| split | source | rows before | positives before | removed | positives removed | "
          + " | ".join(REMOVED_BY_VALUES) + " |\n|" + "---|" * (6 + len(REMOVED_BY_VALUES)))
    for x in report["removals"]:
        print(f"| {x['split']} | {x['source']} | {x['rows_before']} | {x['positives_before']} | {x['removed']} | "
              f"{x['positives_removed']} | " + " | ".join(str(x[v]) for v in REMOVED_BY_VALUES) + " |")
    print("\n## Passages behind MedHallu test removals (first 150 chars)")
    for i, p in enumerate(report["medhallu_shared_passages"], 1):
        print(f"{i:2d}. {p[:150]!r}")
    if not report["medhallu_shared_passages"]:
        print("(none)")
    print("\n## pqa_artificial duplicate questions")
    for d in report["pqa_artificial_duplicates"]:
        print(f"- {d['question']!r}: {d['records']} records; after split {d['splits_after_split']}; final {d['splits_final'] or 'removed'}")
    print(f"\n## {TEST_FLAG} (full test)")
    print("| source | full test rows | flagged | in clinical train | in general train | rows in clean test |\n|---|---|---|---|---|---|")
    for src, f in sorted(report["flag"].items()):
        print(f"| {src} | {f['rows']} | {f['flagged']} | {f['clinical_train']} | {f['general_train']} | {f['in_clean_test']} |")
    print("\n## D2(a): RAGBench full-test passages shared with the same subset's train")
    keys = ["test_occurrences", "test_occ_relevant", "distinct_passages", "relevant_in_a_test_row",
            "relevant_in_a_train_row", "relevant_in_both", "distractor_everywhere"]
    print("| source | " + " | ".join(keys) + " |")
    print("|" + "---|" * (len(keys) + 1))
    for src, v in relevance.items():
        print(f"| {src} | " + " | ".join(str(v[k]) for k in keys) + " |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/clinical")
    ap.add_argument("--general-train", default="data/unified_v2/train.jsonl")
    ap.add_argument("--general-test", default="data/unified_v2/test.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--handcheck-n", type=int, default=25, help="per class, per RAGBench subset")
    args = ap.parse_args()

    refuse_protected_out_dir(args.out_dir)
    raw = load_raw()
    general_train, general_test = read_jsonl(args.general_train), read_jsonl(args.general_test)

    splits, removed, report = build(raw, general_train, seed=args.seed, val_frac=args.val_frac)
    audit = overlap_audit(splits, general_train, general_test, verbose=True)
    relevance = shared_passage_relevance(raw, splits)
    print_report(splits, report, relevance)

    written = [write_split(splits[s], args.out_dir, s) for s in OUT_SPLITS]
    written += [write_split(removed[s], args.out_dir, f"removed_{s}") for s in ("val", "test")]
    hc_dir = os.path.join(args.out_dir, "handcheck")
    for source in RAGBENCH_SOURCES.values():  # sampled from the FULL test set
        label_rows, key_rows = sample_handcheck(splits["test_full"], source, args.handcheck_n, args.seed)
        short = source.replace("ragbench_", "")
        for name, rows in ((f"{short}_label_me.jsonl", label_rows), (f"{short}_key.jsonl", key_rows)):
            write_jsonl(rows, os.path.join(hc_dir, name))
            written.append(os.path.join(hc_dir, name))

    meta = {"git_sha": git_sha(), "seed": args.seed, "val_frac": args.val_frac, "revisions": REVISIONS,
            "general_train": args.general_train, "general_test": args.general_test,
            "files": [os.path.relpath(p, args.out_dir) for p in written],
            **report, "audit": audit, "d2_relevance": relevance}
    report_path = os.path.join(args.out_dir, "build_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    written.append(report_path)

    print("\n## Files")
    print("| file | MB |\n|---|---|")
    for p in written:
        print(f"| {p} | {os.path.getsize(p) / 1e6:.1f} |")
    if any(p.endswith(".gz") for p in written):
        print("\nNOTE: a split was written as .jsonl.gz (over ~90 MB). scripts/mvp_check.py and "
              "scripts/report_truncation.py look for {split}.jsonl only: tell Prajakta before changing them.")


if __name__ == "__main__":
    main()
