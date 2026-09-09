"""Build the unified dataset the loader reads.

Turns RAGTruth and HaluEval into one schema and writes group-split train/val/test
files.

Unified row schema (every row from both sources ends up exactly like this):
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
          --out-dir           data/unified \
          --seed 42
"""

from __future__ import annotations
import argparse
import os

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from sklearn.model_selection import GroupShuffleSplit
from utils.io import read_jsonl, write_jsonl 

def load_halueval(path: str) -> list[dict]:
    """Convert HaluEval QA records into unified rows.

    Each record has: knowledge, question, right_answer, hallucinated_answer.

      - For each record emit TWO rows: one with right_answer (label 0) and one
        with hallucinated_answer (label 1). That is where the 50/50 balance comes
        from. 
      - Build `context` from `knowledge`
      - group_id: both rows from one record share a context, so they must share a
        group_id (e.g. a stable hash of `knowledge`, or the record index). If they
        don't, the faithful and hallucinated halves can leak across the split.
      - task = "qa" for HaluEval QA. source = "halueval".
    """
    rows = []
    for i, r in enumerate(read_jsonl(path)):
        knowledge = r["knowledge"]
        right_answer = r["right_answer"]
        hallucinated_answer = r["hallucinated_answer"]
        context = knowledge

        # hash the knowledge to create a stable group_id for both rows. This ensures that
        # both the faithful and hallucinated answers for the same knowledge/question pair
        # are kept together in the same split (train/val/test) and prevents data leakage    
        group_id = f"halueval-{i}"
        rows.append(
            {
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
        info[s["source_id"]] = {
            "context": s["source_info"],
            "task": TASK_MAP[s["task_type"]],
        }

    rows = []
    for r in read_jsonl(response_path):
        sid = r["source_id"]
        if sid not in info or info[sid]["task"] == "data2txt":
            continue
        rows.append({
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
    fields = {"context", "answer", "label", "source", "task", "group_id"}
    for r in rows:
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
    ap.add_argument("--out-dir", default="data/unified")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

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

    KEEP = {"context", "answer", "label", "source", "task", "group_id"}
    # write out the splits, keeping only the fields in KEEP
    for name, split in [("train", train), ("val", val), ("test", test)]:
        report(name, split)
        clean = [{k: r[k] for k in KEEP} for r in split]
        write_jsonl(clean, os.path.join(args.out_dir, f"{name}.jsonl"))

    print(f"wrote splits to {args.out_dir}")


if __name__ == "__main__":
    main()