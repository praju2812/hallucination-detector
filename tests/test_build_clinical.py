"""Spec for scripts/build_clinical.py (Phase 3, M2).

Decisions (Prajakta, 9 Nov; see the plan):
  D3  group_id = sha1 of the normalised question (lowercase, collapse whitespace, strip), no source prefix.
  D4  cross-dataset overlap = normalised question OR any shared passage (sha1 of normalised passage text).
      A passage gets a key only if its normalised text has >= min_words words (default 8; 0 keeps every
      non-empty passage), so boilerplate such as "None." is not overlap. build() and the audit use the default.
  D2/D9 within RAGBench: drop train/val rows whose question is in test; drop train rows whose question is in val.
  D5  pqa_artificial rows sharing a question or passage with pqa_labeled are dropped from pqa_artificial.
  D6  contexts are joined in RAGTruth QA style: "passage 1:{p1}\\n\\npassage 2:{p2}".
  D7  unlabelled rows dropped only where expected (RAGBench PubMedQA validation: 1); a test row never.
  D8  hand-check: 25 + 25 per RAGBench subset from test; labelling file has no label.
  Flag shares_passage_with_train on test rows: passage in final clinical train or general train (not val).

Toy records only: no network, no `datasets` import.
Run: python -m pytest tests/test_build_clinical.py
"""

import hashlib
import os
import subprocess
import sys

import pytest

from scripts.build_clinical import (
    TEST_FLAG,
    build,
    drop_overlap,
    drop_unlabelled,
    join_passages,
    medhallu_rows,
    overlap_audit,
    overlap_keys,
    passage_key,
    question_group_id,
    ragbench_label,
    ragbench_rows,
    sample_handcheck,
    split_passages,
    split_pqa_artificial,
)


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


def long_p(tag: str) -> str:
    """A passage long enough (11 words) to get an overlap key at the default min_words=8."""
    return f"{tag} is a passage with at least eight words in it."


# ---------------------------------------------------------------- toy raw records
def rb_record(rid, question, docs, response="It does.", adherence=True, relevant=("0a",)):
    return {
        "id": rid,
        "question": question,
        "documents": list(docs),
        "response": response,
        "adherence_score": adherence,
        "documents_sentences": [[[f"{i}a", d]] for i, d in enumerate(docs)],
        "all_relevant_sentence_keys": list(relevant),
    }


def mh_record(question, knowledge, gt="Yes, it does.", ha="No, never.", diff="hard", cat="Incomplete Information"):
    return {
        "Question": question,
        "Knowledge": list(knowledge),
        "Ground Truth": gt,
        "Hallucinated Answer": ha,
        "Difficulty Level": diff,
        "Category of Hallucination": cat,
    }


def row(query, passages, label=0, source="ragbench_pubmedqa", answer="a"):
    """A unified row built by hand (used where a test must not depend on other templates)."""
    return {
        "query": query,
        "context": "\n\n".join(f"passage {i + 1}:{p}" for i, p in enumerate(passages)),
        "answer": answer,
        "label": label,
        "source": source,
        "task": "qa",
        "group_id": sha1(" ".join(query.lower().split())),
        "difficulty": None,
        "category": None,
    }


# ---------------------------------------------------------------- 1. question_group_id (D3)
def test_group_id_is_sha1_of_normalised_question():
    assert question_group_id("Does X cause Y?") == sha1("does x cause y?")


def test_group_id_ignores_case_and_whitespace():
    assert question_group_id("  Does  X\ncause Y? ") == question_group_id("does x cause y?")


def test_group_id_same_question_across_sources():
    """Same question from RAGBench PubMedQA and MedHallu -> same group_id (so mvp_check sees cross-dataset leaks)."""
    q = "Is aspirin effective after stroke?"
    rb = ragbench_rows([rb_record("pubmedqa_1", q, ["d1"])], "ragbench_pubmedqa")[0]
    mh = medhallu_rows([mh_record(q.upper(), ["k1"])])[0]
    assert rb["group_id"] == mh["group_id"] == question_group_id(q)


def test_group_id_different_questions_differ():
    assert question_group_id("Is A true?") != question_group_id("Is B true?")


@pytest.mark.parametrize("q", ["", "   \n"])
def test_group_id_empty_question_raises(q):
    with pytest.raises(ValueError):
        question_group_id(q)


# ---------------------------------------------------------------- 2. ragbench_label
def test_label_false_is_hallucinated():
    assert ragbench_label(False) == 1


def test_label_true_is_faithful():
    assert ragbench_label(True) == 0


def test_label_none_is_unlabelled():
    assert ragbench_label(None) is None


@pytest.mark.parametrize("v", ["True", "False", 1, 0, 0.5])
def test_label_non_bool_raises(v):
    """datasets returns real bools; strings or numbers mean the loader changed, so stop."""
    with pytest.raises(ValueError):
        ragbench_label(v)


# ---------------------------------------------------------------- 3-4. join_passages / split_passages (D6)
def test_join_matches_ragtruth_style():
    assert join_passages(["Alpha.", "Beta."]) == "passage 1:Alpha.\n\npassage 2:Beta."


def test_join_single_passage():
    assert join_passages(["Only one."]) == "passage 1:Only one."


def test_join_strips_each_passage():
    """19 pqa_artificial Knowledge items have leading/trailing whitespace."""
    assert join_passages(["  Alpha. ", "Beta.\n"]) == "passage 1:Alpha.\n\npassage 2:Beta."


@pytest.mark.parametrize("bad", [[], ["ok", "  "]])
def test_join_empty_raises(bad):
    with pytest.raises(ValueError):
        join_passages(bad)


def test_split_inverts_join():
    docs = ["First doc.", "Second doc, with passage 3: inside.", "Third."]
    assert split_passages(join_passages(docs)) == docs


def test_split_keeps_blank_lines_inside_a_passage():
    """2 RAGBench PubMedQA train documents contain a blank line; that is not a passage boundary."""
    docs = ["Para one.\n\nPara two of the same doc.", "Next doc."]
    assert split_passages(join_passages(docs)) == docs


def test_split_plain_text_is_one_passage():
    """HaluEval knowledge and RAGTruth Summary contexts have no passage labels."""
    assert split_passages("Just some knowledge text.\n\nMore text.") == ["Just some knowledge text.\n\nMore text."]


def test_split_ragtruth_context():
    ctx = "passage 1:Butcher Shop - Hayward 826 B Street\n\npassage 2:Lakefield 212 Main Street"
    assert split_passages(ctx) == ["Butcher Shop - Hayward 826 B Street", "Lakefield 212 Main Street"]


# ---------------------------------------------------------------- 5. ragbench_rows
def test_ragbench_row_fields():
    rec = rb_record("pubmedqa_7", " Does X help? ", ["Doc A.", "Doc B."], response="Yes.", adherence=False)
    (r,) = ragbench_rows([rec], "ragbench_pubmedqa")
    assert r["query"] == "Does X help?"
    assert r["context"] == "passage 1:Doc A.\n\npassage 2:Doc B."
    assert r["answer"] == "Yes."
    assert r["label"] == 1
    assert r["source"] == "ragbench_pubmedqa"
    assert r["task"] == "qa"
    assert r["group_id"] == question_group_id("Does X help?")
    assert r["difficulty"] is None and r["category"] is None
    assert r["_orig_id"] == "pubmedqa_7"


def test_ragbench_one_row_per_record_in_order():
    recs = [rb_record(str(i), f"Question {i}?", [f"doc {i}"]) for i in range(5)]
    rows = ragbench_rows(recs, "ragbench_covidqa")
    assert [r["_orig_id"] for r in rows] == ["0", "1", "2", "3", "4"]
    assert all(r["source"] == "ragbench_covidqa" for r in rows)


def test_ragbench_unlabelled_record_kept_with_none():
    """Dropping is drop_unlabelled's job (with its guard), not ragbench_rows'."""
    (r,) = ragbench_rows([rb_record("1", "Q?", ["d"], adherence=None)], "ragbench_pubmedqa")
    assert r["label"] is None


def test_ragbench_empty_response_raises():
    with pytest.raises(ValueError):
        ragbench_rows([rb_record("1", "Q?", ["d"], response="  ")], "ragbench_pubmedqa")


# ---------------------------------------------------------------- 6. medhallu_rows
def test_medhallu_two_rows_per_question():
    rows = medhallu_rows([mh_record("Q1?", ["k"]), mh_record("Q2?", ["k2"])])
    assert len(rows) == 4
    assert sorted(r["label"] for r in rows) == [0, 0, 1, 1]


def test_medhallu_pair_shares_everything_but_answer_and_label():
    faithful, halluc = sorted(medhallu_rows([mh_record("Q1?", ["K one.", "K two."], gt="GT", ha="HA")]),
                              key=lambda r: r["label"])
    assert faithful["answer"] == "GT" and faithful["label"] == 0
    assert halluc["answer"] == "HA" and halluc["label"] == 1
    for k in ("query", "context", "source", "task", "group_id", "difficulty", "category"):
        assert faithful[k] == halluc[k], k
    assert faithful["context"] == "passage 1:K one.\n\npassage 2:K two."
    assert faithful["source"] == "medhallu" and faithful["task"] == "qa"
    assert faithful["group_id"] == question_group_id("Q1?")


def test_medhallu_difficulty_and_category_copied_to_faithful_row():
    rows = medhallu_rows([mh_record("Q?", ["k"], diff="easy", cat="Mechanism and Pathway Misattribution")])
    assert {r["difficulty"] for r in rows} == {"easy"}
    assert {r["category"] for r in rows} == {"Mechanism and Pathway Misattribution"}


@pytest.mark.parametrize("field", ["Ground Truth", "Hallucinated Answer"])
def test_medhallu_empty_answer_raises(field):
    rec = mh_record("Q?", ["k"])
    rec[field] = " "
    with pytest.raises(ValueError):
        medhallu_rows([rec])


# ---------------------------------------------------------------- 7. split_pqa_artificial
def _art_rows(n):
    return medhallu_rows([mh_record(f"Question number {i}?", [f"k{i}"]) for i in range(n)])


def test_split_artificial_deterministic():
    rows = _art_rows(200)
    a = split_pqa_artificial(rows, seed=42)
    b = split_pqa_artificial(rows, seed=42)
    assert a == b


def test_split_artificial_no_group_in_both_and_nothing_lost():
    rows = _art_rows(300)
    train, val = split_pqa_artificial(rows, seed=42)
    assert not {r["group_id"] for r in train} & {r["group_id"] for r in val}
    assert len(train) + len(val) == len(rows)


def test_split_artificial_about_ten_percent_val():
    rows = _art_rows(1000)
    _, val = split_pqa_artificial(rows, seed=42, val_frac=0.1)
    groups = {r["group_id"] for r in val}
    assert 70 <= len(groups) <= 130


def test_split_artificial_case_variants_land_together():
    """The one duplicated pqa_artificial question must not straddle train/val."""
    rows = medhallu_rows([mh_record("Does A cause B?", ["k1"]), mh_record("does a  cause b?", ["k2"])]) + _art_rows(50)
    train, val = split_pqa_artificial(rows, seed=42)
    gid = question_group_id("Does A cause B?")
    assert sum(r["group_id"] == gid for r in train) in (0, 4)
    assert sum(r["group_id"] == gid for r in val) in (0, 4)


def test_split_artificial_seed_matters():
    rows = _art_rows(300)
    assert split_pqa_artificial(rows, seed=42)[1] != split_pqa_artificial(rows, seed=43)[1]


def test_split_artificial_preserves_row_order():
    rows = _art_rows(100)
    train, _ = split_pqa_artificial(rows, seed=42)
    pos = {id(r): i for i, r in enumerate(rows)}
    idx = [pos[id(r)] for r in train]
    assert idx == sorted(idx)


def test_split_artificial_independent_of_pythonhashseed():
    """Hard rule 3: hashlib.sha1, never hash(). Same result under two PYTHONHASHSEED values."""
    code = (
        "from scripts.build_clinical import medhallu_rows, split_pqa_artificial\n"
        "rows = medhallu_rows([{'Question': f'Q{i}?', 'Knowledge': ['k'], 'Ground Truth': 'g',"
        " 'Hallucinated Answer': 'h', 'Difficulty Level': 'easy', 'Category of Hallucination': 'c'} for i in range(200)])\n"
        "print(sorted(r['group_id'] for r in split_pqa_artificial(rows, seed=42)[1]))\n"
    )
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outs = []
    for hs in ("0", "123"):
        env = {**os.environ, "PYTHONHASHSEED": hs}
        p = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True)
        assert p.returncode == 0, p.stderr[-2000:]
        outs.append(p.stdout)
    assert outs[0] == outs[1]


# ---------------------------------------------------------------- 8. overlap_keys (D4)
def test_overlap_keys_question_and_passages():
    r = row("Does X help?", [long_p("Doc A"), long_p("Doc B")])
    keys = overlap_keys(r)
    assert keys == {"q:" + question_group_id("Does X help?"),
                    "p:" + sha1(long_p("doc a")), "p:" + sha1(long_p("doc b"))}


def test_overlap_keys_passage_normalised_and_position_free():
    a = overlap_keys(row("Q1?", [long_p("Shared   Passage text"), long_p("other")]))
    b = overlap_keys(row("Q2?", [long_p("unrelated"), long_p("shared passage TEXT")]))
    assert {k for k in a & b} == {"p:" + sha1(long_p("shared passage text"))}


def test_overlap_keys_plain_context():
    r = row("Q?", ["x"])
    r["context"] = "Plain knowledge with no passage labels, long enough to count."
    assert overlap_keys(r) == {"q:" + question_group_id("Q?"),
                               "p:" + sha1("plain knowledge with no passage labels, long enough to count.")}


def test_overlap_keys_short_passage_skipped_by_default():
    """Default min_words=8: a short passage gets no key; the question key is unchanged."""
    assert overlap_keys(row("Q?", ["None."])) == {"q:" + question_group_id("Q?")}


def test_overlap_keys_short_shared_passage_not_overlap_by_default():
    """The D5 case: two different questions sharing only boilerplate ("None.") do not overlap."""
    a, b = row("Q1?", ["None.", "a1"]), row("Q2?", ["None.", "b1"])
    assert not overlap_keys(a) & overlap_keys(b)


def test_overlap_keys_short_shared_passage_is_overlap_with_min_words_0():
    a, b = row("Q1?", ["None.", "a1"]), row("Q2?", ["none.", "b1"])
    assert overlap_keys(a, min_words=0) & overlap_keys(b, min_words=0) == {"p:" + sha1("none.")}


@pytest.mark.parametrize("min_words", [0, 8])
def test_overlap_keys_long_shared_passage_is_overlap_either_way(min_words):
    shared = long_p("Shared")
    a, b = row("Q1?", [shared, "a1"]), row("Q2?", ["b1", shared])
    assert overlap_keys(a, min_words=min_words) & overlap_keys(b, min_words=min_words) == {"p:" + sha1(shared.lower())}


def test_overlap_keys_min_words_boundary():
    """Exactly min_words words counts; one fewer does not. Words are counted on the normalised text."""
    eight = "one two three four five six seven eight"
    seven = "one two three four five six seven"
    assert overlap_keys(row("Q?", [eight])) == {"q:" + question_group_id("Q?"), "p:" + sha1(eight)}
    assert overlap_keys(row("Q?", ["  " + seven.replace(" ", "\n  ") + "  "])) == {"q:" + question_group_id("Q?")}


def test_overlap_keys_min_words_0_keeps_every_nonempty_passage():
    r = row("Q?", ["x", "Y."])
    assert overlap_keys(r, min_words=0) == {"q:" + question_group_id("Q?"), "p:" + sha1("x"), "p:" + sha1("y.")}


def test_passage_key_none_for_short_passage():
    """Plumbing: passage_key follows overlap_keys' default, and returns None for a skipped passage."""
    assert passage_key("None.") is None
    assert passage_key(long_p("Some")) == "p:" + sha1(long_p("some"))


# ---------------------------------------------------------------- 9. drop_overlap
def test_drop_overlap_by_question():
    rows = [row("Shared question?", ["a"]), row("Other?", ["b"])]
    ref = [row("shared  QUESTION?", ["z"], source="medhallu")]
    kept, removed = drop_overlap(rows, ref)
    assert [r["query"] for r in kept] == ["Other?"]
    assert [r["query"] for r in removed] == ["Shared question?"]


def test_drop_overlap_by_passage():
    rows = [row("Q1?", ["clean"]), row("Q2?", [long_p("dirty passage"), "clean2"])]
    ref = [row("Q9?", [long_p("Dirty passage")], source="medhallu")]
    kept, removed = drop_overlap(rows, ref)
    assert [r["query"] for r in kept] == ["Q1?"] and len(removed) == 1


def test_drop_overlap_short_passage_alone_does_not_remove():
    rows = [row("Q1?", ["None.", "clean"])]
    kept, removed = drop_overlap(rows, [row("Q9?", ["None."], source="medhallu")])
    assert len(kept) == 1 and not removed


def test_drop_overlap_removes_whole_group():
    """Two rows of one group; only one shares a passage -> both go (the plan: whole groups)."""
    rows = [row("Q1?", [long_p("dirty")]), row("Q1?", ["clean"], answer="b"), row("Q2?", ["fine"])]
    kept, removed = drop_overlap(rows, [row("Z?", [long_p("dirty")])])
    assert [r["query"] for r in kept] == ["Q2?"] and len(removed) == 2


def test_drop_overlap_reference_untouched_and_order_kept():
    rows = [row(f"Q{i}?", [f"p{i}"]) for i in range(6)]
    ref = [row("Q3?", ["zzz"])]
    ref_before = [dict(r) for r in ref]
    kept, removed = drop_overlap(rows, ref)
    assert ref == ref_before
    assert [r["query"] for r in kept] == ["Q0?", "Q1?", "Q2?", "Q4?", "Q5?"]
    assert len(kept) + len(removed) == len(rows)


def test_drop_overlap_custom_keys_question_only():
    """D2/D9 use a question-only key: passage sharing alone does not remove a row."""
    q_only = lambda r: {k for k in overlap_keys(r) if k.startswith("q:")}
    rows = [row("Q1?", [long_p("shared")]), row("Q2?", ["x"])]
    kept, removed = drop_overlap(rows, [row("Q2?", [long_p("shared")])], keys=q_only)
    assert [r["query"] for r in kept] == ["Q1?"] and [r["query"] for r in removed] == ["Q2?"]


# ---------------------------------------------------------------- drop_unlabelled guard (FULL, D7)
def _lab(labels):
    return [dict(row(f"Q{i}?", ["p"]), label=l) for i, l in enumerate(labels)]


def test_unlabelled_expected_count_dropped():
    kept, n = drop_unlabelled(_lab([0, None, 1]), "ragbench_pubmedqa", "val")
    assert n == 1 and [r["label"] for r in kept] == [0, 1]


def test_unlabelled_wrong_count_raises():
    with pytest.raises(ValueError):
        drop_unlabelled(_lab([None, None, 1]), "ragbench_pubmedqa", "val")
    with pytest.raises(ValueError):
        drop_unlabelled(_lab([0, 1]), "ragbench_pubmedqa", "val")  # expected 1, found 0
    with pytest.raises(ValueError):
        drop_unlabelled(_lab([None, 1]), "ragbench_covidqa", "train")


def test_unlabelled_test_row_always_raises():
    with pytest.raises(ValueError):
        drop_unlabelled(_lab([None, 1]), "ragbench_pubmedqa", "test", expected={("ragbench_pubmedqa", "test"): 1})


def test_unlabelled_none_expected_none_found():
    kept, n = drop_unlabelled(_lab([0, 1]), "ragbench_covidqa", "test")
    assert n == 0 and len(kept) == 2


# ---------------------------------------------------------------- hand-check sample (FULL, D8)
def _test_rows(n_pos, n_neg, source="ragbench_covidqa"):
    out = []
    for i in range(n_pos + n_neg):
        r = row(f"Question {i}?", [f"doc {i}"], label=int(i < n_pos), source=source, answer=f"answer {i}")
        r["_orig_id"] = str(i)
        out.append(r)
    return out


def test_handcheck_25_25_and_no_label_in_labelling_file():
    label_rows, key_rows = sample_handcheck(_test_rows(39, 207), "ragbench_covidqa", n_per_class=25, seed=42)
    assert len(label_rows) == len(key_rows) == 50
    assert sum(k["label"] for k in key_rows) == 25
    for r in label_rows:
        assert set(r) == {"row_id", "question", "context", "response"}
    assert {r["row_id"] for r in label_rows} == {k["row_id"] for k in key_rows}
    assert all(k["source"] == "ragbench_covidqa" and k["ragbench_id"] is not None for k in key_rows)


def test_handcheck_deterministic_and_shuffled():
    rows = _test_rows(60, 60)
    a = sample_handcheck(rows, "ragbench_covidqa", seed=42)
    assert a == sample_handcheck(rows, "ragbench_covidqa", seed=42)
    labels = [k["label"] for k in sorted(a[1], key=lambda k: k["row_id"])]
    assert labels != sorted(labels) and labels != sorted(labels, reverse=True), "labels must not be blocked by row id"


def test_handcheck_only_that_source():
    rows = _test_rows(30, 30) + _test_rows(30, 30, source="ragbench_pubmedqa")
    _, keys = sample_handcheck(rows, "ragbench_pubmedqa", seed=42)
    assert {k["source"] for k in keys} == {"ragbench_pubmedqa"}


def test_handcheck_too_few_raises():
    with pytest.raises(ValueError):
        sample_handcheck(_test_rows(10, 100), "ragbench_covidqa", n_per_class=25, seed=42)


# ---------------------------------------------------------------- build() end to end on toy data
SHARED_RB, MH_TEST_K, VAL_ONLY, RB_VAL_P, GENERAL_P = (
    long_p(t) for t in ("SHARED-RB", "MH-TEST-K", "VAL-ONLY", "RB-VAL-P", "GENERAL-P"))


def toy_raw():
    """Planted cases:
    pubmedqa train  'Leak to test?'        same question as a pubmedqa test row   -> D2 drop
    pubmedqa train  'Leak to val?'         same question as a pubmedqa val row    -> D9 drop (from train)
    pubmedqa train  'Shares a passage?'    passage 'SHARED-RB' also in pubmedqa test -> kept, test row flagged
    pubmedqa train  'Touches MedHallu?'    passage 'MH-TEST-K' from pqa_labeled  -> D4 drop
    pubmedqa val    one unlabelled row                                            -> D7 drop (expected 1)
    pubmedqa val    passage 'VAL-ONLY' also in a pubmedqa test row                -> no flag (val excluded)
    covidqa test    passage 'GENERAL-P' also in general train                     -> flagged (general)
    pqa_artificial  'Art touches test?'    passage 'MH-TEST-K'                    -> D5 drop
    pqa_artificial  'Art touches val?'     passage 'RB-VAL-P' from pubmedqa val   -> D4 drop
    pqa_artificial  'Art boilerplate?'     only shares 'None.' with pqa_labeled   -> kept (passage < 8 words)
    pubmedqa train  'Clean train?'         only shares 'None.' with pqa_labeled   -> kept (passage < 8 words)
    pubmedqa test   'Boilerplate test?'    only shares 'None.' with clinical train ('Covid train?') -> not flagged
    Shared passages are 8+ words (long_p) so they count at the default min_words; 'None.' is not.
    """
    pub = {
        "train": [
            rb_record("pubmedqa_1", "Leak to test?", ["t1"]),
            rb_record("pubmedqa_2", "Leak to val?", ["t2"]),
            rb_record("pubmedqa_3", "Shares a passage?", [SHARED_RB, "t3"], adherence=False),
            rb_record("pubmedqa_4", "Touches MedHallu?", [MH_TEST_K]),
            rb_record("pubmedqa_5", "Clean train?", ["t5", "None."]),
        ],
        "val": [
            rb_record("pubmedqa_6", "Leak to val?", ["v1"]),
            rb_record("pubmedqa_7", "No label?", ["v2"], adherence=None),
            rb_record("pubmedqa_8", "Val passage?", [VAL_ONLY, RB_VAL_P], adherence=False),
        ],
        "test": [
            rb_record("pubmedqa_9", "Leak to test?", ["x1"]),
            rb_record("pubmedqa_10", "Test sharing?", [SHARED_RB, "x2"], adherence=False),
            rb_record("pubmedqa_11", "Test clean?", ["x3"]),
            rb_record("pubmedqa_12", "Val-only passage?", [VAL_ONLY]),
            rb_record("pubmedqa_13", "Boilerplate test?", ["None.", "x4"]),
        ],
    }
    cov = {
        "train": [rb_record("1", "Covid train?", ["c1", "None."])],
        "val": [rb_record("2", "Covid val?", ["c2"], adherence=False)],
        "test": [rb_record("3", "Covid general?", [GENERAL_P], adherence=False)],
    }
    labeled = [mh_record("MedHallu test?", [MH_TEST_K, "m2"]), mh_record("MedHallu test 2?", ["m3", "None."])]
    artificial = [
        mh_record("Art touches test?", [MH_TEST_K]),
        mh_record("Art touches val?", [RB_VAL_P]),
        mh_record("Art boilerplate?", ["None.", "ab"]),
    ] + [mh_record(f"Art clean {i}?", [f"a{i}"]) for i in range(20)]
    return {"ragbench_pubmedqa": pub, "ragbench_covidqa": cov, "pqa_labeled": labeled, "pqa_artificial": artificial}


GENERAL_TRAIN = [
    {"query": "General q?", "context": f"passage 1:{GENERAL_P}\n\npassage 2:g2", "answer": "a", "label": 0,
     "source": "ragtruth", "task": "qa", "group_id": "rt-1"},
    {"query": "Halu q?", "context": "Plain knowledge.", "answer": "a", "label": 1,
     "source": "halueval", "task": "qa", "group_id": "halueval-1"},
]


@pytest.fixture(scope="module")
def built():
    return build(toy_raw(), GENERAL_TRAIN, seed=42, val_frac=0.2)


def _queries(rows, source=None):
    return {r["query"] for r in rows if source is None or r["source"] == source}


def test_build_schema(built):
    splits, _ = built
    cols = {"query", "context", "answer", "label", "source", "task", "group_id", "difficulty", "category"}
    for name, rows in splits.items():
        assert rows, name
        for r in rows:
            assert cols <= set(r), (name, cols - set(r))
            assert r["label"] in (0, 1)
            assert r["source"] in ("ragbench_pubmedqa", "ragbench_covidqa", "medhallu")
    assert all(TEST_FLAG in r for r in splits["test"])


def test_build_no_group_across_splits(built):
    splits, _ = built
    g = {n: {r["group_id"] for r in rows} for n, rows in splits.items()}
    assert not (g["train"] & g["val"]) and not (g["train"] & g["test"]) and not (g["val"] & g["test"])


def test_build_test_rows_never_removed(built):
    splits, _ = built
    raw = toy_raw()
    assert len([r for r in splits["test"] if r["source"] == "ragbench_pubmedqa"]) == 5
    assert len([r for r in splits["test"] if r["source"] == "ragbench_covidqa"]) == 1
    assert len([r for r in splits["test"] if r["source"] == "medhallu"]) == 2 * len(raw["pqa_labeled"])


def test_build_planted_drops(built):
    splits, _ = built
    tr = _queries(splits["train"], "ragbench_pubmedqa")
    assert "Leak to test?" not in tr          # D2
    assert "Leak to val?" not in tr           # D9 (dropped from train)
    assert "Leak to val?" in _queries(splits["val"], "ragbench_pubmedqa")
    assert "Touches MedHallu?" not in tr      # D4
    assert {"Shares a passage?", "Clean train?"} <= tr   # passage sharing within RAGBench is kept; 'None.' is not D4
    assert "No label?" not in _queries(splits["val"])    # D7
    mh = _queries(splits["train"] + splits["val"], "medhallu")
    assert "Art touches test?" not in mh      # D5
    assert "Art touches val?" not in mh       # D4
    assert "Art boilerplate?" in mh           # D5 ignores the short shared passage 'None.'
    assert len(mh) == 21


def test_build_flag(built):
    splits, _ = built
    flag = {r["query"]: r[TEST_FLAG] for r in splits["test"]}
    assert flag["Test sharing?"] is True       # passage in clinical train
    assert flag["Covid general?"] is True      # passage in general train
    assert flag["Val-only passage?"] is False  # passage only in val: not counted
    assert flag["Test clean?"] is False
    assert flag["Boilerplate test?"] is False  # shares only 'None.' with train
    assert all(isinstance(v, bool) for v in flag.values())


def test_build_report_counts(built):
    _, report = built
    assert isinstance(report, dict) and report, "build returns a report of what was removed"


def test_build_audit_passes(built):
    splits, _ = built
    overlap_audit(splits)  # raises AssertionError on any asserted overlap


def test_audit_catches_cross_dataset_leak(built):
    splits, _ = built
    bad = {k: list(v) for k, v in splits.items()}
    leak = next(r for r in splits["test"] if r["source"] == "medhallu")
    bad["train"].append(dict(leak, source="ragbench_pubmedqa", query="Fresh question?",
                             group_id=question_group_id("Fresh question?")))
    with pytest.raises(AssertionError):
        overlap_audit(bad)
