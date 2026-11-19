"""Phase 6 plumbing: pair building and batching (scripts/score_nli.py) and outputs of scripts/report_nli.py.

No model download: a fake word-level tokenizer and a fake 3-label NLI model stand in.
Tests marked "needs bodies" call the src/nli.py templates, so they pass only once those bodies are written.
The pair-layout, M5-shape and report tests run now.

Run: python -m pytest tests/test_nli_plumbing.py
"""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import scripts.mvp_check as mvp                 # module import: pytest must not collect mvp.test_block as a test
from scripts.report_nli import build_m5_summary, main as report_main
from scripts.score_nli import (
    PairBuilder,
    determinism_check,
    make_records,
    n_pairs,
    prepare_row,
    score_prepared,
    split_counts,
)
from src.rq1 import assign_row_keys

CLS, SEP, PAD = 1, 2, 0


class FakeTokenizer:
    """Word-level: one id per lower-cased word (punctuation stripped). Pair layout [CLS] a [SEP] b [SEP]."""
    cls_token_id, sep_token_id, pad_token_id = CLS, SEP, PAD
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self):
        self.vocab = {}

    def _ids(self, text):
        return [self.vocab.setdefault(w.lower().strip(".,:;!?*-"), len(self.vocab) + 10)
                for w in text.split() if w.strip(".,:;!?*-")]

    def __call__(self, text, text_pair=None, add_special_tokens=True):
        a = self._ids(text)
        if not add_special_tokens:
            ids = a
        elif text_pair is None:
            ids = [CLS] + a + [SEP]
        else:
            ids = [CLS] + a + [SEP] + self._ids(text_pair) + [SEP]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}       # like a real tokenizer

    def num_special_tokens_to_add(self, pair=False):
        return 3 if pair else 2


class FakeNLI:
    """P(entailment) rises with the share of hypothesis words found in the premise window.
    Entailment is label 1 here, so a hard-coded index 0 would fail the tests."""
    config = SimpleNamespace(id2label={0: "contradiction", 1: "entailment", 2: "neutral"})

    def __call__(self, input_ids, attention_mask, token_type_ids=None):
        rows = []
        for ids, m in zip(input_ids.tolist(), attention_mask.tolist()):
            ids = [t for t, k in zip(ids, m) if k]
            p1 = ids.index(SEP)
            p2 = ids.index(SEP, p1 + 1)
            premise, hyp = set(ids[1:p1]), ids[p1 + 1:p2]
            share = sum(t in premise for t in hyp) / len(hyp)
            rows.append([0.0, 8.0 * share - 4.0, 0.0])
        return SimpleNamespace(logits=torch.tensor(rows))


# ---- pair layout (runs now) ---------------------------------------------------------------------

def test_pair_builder_layout_verified_against_tokenizer():
    b = PairBuilder(FakeTokenizer())
    info = b.verify()
    assert info["verified"] and info["n_special"] == 3
    assert info["keys_compared"] == ["attention_mask", "input_ids"]           # every key the tokenizer returns
    ids, tt = b.build([10, 11], [12])
    assert ids == [CLS, 10, 11, SEP, 12, SEP] and tt == [0, 0, 0, 0, 1, 1]


class _PairOverride(FakeTokenizer):
    """FakeTokenizer whose PAIR output is rewritten by self.edit(out, a_ids, b_ids)."""
    def __call__(self, text, text_pair=None, add_special_tokens=True):
        out = super().__call__(text, text_pair, add_special_tokens)
        if text_pair is not None and add_special_tokens:
            out = self.edit(out, self._ids(text), self._ids(text_pair))
        return out


def test_pair_builder_rejects_a_different_layout():
    class RobertaStyle(_PairOverride):
        def edit(self, out, a, b):
            ids = [CLS] + a + [SEP, SEP] + b + [SEP]
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}
    with pytest.raises(RuntimeError, match="input_ids differs"):
        PairBuilder(RobertaStyle()).verify()


def test_pair_builder_compares_attention_mask():
    class OddMask(_PairOverride):
        def edit(self, out, a, b):
            return {**out, "attention_mask": [1] * (len(out["input_ids"]) - 1) + [0]}
    with pytest.raises(RuntimeError, match="attention_mask differs"):
        PairBuilder(OddMask()).verify()


def test_pair_builder_token_type_ids_returned_but_not_a_model_input():
    class ExtraTypes(_PairOverride):                     # model_input_names has no token_type_ids
        def edit(self, out, a, b):
            return {**out, "token_type_ids": [0] * (len(a) + 2) + [1] * (len(b) + 1)}
    with pytest.raises(RuntimeError, match="keys differ"):
        PairBuilder(ExtraTypes()).verify()


def test_pair_builder_compares_token_type_ids_when_used():
    class WithTypes(_PairOverride):
        model_input_names = ["input_ids", "token_type_ids", "attention_mask"]
        def edit(self, out, a, b):
            return {**out, "token_type_ids": [0] * (len(a) + 2) + [1] * (len(b) + 1)}
    info = PairBuilder(WithTypes()).verify()
    assert info["keys_compared"] == ["attention_mask", "input_ids", "token_type_ids"] and info["token_type_ids"]

    class WrongTypes(WithTypes):
        def edit(self, out, a, b):
            return {**out, "token_type_ids": [0] * len(out["input_ids"])}
    with pytest.raises(RuntimeError, match="token_type_ids differs"):
        PairBuilder(WrongTypes()).verify()


# ---- scoring loop with the fake model (needs bodies) ---------------------------------------------

def _row(answer, context, label=0, source="ragbench_pubmedqa", task="qa", gid="g", query="What happened here?"):
    return {"answer": answer, "context": context, "label": label, "source": source, "task": task,
            "group_id": gid, "query": query}


CTX = "the patient took aspirin daily for pain relief and recovered well"


def test_score_supported_vs_unsupported_rows():
    b = PairBuilder(FakeTokenizer())
    rows = [_row("The patient took aspirin daily. The patient recovered well.", CTX),     # every word is in CTX
            _row("The patient took aspirin daily. The patient died yesterday suddenly.", CTX, label=1)]
    prepared = [prepare_row(r, b, False, stride=4, max_length=64, hyp_cap=16) for r in rows]
    res, st = score_prepared(prepared, FakeNLI(), b, ent_idx=1, batch_size=3, device="cpu")
    assert st["pairs"] == sum(n_pairs(p) for p in prepared) == 4
    assert res[0]["prob"] < 0.1 < 0.5 < res[1]["prob"]                       # min over sentences drives row 2 up
    assert res[1]["sentence_support"][0] > 0.9 > res[1]["sentence_support"][1]
    assert res[0]["n_sentences"] == 2 and res[0]["n_windows"] == 1


def test_support_found_in_a_later_window_counts():
    b = PairBuilder(FakeTokenizer())
    filler = " ".join(f"w{i}" for i in range(80))
    row = _row("The rare drug cured the disease.", filler + " the rare drug cured the disease")
    p = prepare_row(row, b, False, stride=4, max_length=24, hyp_cap=8)
    assert len(p["spans"][0]) > 1                                             # several windows
    res, _ = score_prepared([p], FakeNLI(), b, ent_idx=1, batch_size=2, device="cpu")
    assert res[0]["prob"] < 0.1                                               # max over windows finds the support
    assert res[0]["n_windows"] == len(p["spans"][0])


def test_batch_size_does_not_change_scores():
    b = PairBuilder(FakeTokenizer())
    rows = [_row("The patient took aspirin daily. The patient died yesterday suddenly.", CTX * 3),
            _row("Aspirin gave pain relief to the patient.", CTX)]
    prepared = [prepare_row(r, b, False, stride=4, max_length=20, hyp_cap=8) for r in rows]
    r1, _ = score_prepared(prepared, FakeNLI(), b, 1, batch_size=1, device="cpu")
    r7, _ = score_prepared(prepared, FakeNLI(), b, 1, batch_size=7, device="cpu")
    assert [x["prob"] for x in r1] == pytest.approx([x["prob"] for x in r7], abs=1e-6)


def test_question_format_changes_only_qa_rows():
    b = PairBuilder(FakeTokenizer())
    qa = _row("Aspirin gave pain relief.", CTX, query="Did aspirin help the patient?")
    sm = _row("Aspirin gave pain relief.", CTX, task="summary", query="Summarize the article.")
    assert prepare_row(qa, b, True, 4, 64, 16)["hyps"] != prepare_row(qa, b, False, 4, 64, 16)["hyps"]
    assert prepare_row(sm, b, True, 4, 64, 16)["hyps"] == prepare_row(sm, b, False, 4, 64, 16)["hyps"]


def test_fallback_and_capped_counts():
    b = PairBuilder(FakeTokenizer())
    long_sentence = " ".join(f"x{i}" for i in range(30)) + "."
    rows = [_row("---\n***", CTX), _row(long_sentence, CTX), _row("Aspirin gave pain relief.", CTX)]
    prepared = [prepare_row(r, b, False, stride=4, max_length=40, hyp_cap=16) for r in rows]
    assert [p["fell_back"] for p in prepared] == [True, False, False]
    assert [p["n_capped"] for p in prepared] == [0, 1, 0]
    assert all(40 - len(h) - 3 > 4 for p in prepared for h in p["hyps"])      # budget stays above the stride
    c = split_counts({"rows": rows}, prepared)["ragbench_pubmedqa"]
    assert c["rows"] == 3 and c["fallback_rows"] == 1 and c["capped_hypotheses"] == 1


def test_determinism_check_and_records():
    b = PairBuilder(FakeTokenizer())
    rows = [_row("Aspirin gave pain relief.", CTX, gid="g1"), _row("The patient died suddenly at home.", CTX, 1, gid="g2")]
    prepared = [prepare_row(r, b, False, 4, 64, 16) for r in rows]
    det = determinism_check(prepared, FakeNLI(), b, 1, 2, "cpu")
    assert det == {"n": 2, "max_abs_diff": 0.0, "identical": True}
    res, _ = score_prepared(prepared, FakeNLI(), b, 1, 2, "cpu")
    split = {"name": "clinical_test_full", "rows": rows, "idx": [5, 9], "keys": assign_row_keys(rows)}
    recs = make_records(split, res)
    assert [r["idx"] for r in recs] == [5, 9] and recs[1]["label"] == 1 and recs[0]["split"] == "clinical_test_full"
    assert {"row_key", "prob", "sentence_support", "n_capped", "fell_back", "difficulty", "category"} <= set(recs[0])


# ---- inspection: preamble-like sentences ----------------------------------------------------------

def test_preamble_rules_match_the_definition():
    from scripts.inspect_nli_inputs import preamble_rules
    # "here is" mid-sentence no longer counts; the colon and the leading "Based on ..." do
    assert preamble_rules("Based on the given passages, here is a brief answer to the question:") == \
        ["ends_with_colon", "based_on"]
    assert preamble_rules("BASED ON CONTEXT the drug works.") == ["based_on"]                 # case-insensitive
    assert preamble_rules("  Based on provided information, it is safe.") == ["based_on"]      # leading spaces allowed
    assert preamble_rules("Symptoms include the following:  ") == ["ends_with_colon"]
    assert preamble_rules("The drug reduced mortality in adults.") == []
    assert preamble_rules("Based on these passages, the drug works.") == []                   # "these" is not in the pattern


def test_preamble_rules_anchored_with_word_boundaries():
    from scripts.inspect_nli_inputs import preamble_rules
    assert preamble_rules("Here is a summary of the trial findings.") == ["here_is"]           # leading "Here is"
    assert preamble_rules("here are the main symptoms") == ["here_is"]
    assert preamble_rules("There is a strong correlation between the two.") == []             # "there is"
    assert preamble_rules("There are three trials in the review.") == []
    assert preamble_rules("The method we present here is able to purify complexes.") == []    # mid-sentence "here is"
    assert preamble_rules("The answer, based on the context, is yes.") == []                  # mid-sentence "based on"
    assert preamble_rules("Based on contextual cues, the drug works.") == []                  # \b: "contextual"
    assert preamble_rules("Hereafter is the plan.") == []


def test_preamble_stats_counts_rows_and_sentences():          # needs bodies (split_response_sentences)
    from scripts.inspect_nli_inputs import preamble_stats
    rows = [{"answer": "Based on the given passages, here is the answer:\n* The drug reduced mortality in adults."},
            {"answer": "The drug reduced mortality in adults. Side effects were mild overall."},
            {"answer": "Here are the findings of the trial. The drug works well."}]
    ps = preamble_stats(rows)
    assert ps["rows"] == 3 and ps["rows_with"] == 2 and ps["sentences"] == 6
    assert [rules for _, rules in ps["matches"]] == [["ends_with_colon", "based_on"], ["here_is"]]


# ---- M5 summary shape (runs now) ---------------------------------------------------------------

def _m(prev=0.2, pr=0.5):
    blk = lambda t: {"threshold": t, "precision": 0.5, "recall": 0.5, "f1": 0.4, "faithful_precision": 0.9}
    return {"n": 10, "n_pos": int(10 * prev), "prevalence": prev, "pr_auc": pr, "roc_auc": 0.6,
            "at_ragtruth_thr": blk(0.3), "at_clinical_thr": blk(0.2)}


def test_m5_summary_is_found_by_mvp_check(tmp_path, monkeypatch):
    srcs = ["ragbench_pubmedqa", "ragbench_covidqa", "medhallu"]
    metrics = {"model": "x/y", "hypothesis": "response", "ragtruth_test": _m(),
               "test_full": {s: _m() for s in srcs}, "test_clean": {s: _m() for s in srcs}}
    summ = build_m5_summary(metrics, "results/nli/y/metrics.json")
    assert "per_source" not in summ
    found = {mvp.clinical_key(k) or k.lower() for k in mvp.test_block(summ)}
    assert found == {"ragtruth", *mvp.CLINICAL_KEYS}
    (tmp_path / "y.json").write_text(json.dumps(summ))
    monkeypatch.setattr(mvp, "NLI_DIR", tmp_path)
    mvp.check_m5()
    assert mvp.RESULTS["M5"] == "PASS"


# ---- report end to end on toy predictions (runs now: uses src/rq1.py, not the NLI templates) -------------

SRCS = ["ragbench_pubmedqa", "ragbench_covidqa", "medhallu"]


def _write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _toy_rows(split, sources, n, tasks=("qa",)):
    rows = []
    for s in sources:
        for t in tasks:
            for i in range(n):
                y = int(i % 2 == 0) if s == "medhallu" else int(i % 4 == 0)
                rows.append({"answer": f"{split}-{s}-{t}-{i}", "label": y, "source": s, "task": t,
                             "group_id": f"{s}-{split}-{t}-g{i}"})
    return rows


def _make_toy(tmp_path, limit=None):
    rng = np.random.default_rng(0)
    data, nli = tmp_path / "clinical", tmp_path / "nli"
    splits = {"ragtruth_val": _toy_rows("rv", ["ragtruth"], 30, ("qa", "summary")),
              "ragtruth_test": _toy_rows("rt", ["ragtruth"], 40, ("qa", "summary")),
              "clinical_val": _toy_rows("cv", SRCS, 40),
              "clinical_test_full": _toy_rows("cf", SRCS, 48)}
    clean = [r for i, r in enumerate(splits["clinical_test_full"]) if i % 3 != 0]
    _write_jsonl(data / "test.jsonl", clean)
    preds = []
    for split, rows in splits.items():
        for i, (r, k) in enumerate(zip(rows, assign_row_keys(rows))):
            preds.append({"split": split, "idx": i, "source": r["source"], "task": r["task"], "group_id": r["group_id"],
                          "label": r["label"], "prob": float(np.clip(rng.normal(0.4 + 0.3 * r["label"], 0.15), 0, 1)),
                          "row_key": k, "difficulty": None, "category": None, "n_sentences": 1, "n_windows": 1,
                          "fell_back": False, "n_capped": 0, "sentence_support": [0.5]})
    _write_jsonl(nli / "toy-model" / "predictions.jsonl", preds)
    scoring = {"model": "org/toy-model", "revision": "abc123", "hypothesis": "response", "include_question": False,
               "max_length": 512, "stride": 128, "hyp_cap": 256, "min_words": 3,
               "splitter": {"name": "pysbd", "version": "0.3.4"}, "device": "cpu", "device_name": "cpu",
               "precision": {"param_dtype": "torch.float32", "autocast": False}, "git_sha": "abc", "git_dirty": False,
               "limit": limit, "torch_version": "t", "transformers_version": "tf", "pairs_scored": 123,
               "data_files": {s: {"path": s, "rows_expected": len(r)} for s, r in splits.items()},
               "counts": {"ragtruth_test": {"ragtruth": {"rows": 80, "fallback_rows": 1, "capped_hypotheses": 2}}},
               "determinism": {"n": 20, "max_abs_diff": 0.0, "identical": True}}
    (nli / "toy-model" / "scoring.json").write_text(json.dumps(scoring))
    return data, nli


def test_report_end_to_end(tmp_path, monkeypatch):
    data, nli = _make_toy(tmp_path)
    md = tmp_path / "nli_baseline.md"
    rc = report_main(["--nli-root", str(nli), "--model-dir", "toy-model", "--data-dir", str(data), "--out-md", str(md)])
    assert rc == 0
    m = json.loads((nli / "toy-model" / "metrics.json").read_text())
    assert m["ragtruth_test"]["n"] == 80 and set(m["ragtruth_test_by_task"]) == {"qa", "summary"}
    assert set(m["test_full"]) == set(SRCS) and m["test_full"]["ragbench_pubmedqa"]["n"] == 48
    assert m["test_clean"]["ragbench_pubmedqa"]["n"] == 32
    a = m["thresholds"]["ragtruth_val"]["threshold"]
    assert m["ragtruth_test"]["at_ragtruth_thr"]["threshold"] == m["ragtruth_test"]["at_clinical_thr"]["threshold"] == a
    assert all(m["test_full"][s]["at_ragtruth_thr"]["threshold"] == a for s in SRCS)          # (a) on every source
    assert m["thresholds"]["clinical_val"]["ragbench_covidqa"]["tuned_on"] == ["ragbench_covidqa", "ragbench_pubmedqa"]
    # M5 summary next to the folder, found by mvp_check
    monkeypatch.setattr(mvp, "NLI_DIR", nli)
    mvp.check_m5()
    assert mvp.RESULTS["M5"] == "PASS"
    text = md.read_text()
    for heading in ("## Run", "## Thresholds", "## RAGTruth test", "## Clinical test: full", "## Clinical test: clean",
                    "## Notes"):
        assert heading in text
    assert "| RAGTruth QA |" in text and "| RAGTruth Summary |" in text
    assert text.count("| F1 if every row flagged |") == 3
    assert "—" not in text                 # no em dashes in reports


def test_report_refuses_smoke_predictions(tmp_path):
    data, nli = _make_toy(tmp_path, limit=100)
    with pytest.raises(ValueError):
        report_main(["--nli-root", str(nli), "--model-dir", "toy-model", "--data-dir", str(data),
                     "--out-md", str(tmp_path / "x.md")])
