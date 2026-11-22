"""Phase 6 (M5): spec for the src/nli.py templates (zero-shot NLI baseline).

Toy inputs only: no model or tokenizer download. Token ids are plain int lists.
Run: python -m pytest tests/test_nli.py
"""

import numpy as np
import pytest

from src.nli import (
    aggregate_support,
    build_hypotheses,
    cap_hypothesis_ids,
    entailment_index,
    entailment_probs,
    hallucination_score,
    normalize_preambles,
    split_response_sentences,
    window_spans,
    ATTRIBUTION_PREFIX,
)


# ---- split_response_sentences ----------------------------------------------------------

def test_split_two_sentences():
    s, fell_back = split_response_sentences("The drug reduced mortality in adults. Side effects were mild overall.")
    assert s == ["The drug reduced mortality in adults.", "Side effects were mild overall."]
    assert fell_back is False


def test_split_keeps_biomedical_abbreviations_and_decimals():
    text = ("Smith et al. reported a 2.5-fold rise in risk. The effect was significant (p < 0.05). "
            "Common causes include viruses, e.g. influenza.")
    s, _ = split_response_sentences(text)
    assert s == ["Smith et al. reported a 2.5-fold rise in risk.",
                 "The effect was significant (p < 0.05).",
                 "Common causes include viruses, e.g. influenza."]


def test_split_on_newlines_first():
    # RAGTruth QA answers are often bullet lists with no full stops: each line is its own sentence.
    # Bullet markers at the start of a line are kept (only bullet-ONLY pieces are dropped).
    text = "Symptoms include the following:\n* Diarrhea that is often violent\n* Abdominal pain after eating"
    s, _ = split_response_sentences(text)
    assert s == ["Symptoms include the following:", "* Diarrhea that is often violent", "* Abdominal pain after eating"]


def test_split_drops_blank_and_bullet_only_pieces():
    text = "First point is clearly stated here.\n\n*\n  - \nSecond point is also stated here."
    s, fell_back = split_response_sentences(text)
    assert s == ["First point is clearly stated here.", "Second point is also stated here."]
    assert fell_back is False


def test_short_fragment_first_merges_into_next():
    s, _ = split_response_sentences("Yes. The drug reduced mortality in adults.")
    assert s == ["Yes. The drug reduced mortality in adults."]


def test_short_fragment_middle_merges_into_previous():
    s, _ = split_response_sentences("The trial enrolled 300 adults. It worked. Side effects were mild overall.")
    assert s == ["The trial enrolled 300 adults. It worked.", "Side effects were mild overall."]


def test_short_fragment_last_merges_into_previous():
    s, _ = split_response_sentences("The trial enrolled 300 adults in total.\nResults:")
    assert s == ["The trial enrolled 300 adults in total. Results:"]


def test_consecutive_short_fragments_at_start_merge_into_first_long_sentence():
    s, _ = split_response_sentences("Yes.\nNo.\nThe answer depends on the dose given.")
    assert s == ["Yes. No. The answer depends on the dose given."]


def test_three_words_is_not_a_fragment():
    s, _ = split_response_sentences("The trial enrolled 300 adults. It worked well. Side effects were mild overall.")
    assert s == ["The trial enrolled 300 adults.", "It worked well.", "Side effects were mild overall."]


def test_lone_short_sentence_is_kept():
    assert split_response_sentences("No.") == (["No."], False)
    assert split_response_sentences("  It works.  ") == (["It works."], False)


def test_only_short_fragments_join_into_one_sentence():
    assert split_response_sentences("Yes.\nNo.") == (["Yes. No."], False)


def test_merging_never_drops_words():
    texts = ["Yes. The drug reduced mortality in adults. It worked. Side effects were mild.",
             "Overview:\nThe trial enrolled 300 adults.\nOK.\nNo serious adverse events were seen.",
             "A. B. C. The final sentence is long enough here."]
    for t in texts:
        s, _ = split_response_sentences(t)
        assert " ".join(s).split() == t.split()


def test_multi_sentence_output_has_no_fragments():
    texts = ["Yes. The drug reduced mortality in adults. It worked. Side effects were mild.",
             "Overview:\nThe trial enrolled 300 adults.\nOK.\nNo serious adverse events were seen."]
    for t in texts:
        s, _ = split_response_sentences(t)
        assert len(s) > 1 and all(len(x.split()) >= 3 for x in s)


def test_nothing_remains_falls_back_to_whole_response():
    assert split_response_sentences("---\n***") == (["---\n***"], True)
    assert split_response_sentences("  * \n - ") == (["* \n -"], True)       # the STRIPPED response


def test_empty_response_raises():
    for t in ["", "   \n  "]:
        with pytest.raises(ValueError):
            split_response_sentences(t)


# ---- normalize_preambles ----------------------------------------------------------------

def test_preamble_prefix_stripped_claim_kept():
    s = ["Based on the given passages, the drug reduced mortality in adults.", "Side effects were mild overall."]
    assert normalize_preambles(s) == (["the drug reduced mortality in adults.", "Side effects were mild overall."], 1, 0)


def test_preamble_prefix_variants_case_insensitive_up_to_first_comma():
    assert normalize_preambles(["BASED ON CONTEXT given above, aspirin lowers the risk."]) == \
        (["aspirin lowers the risk."], 1, 0)
    assert normalize_preambles(["Based on the information provided in the passages, aspirin lowers the risk."]) == \
        (["aspirin lowers the risk."], 1, 0)
    assert normalize_preambles(["based on provided passages,   aspirin lowers the risk, mostly."]) == \
        (["aspirin lowers the risk, mostly."], 1, 0)                 # [^,]* stops at the FIRST comma


def test_preamble_prefix_not_matched_is_left_alone():
    s = ["Based on the context the drug works well.",                # no comma
         "The answer, based on the context, is yes overall.",        # not at the start
         "Based on these passages, the drug works well."]            # "these" is not in the pattern
    assert normalize_preambles(s) == (s, 0, 0)


def test_preamble_prefix_only_sentence_is_kept_unchanged():
    # Nothing would remain after stripping (none in validation, 21 Nov): the sentence stays as it is, not counted.
    s = ["Based on the given passages,", "The drug reduced mortality in adults."]
    assert normalize_preambles(s) == (s, 0, 0)


def test_preamble_colon_line_merged_forward():
    s = ["Here are the main symptoms:", "* Diarrhea that is often violent", "* Abdominal pain after eating"]
    assert normalize_preambles(s) == \
        (["Here are the main symptoms: * Diarrhea that is often violent", "* Abdominal pain after eating"], 0, 1)


def test_preamble_colon_line_last_merges_backward():
    s = ["The trial enrolled 300 adults in total.", "The main results were as follows:"]
    assert normalize_preambles(s) == (["The trial enrolled 300 adults in total. The main results were as follows:"], 0, 1)


def test_preamble_remainder_ending_with_colon_is_merged():
    s = ["Based on the given passages, here is the answer:", "* The drug reduced mortality in adults."]
    assert normalize_preambles(s) == (["here is the answer: * The drug reduced mortality in adults."], 1, 1)


def test_preamble_consecutive_colon_lines_chain_forward():
    s = ["The findings are listed below:", "First the main outcome measures:", "* Mortality fell by a third overall."]
    assert normalize_preambles(s) == \
        (["The findings are listed below: First the main outcome measures: * Mortality fell by a third overall."], 0, 2)


def test_preamble_colon_chain_at_the_end_merges_backward():
    s = ["The trial enrolled 300 adults.", "Results were as follows:", "Details are listed below:"]
    assert normalize_preambles(s) == \
        (["The trial enrolled 300 adults. Results were as follows: Details are listed below:"], 0, 2)


def test_preamble_lone_colon_sentence_kept():
    assert normalize_preambles(["The results are as follows:"]) == (["The results are as follows:"], 0, 0)


def test_preamble_colon_inside_a_sentence_is_not_a_colon_line():
    s = ["Note: the drug works well here.", "Side effects were mild overall."]
    assert normalize_preambles(s) == (s, 0, 0)


def test_preamble_short_remainder_merged_again():
    s = ["The drug reduced mortality in adults.", "Based on the context, yes."]
    assert normalize_preambles(s) == (["The drug reduced mortality in adults. yes."], 1, 0)       # into previous
    s = ["Based on the passages, yes.", "The drug reduced mortality in adults."]
    assert normalize_preambles(s) == (["yes. The drug reduced mortality in adults."], 1, 0)       # first: into next
    assert normalize_preambles(["Based on the context, yes."]) == (["yes."], 1, 0)               # lone: kept


def test_preamble_nothing_is_ever_lost():
    cases = [
        ["Based on the provided passages, here are some ways to get rid of a smell:", "* Open the windows wide",
         "* Bake soda in a bowl overnight", "These steps usually help a lot:"],
        ["Based on the context provided, yes.", "Summary of the findings:", "Mortality fell by a third."],
        ["The trial enrolled 300 adults.", "It worked well.", "Side effects were mild overall."],
    ]
    for s in cases:
        out, n_stripped, _ = normalize_preambles(list(s))
        expected = " ".join(ATTRIBUTION_PREFIX.sub("", x, count=1) for x in s).split()
        assert " ".join(out).split() == expected                    # every word kept, in order, minus the prefixes
        assert n_stripped == sum(bool(ATTRIBUTION_PREFIX.match(x)) for x in s)
        assert all(x == x.strip() and x for x in out)
        if len(out) > 1:
            assert all(len(x.split()) >= 3 for x in out)             # MIN_WORDS invariant holds again
            assert not any(x.endswith(":") for x in out[:-1])        # only the last may end with ":" (merged back)


def test_preamble_does_not_modify_input_and_rejects_empty():
    s = ["Based on the context, the drug works well.", "Results were as follows:", "Mortality fell by a third."]
    before = list(s)
    normalize_preambles(s)
    assert s == before
    with pytest.raises(ValueError):
        normalize_preambles([])


# ---- cap_hypothesis_ids ----------------------------------------------------------------

def test_cap_not_needed():
    q, s = list(range(100, 110)), list(range(10))
    assert cap_hypothesis_ids(q, s, max_tokens=256) == (q + s, False)
    assert cap_hypothesis_ids([], s, max_tokens=256) == (s, False)


def test_cap_exactly_at_limit_is_not_capped():
    q, s = [7] * 6, [8] * 4
    assert cap_hypothesis_ids(q, s, max_tokens=10) == (q + s, False)


def test_cap_truncates_question_from_its_end_keeps_sentence_whole():
    q, s = list(range(1000, 1010)), list(range(6))
    out, capped = cap_hypothesis_ids(q, s, max_tokens=10)
    assert out == q[:4] + s and capped is True


def test_cap_drops_question_when_sentence_fills_the_cap():
    q, s = [9] * 5, list(range(10))
    assert cap_hypothesis_ids(q, s, max_tokens=10) == (s, True)


def test_cap_sentence_alone_over_cap_truncated_from_end_question_dropped():
    q, s = [9] * 5, list(range(15))
    assert cap_hypothesis_ids(q, s, max_tokens=10) == (list(range(10)), True)


def test_cap_response_only_sentence_over_cap():
    assert cap_hypothesis_ids([], list(range(300)), max_tokens=256) == (list(range(256)), True)


def test_cap_default_256_keeps_context_budget_above_stride():
    out, capped = cap_hypothesis_ids(list(range(50)), list(range(400)))
    assert len(out) == 256 and capped
    assert 512 - len(out) - 3 > 128                    # context budget never falls to the stride


def test_cap_does_not_modify_inputs():
    q, s = list(range(20)), list(range(20))
    cap_hypothesis_ids(q, s, max_tokens=10)
    assert q == list(range(20)) and s == list(range(20))


# ---- entailment_index --------------------------------------------------------------------

def test_entailment_index_read_from_id2label():
    assert entailment_index({0: "entailment", 1: "neutral", 2: "contradiction"}) == 0
    idx = entailment_index({"0": "CONTRADICTION", "1": "NEUTRAL", "2": "ENTAILMENT"})   # str keys, upper case
    assert idx == 2 and isinstance(idx, int)


def test_entailment_index_exact_label_not_substring():
    assert entailment_index({0: "not_entailment", 1: "entailment"}) == 1


def test_entailment_index_missing_or_duplicate_raises():
    with pytest.raises(ValueError):
        entailment_index({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})
    with pytest.raises(ValueError):
        entailment_index({0: "entailment", 1: "Entailment", 2: "neutral"})


# ---- build_hypotheses ----------------------------------------------------------------------

QA = {"task": "qa", "query": "  What causes IBS?  ", "answer": "x", "context": "c"}
SUMMARY = {"task": "summary", "query": "Summarize the following news.", "answer": "x", "context": "c"}


def test_hypotheses_response_only():
    assert build_hypotheses(QA, ["The cause is unknown here."], include_question=False) == [(None, "The cause is unknown here.")]


def test_hypotheses_with_question_for_qa_rows():
    out = build_hypotheses(QA, ["First claim is here.", "Second claim is here."], include_question=True)
    assert out == [("What causes IBS?", "First claim is here."), ("What causes IBS?", "Second claim is here.")]


def test_hypotheses_summary_rows_never_get_a_question():
    assert build_hypotheses(SUMMARY, ["A summary sentence here."], include_question=True) == [(None, "A summary sentence here.")]


def test_hypotheses_missing_query_raises_only_when_needed():
    row = {"task": "qa", "query": "   ", "answer": "x", "context": "c"}
    with pytest.raises(ValueError):
        build_hypotheses(row, ["A claim is here."], include_question=True)
    assert build_hypotheses(row, ["A claim is here."], include_question=False) == [(None, "A claim is here.")]


def test_hypotheses_no_sentences_raises():
    with pytest.raises(ValueError):
        build_hypotheses(QA, [], include_question=False)


# ---- window_spans ------------------------------------------------------------------------

def test_window_single_when_context_fits():
    assert window_spans(100, 300, 128) == [(0, 100)]
    assert window_spans(300, 300, 128) == [(0, 300)]


def test_windows_overlap_by_stride():
    assert window_spans(10, 4, 1) == [(0, 4), (3, 7), (6, 10)]
    assert window_spans(11, 4, 1) == [(0, 4), (3, 7), (6, 10), (9, 11)]


def test_windows_zero_stride_are_contiguous():
    assert window_spans(10, 4, 0) == [(0, 4), (4, 8), (8, 10)]


def test_windows_cover_every_token_property():
    for n in range(1, 2300, 37):
        for budget in (253, 381, 509):
            spans = window_spans(n, budget, 128)
            assert spans[0][0] == 0 and spans[-1][1] == n
            assert all(0 < e - s <= budget for s, e in spans)
            assert all(e - s == budget for s, e in spans[:-1])                 # only the last may be shorter
            assert all(a[1] - b[0] == 128 for a, b in zip(spans, spans[1:]))   # overlap = stride exactly


def test_windows_invalid_arguments_raise():
    with pytest.raises(ValueError):
        window_spans(1000, 128, 128)          # budget must exceed the stride
    with pytest.raises(ValueError):
        window_spans(0, 300, 128)             # empty context
    with pytest.raises(ValueError):
        window_spans(1000, 300, -1)


# ---- entailment_probs ----------------------------------------------------------------------

def test_entailment_probs_softmax_column():
    logits = np.array([[2.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    e = np.exp([2.0, 1.0, 0.0])
    np.testing.assert_allclose(entailment_probs(logits, 0), [e[0] / e.sum(), 1 / 3])
    np.testing.assert_allclose(entailment_probs(logits, 2), [e[2] / e.sum(), 1 / 3])


def test_entailment_probs_stable_for_large_logits():
    p = entailment_probs(np.array([[1000.0, 0.0, -1000.0], [-1000.0, 0.0, 1000.0]]), 0)
    assert np.all(np.isfinite(p))
    np.testing.assert_allclose(p, [1.0, 0.0], atol=1e-12)


def test_entailment_probs_shape_and_bad_input():
    p = entailment_probs(np.zeros((5, 3)), 1)
    assert p.shape == (5,)
    with pytest.raises(ValueError):
        entailment_probs(np.zeros(3), 0)       # must be 2-D: (pairs, labels)


# ---- aggregate_support and hallucination_score ------------------------------------------------

def test_aggregate_max_over_windows_then_min_over_sentences():
    support = [[0.2, 0.9], [0.7], [0.4, 0.6, 0.1]]       # per-sentence maxima 0.9, 0.7, 0.6
    assert aggregate_support(support) == pytest.approx(0.6)


def test_aggregate_single_pair_and_window_order():
    assert aggregate_support([[0.3]]) == pytest.approx(0.3)
    assert aggregate_support([[0.1, 0.8]]) == aggregate_support([[0.8, 0.1]])


def test_aggregate_empty_raises():
    with pytest.raises(ValueError):
        aggregate_support([])
    with pytest.raises(ValueError):
        aggregate_support([[0.5], []])


def test_hallucination_score_is_one_minus_support():
    assert hallucination_score(0.6) == pytest.approx(0.4)
    assert hallucination_score(0.0) == 1.0 and hallucination_score(1.0) == 0.0
    assert isinstance(hallucination_score(np.float32(0.25)), float)
    for bad in (1.2, -0.1):
        with pytest.raises(ValueError):
            hallucination_score(bad)
