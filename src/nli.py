"""Phase 6 (M5): zero-shot NLI baseline (RQ3 comparison point).

A pretrained NLI model judges whether the retrieved context ENTAILS each sentence of the response.
No training. The CLI is scripts/score_nli.py (scoring) and scripts/report_nli.py (metrics + tables).

Method (decided, AE-1 §3.4 and Phase 6):
    premise     = a window of the context (512-token model input, 128-token overlap between windows)
    hypothesis  = one response sentence (optionally preceded by the question for QA rows: open decision)
    support     = P(entailment) for one (window, sentence) pair
    row support = min over sentences of (max over windows of support)
    score       = 1 - row support  (higher = more likely hallucinated, same direction as G's prob)

Spec: tests/test_nli.py.
"""

from __future__ import annotations

import numpy as np

MODEL_ID = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
MAX_LENGTH = 512          # model input: [CLS] premise [SEP] hypothesis [SEP]
STRIDE = 128              # tokens shared by consecutive context windows (decision, 19 Nov)
HYP_CAP = 256             # max hypothesis tokens, so the context budget stays above STRIDE (decision, 19 Nov)
MIN_WORDS = 3             # pieces with fewer words are merged into a neighbour, never dropped (decision, 19 Nov)

_SEGMENTER = None


def get_segmenter():
    """The pysbd sentence segmenter (English, clean=False so the text is not altered). Built once."""
    global _SEGMENTER
    if _SEGMENTER is None:
        import pysbd
        _SEGMENTER = pysbd.Segmenter(language="en", clean=False)
    return _SEGMENTER


# ---- response -> hypotheses --------------------------------------------------------------

def split_response_sentences(text: str) -> tuple[list[str], bool]:
    """Split a response into sentences that each become one NLI hypothesis.

    Input:  the response (row["answer"]).
    Output: (sentences, fell_back)
        sentences: non-empty list of stripped strings, in response order.
        fell_back: True only when splitting left nothing (see rule 4).
    Rules:
        1. Split on newlines FIRST (RAGTruth QA answers are often bullet lists without full stops), then
           split each line into sentences with get_segmenter().segment(line).
        2. Strip every piece. Drop empty pieces and pieces with no letter or digit (bullet-only: "*", "-", "---").
           A bullet marker at the start of a real line is kept ("* Diarrhea ..." stays as it is).
        3. A piece with fewer than MIN_WORDS words (str.split()) is MERGED, never dropped: joined with one space
           onto the end of the previous sentence; if no sentence exists yet, it is held and put in front of the
           next piece. If only short pieces exist, they are joined into one sentence.
        4. If nothing remains after rule 2, return ([text.strip()], True).
    Raises ValueError if the response is empty or whitespace only (no data row is).
    Invariant: when there is more than one sentence, every sentence has at least MIN_WORDS words; the words of the
    output, joined, equal the words of the kept pieces (nothing is lost by merging).

    Why merge: the row score is a MIN over sentences, so a fragment like "Yes." or "Results:", which no context
    entails on its own, would become the row's score.
    """
    # TODO 1: strip the text; raise on empty.
    # TODO 2: split into lines, then each line into sentence pieces with the segmenter; strip; drop per rule 2.
    # TODO 3: if no piece is left, return the fallback.
    # TODO 4: walk the pieces in order and merge the short ones per rule 3 (keep track of a "held" prefix).
    # CHECK: "Yes. The drug works well." -> one sentence; "Long sentence here. OK. Another long one here." -> 2.
    # CHECK: a line "Results:" between two long lines joins the PREVIOUS sentence, not the next.
    # CHECK: does the segmenter return pieces with trailing spaces? (strip them)
    raise NotImplementedError


def build_hypotheses(row: dict, sentences: list[str], include_question: bool) -> list[tuple[str | None, str]]:
    """One (question, sentence) pair per response sentence; question is None when it is not used.

    Inputs:
        row: a data row (task, query, ...). sentences: output of split_response_sentences.
        include_question: the --hypothesis flag (open decision, Prajakta, after the inspection step).
    Output: [(question or None, sentence), ...], same order and length as sentences.
    Rules:
        - the question (row["query"], stripped) is used only if include_question is True AND row["task"] == "qa".
          Summary rows never get a question.
        - sentences are not changed.
    Raises ValueError if sentences is empty, or if the question is needed but the query is missing or blank.
    Why a pair, not one string: cap_hypothesis_ids must shorten the question before the sentence, so the two
    parts are tokenised separately (scripts/score_nli.py) and joined only after the cap.
    """
    # TODO 1: validate sentences.
    # TODO 2: decide whether this row gets the question; if yes, read and validate the query.
    # TODO 3: build the list of pairs.
    # CHECK: the task value in the data is lower case ("qa", "summary").
    raise NotImplementedError


def cap_hypothesis_ids(question_ids: list[int], sentence_ids: list[int],
                       max_tokens: int = HYP_CAP) -> tuple[list[int], bool]:
    """Join question and sentence token ids into one hypothesis of at most max_tokens ids.

    Inputs: token ids WITHOUT special tokens; question_ids is [] for response-only hypotheses.
    Output: (hypothesis ids = question part + sentence part, capped) with capped True iff anything was removed.
    Rules (decision, 19 Nov):
        - fits (len(question) + len(sentence) <= max_tokens): return everything, capped False.
        - otherwise, shorten the QUESTION first, keeping the sentence whole: keep the question's first ids
          (cut from its end) so that the total is max_tokens. If no room is left, the question is dropped.
        - only if the sentence ALONE exceeds max_tokens: drop the question and keep the sentence's first
          max_tokens ids (cut from its end).
        - inputs are not modified.
    Why: with max_tokens 256 the context budget is at least 512 - 256 - 3 = 253 tokens, always above the
    128-token stride, so the window logic never breaks. Only the hypothesis is ever truncated.
    """
    # TODO 1: the no-cap case.
    # TODO 2: the sentence fits on its own: how much room is left for the question?
    # TODO 3: the sentence alone is too long.
    # CHECK: exactly max_tokens in total is NOT capped.
    # CHECK: return new lists (slicing), never the caller's lists modified in place.
    raise NotImplementedError


# ---- windows over the context -------------------------------------------------------------

def window_spans(n_context: int, budget: int, stride: int = STRIDE) -> list[tuple[int, int]]:
    """Token spans [start, end) of the context windows used as premises.

    Inputs:
        n_context: number of context tokens (no special tokens).
        budget: max context tokens per window = MAX_LENGTH - len(hypothesis ids) - special tokens.
        stride: tokens shared by consecutive windows (the tokenizer's "stride" meaning: overlap, not step).
    Output: list of (start, end) with
        - first start 0, last end n_context (every token is in some window);
        - every window has length budget except possibly the last, which may be shorter;
        - consecutive windows overlap by exactly `stride` tokens.
        A context that fits gives one window [(0, n_context)].
    These are the same windows as a fast tokenizer's return_overflowing_tokens=True with stride=stride,
    written out so they can be tested without a model.
    Raises ValueError if n_context < 1, stride < 0, or budget <= stride (no progress would be made).
    """
    # TODO 1: validate the arguments.
    # TODO 2: the single-window case.
    # TODO 3: step through the context; work out how far each new window starts after the previous one.
    # CHECK: (10, 4, 1) -> [(0, 4), (3, 7), (6, 10)]; (11, 4, 1) adds (9, 11).
    # CHECK: the loop stops as soon as a window reaches n_context (no extra empty or duplicate window).
    raise NotImplementedError


# ---- model outputs -> scores ----------------------------------------------------------------

def entailment_index(id2label: dict) -> int:
    """Index of the entailment class, read from model.config.id2label (never hard-coded).

    Input: {index: label}; keys may be int or str ("0"), labels in any case.
    Output: the int index whose label, lower-cased, is exactly "entailment" ("not_entailment" does not match).
    Raises ValueError if no label or more than one label matches.
    """
    # TODO 1: find the matching keys.
    # TODO 2: validate that there is exactly one; return it as an int.
    raise NotImplementedError


def entailment_probs(logits: np.ndarray, ent_idx: int) -> np.ndarray:
    """P(entailment) per pair from the model's logits.

    Input: logits, shape (n_pairs, n_labels), a float array. ent_idx from entailment_index.
    Output: shape (n_pairs,) = softmax over the label axis, entailment column.
    Raises ValueError if logits is not 2-D.
    """
    # TODO 1: validate the shape.
    # TODO 2: a numerically stable softmax over the label axis.
    # TODO 3: take the entailment column.
    # CHECK: logits like 1000 must not give inf or nan (what do you subtract before exponentiating?).
    raise NotImplementedError


def aggregate_support(support_by_sentence: list[list[float]]) -> float:
    """Row support from pair supports: max over windows for each sentence, then min over sentences.

    Input: one inner list per sentence, holding P(entailment) for each of its windows.
    Output: a float in [0, 1].
    Why: a sentence is supported if ANY window supports it (max); a response is only as faithful as its LEAST
    supported sentence (min).
    Raises ValueError if there are no sentences, or a sentence has no windows.
    """
    # TODO 1: validate.
    # TODO 2: max per sentence, then min over sentences; return a plain float.
    raise NotImplementedError


def hallucination_score(support: float) -> float:
    """Score in the same direction as G's prob: 1 - support (1 = no sentence supported at all).

    Returns a plain Python float. Raises ValueError if support is outside [0, 1].
    """
    # TODO 1: validate; return the score as a plain float.
    raise NotImplementedError
