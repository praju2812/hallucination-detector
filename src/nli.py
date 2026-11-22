"""Phase 6 (M5): zero-shot NLI baseline (RQ3 comparison point).

A pretrained NLI model judges whether the retrieved context ENTAILS each sentence of the response.
No training. The CLI is scripts/score_nli.py (scoring) and scripts/report_nli.py (metrics + tables).

Method:
    premise     = a window of the context (512-token model input, 128-token overlap between windows)
    sentences   = split_response_sentences, then normalize_preambles (attribution prefix, colon lines)
    hypothesis  = one response sentence (response-only by default)
    support     = P(entailment) for one (window, sentence) pair
    row support = min over sentences of (max over windows of support)
    score       = 1 - row support  (higher = more likely hallucinated, same direction as G's prob)

Spec: tests/test_nli.py.
"""

from __future__ import annotations

import re

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

def merge_short_pieces(pieces: list[str]) -> list[str]:
    """merge pieces with fewer than MIN_WORDS words into a neighbour, never drop them.

    A short piece joins the end of the previous sentence; if there is none yet, it is held and put in
    front of the next piece. If only short pieces exist, they become one sentence.
    Input is not modified. An empty input gives an empty list.
    """
    sentences = []
    held = None
    for piece in pieces:
        if len(piece.split()) < MIN_WORDS:
            if sentences:
                sentences[-1] += ' ' + piece
            else:
                held = (held + ' ' + piece).strip() if held else piece
        else:
            if held:
                piece = held + ' ' + piece
                held = None
            sentences.append(piece)
    # only short pieces existed: together they form the single sentence
    if held:
        sentences.append(held)
    return sentences

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
    # strip the text; raise on empty
    text = text.strip() if text else None
    if not text:
        raise ValueError("response is empty or whitespace only")

    # split into lines, then each line into sentence pieces with the segmenter; strip; drop per rule 2.
    lines = text.split('\n')
    pieces = []
    for line in lines:
        if not line:
            continue
        line_pieces = get_segmenter().segment(line)
        pieces.extend(p.strip() for p in line_pieces if any(ch.isalnum() for ch in p))
    
    # if no piece is left, return the fallback.
    if not pieces:
        return [text], True
    
    # walk the pieces in order and merge the short ones per rule 3 (keep track of a "held" prefix)
    return merge_short_pieces(pieces), False

# Attribution prefix stripped by normalize_preambles. Matched at the start of a sentence,
# case-insensitive; [^,]* stops at the first comma.
ATTRIBUTION_PREFIX = re.compile(r"^based on (the )?(given |provided )?(passages|context|information)[^,]*,\s*",
                                re.IGNORECASE)


def normalize_preambles(sentences: list[str]) -> tuple[list[str], int, int]:
    """Remove generator preambles from the split sentences. Same for every source.

    Applied after split_response_sentences and before build_hypotheses (scripts/score_nli.py prepare_row).
    Input:  sentences from split_response_sentences (stripped, non-empty). Not modified.
    Output: (sentences, n_stripped, n_merged)
        n_stripped: sentences whose ATTRIBUTION_PREFIX was removed.
        n_merged:   colon-line merges performed (step 2). Step 3 merges are not counted.
    Steps, in this order:
        1. Strip: if a sentence matches ATTRIBUTION_PREFIX, keep only the remainder after the match.
           If nothing would remain, leave the sentence unchanged and do not count it.
        2. Colon lines: a sentence ending with ":" is merged (one space) onto the FRONT of the next sentence.
           Consecutive colon lines chain forward into the first sentence that does not end with ":".
           If the LAST sentence ends with ":" (after the forward merges), it is merged onto the end of the
           previous sentence. A lone sentence is left as it is. Never drop.
        3. Short fragments: apply split_response_sentences' MIN_WORDS rule again (merge into previous; a first
           fragment goes in front of the next; a lone short sentence stays), since stripping can leave
           a short remainder such as "yes.".
    Raises ValueError if sentences is empty.
    Invariant: the words of the output, joined, equal the words of the input minus the stripped prefixes;
    with more than one sentence, every sentence has at least MIN_WORDS words and only the last may end with ":".

    Why: the row score is a MIN over sentences. "Based on the passages, X" is judged as X; a line like
    "Here are some ways to ...:" is never entailed on its own, so it is judged together with what it introduces.
    """
    # validate the input
    if not sentences or any(not isinstance(s, str) or not s.strip() for s in sentences):
        raise ValueError("Empty or invalid sentences to normalize")
    sentences = list(sentences)  # work on a copy

    # strip the prefix (ATTRIBUTION_PREFIX.match / .end()), counting n_stripped
    n_stripped = 0
    for i, s in enumerate(sentences):
        m = ATTRIBUTION_PREFIX.match(s)
        if m:
            remainder = s[m.end():].strip()
            if remainder:
                sentences[i] = remainder
                n_stripped += 1
        
    # forward pass: colon lines are carried forward onto the front of the next sentence
    n_merged = 0
    result = []
    pending = None        # colon lines waiting for the sentence they introduce
    pending_count = 0
    for s in sentences:
        if s.endswith(":"):
            pending = s if pending is None else pending + " " + s
            pending_count += 1
        else:
            if pending is not None:
                s = pending + " " + s
                n_merged += pending_count
                pending, pending_count = None, 0
            result.append(s)

    # colon lines left at the end: chained together, then merged onto the end of the previous sentence
    if pending is not None:
        n_merged += pending_count - 1          # the forward chaining among them
        if result:
            result[-1] += " " + pending
            n_merged += 1                      # the backward merge
        else:
            result.append(pending)             # only colon lines: left as one sentence
    sentences = result

    # the MIN_WORDS merge again
    sentences = merge_short_pieces(sentences)

    return sentences, n_stripped, n_merged


def build_hypotheses(row: dict, sentences: list[str], include_question: bool) -> list[tuple[str | None, str]]:
    """One (question, sentence) pair per response sentence; question is None when it is not used.

    Inputs:
        row: a data row (task, query, ...). sentences:output of normalize_preambles
        include_question: the --hypothesis flag.
    Output: [(question or None, sentence), ...], same order and length as sentences.
    Rules:
        - the question (row["query"], stripped) is used only if include_question is True AND row["task"] == "qa".
          Summary rows never get a question.
        - sentences are not changed.
    Raises ValueError if sentences is empty, or if the question is needed but the query is missing or blank.
    Why a pair, not one string: cap_hypothesis_ids must shorten the question before the sentence, so the two
    parts are tokenised separately (scripts/score_nli.py) and joined only after the cap.
    """
    # validate sentences
    if not sentences or not all(isinstance(s, str) and s.strip() for s in sentences):
        raise ValueError("Empty or invalid sentences to build hypotheses from")
    
    # decide whether this row gets the question; if yes, read and validate the query.
    question = None
    if include_question and row.get("task") == "qa":
        question = (row.get("query") or "").strip()
        if not question:
            raise ValueError("QA row needs a non-empty query for the hypothesis")
        
    # build the list of pairs
    return [(question, sentence) for sentence in sentences]


def cap_hypothesis_ids(question_ids: list[int], sentence_ids: list[int],
                       max_tokens: int = HYP_CAP) -> tuple[list[int], bool]:
    """Join question and sentence token ids into one hypothesis of at most max_tokens ids.

    Inputs: token ids WITHOUT special tokens; question_ids is [] for response-only hypotheses.
    Output: (hypothesis ids = question part + sentence part, capped) with capped True iff anything was removed.
    Rules:
        - fits (len(question) + len(sentence) <= max_tokens): return everything, capped False.
        - otherwise, shorten the QUESTION first, keeping the sentence whole: keep the question's first ids
          (cut from its end) so that the total is max_tokens. If no room is left, the question is dropped.
        - only if the sentence ALONE exceeds max_tokens: drop the question and keep the sentence's first
          max_tokens ids (cut from its end).
        - inputs are not modified.
    Why: with max_tokens 256 the context budget is at least 512 - 256 - 3 = 253 tokens, always above the
    128-token stride, so the window logic never breaks. Only the hypothesis is ever truncated.
    """
    # the no-cap case.
    if len(question_ids) + len(sentence_ids) <= max_tokens:
        return question_ids + sentence_ids, False
    
    # the sentence fits on its own: how much room is left for the question?
    room_for_question = max_tokens - len(sentence_ids)
    if room_for_question > 0:
        return question_ids[:room_for_question] + sentence_ids, True
    
    # the sentence alone is too long.
    return sentence_ids[:max_tokens], True

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
    # validate the arguments.
    if n_context < 1:
        raise ValueError(f"n_context must be >= 1, got {n_context}")
    if stride < 0:
        raise ValueError(f"stride must be >= 0, got {stride}")
    if budget <= stride:
        raise ValueError(f"budget must be > stride, got budget={budget}, stride={stride}")
    
    # the single-window case
    if n_context <= budget:
        return [(0, n_context)]
    
    # step through the context; work out how far each new window starts after the previous one.
    windows = []
    start = 0
    while start < n_context:
        end = min(start + budget, n_context)
        windows.append((start, end))
        if end == n_context:
            break
        start += budget - stride

    return windows

# ---- model outputs -> scores ----------------------------------------------------------------

def entailment_index(id2label: dict) -> int:
    """Index of the entailment class, read from model.config.id2label (never hard-coded).

    Input: {index: label}; keys may be int or str ("0"), labels in any case.
    Output: the int index whose label, lower-cased, is exactly "entailment" ("not_entailment" does not match).
    Raises ValueError if no label or more than one label matches.
    """
    # find the matching keys
    matches = [int(k) for k, v in id2label.items() if str(v).lower() == "entailment"]
    if len(matches) == 0:
        raise ValueError("No entailment label found")
    if len(matches) > 1:
        raise ValueError("Multiple entailment labels found")
    return matches[0]


def entailment_probs(logits: np.ndarray, ent_idx: int) -> np.ndarray:
    """P(entailment) per pair from the model's logits.

    Input: logits, shape (n_pairs, n_labels), a float array. ent_idx from entailment_index.
    Output: shape (n_pairs,) = softmax over the label axis, entailment column.
    Raises ValueError if logits is not 2-D.
    """

    # validate the shape.
    if len(logits.shape) != 2:
        raise ValueError(f"Expected 2-D logits, got {logits.shape}")
    
    # validate ent_idx is in range [0, n_labels).
    if ent_idx < 0 or ent_idx >= logits.shape[1]:
        raise ValueError(f"ent_idx {ent_idx} is out of range [0, {logits.shape[1]})")

    #a numerically stable softmax over the label axis.
    max_logits = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return softmax[:, ent_idx]

def aggregate_support(support_by_sentence: list[list[float]]) -> float:
    """Row support from pair supports: max over windows for each sentence, then min over sentences.

    Input: one inner list per sentence, holding P(entailment) for each of its windows.
    Output: a float in [0, 1].
    Why: a sentence is supported if ANY window supports it (max); a response is only as faithful as its LEAST
    supported sentence (min).
    Raises ValueError if there are no sentences, or a sentence has no windows.
    """
    # validate.
    if not support_by_sentence or any(len(s) == 0 for s in support_by_sentence):
        raise ValueError("Empty sentences provided for support aggregation")
    
    # max per sentence, then min over sentences; return a plain float.
    max_per_sentence = [max(supports) for supports in support_by_sentence]
    return float(min(max_per_sentence))

def hallucination_score(support: float) -> float:
    """Score in the same direction as G's prob: 1 - support (1 = no sentence supported at all).

    Returns a plain Python float. Raises ValueError if support is outside [0, 1].
    """
    # validate.
    if support < 0 or support > 1 or np.isnan(support) or np.isinf(support):
        raise ValueError("Support must be in [0, 1]")
    # return the score as a plain float.
    return float(1 - support)
