"""Spec for Collator.encode_with_response_budget (query field).

Output layout:  [CLS] first_segment [SEP] response_segment [SEP]
where first_segment comes from INPUT_FORMAT ("Question: {query} Context: {context}").

ONE reference definition is used throughout (because tokenizing pieces separately
differs from tokenizing the joined string at the boundaries, BPE):

    ref_first = tokenizer(INPUT_FORMAT.format(query=q, context=c),
                          add_special_tokens=False)["input_ids"]
    ref_resp  = tokenizer(response, add_special_tokens=False)["input_ids"]

- Short inputs: output == tokenizer(INPUT_FORMAT.format(...), response)["input_ids"] exactly.
- Long inputs: kept first segment is a PREFIX of ref_first; kept response is a PREFIX of ref_resp.

Run: python -m pytest tests/test_encoding.py
"""

import pytest
import torch
import yaml

from src.encoding import INPUT_FORMAT, Collator, build_tokenizer

with open("configs/base.yaml") as f:
    BASE = yaml.safe_load(f)
MODEL = BASE["model"]["name"]
MAX_LEN = BASE["model"]["max_length"]   # 1024
RESP_MAX = BASE["model"]["resp_max"]    # 384


@pytest.fixture(scope="module")
def tok():
    return build_tokenizer(MODEL)


@pytest.fixture(scope="module")
def coll(tok):
    return Collator(tok, MAX_LEN, RESP_MAX, "legacy")   # the tests below up to fit_first specify legacy


# ------------------------------- helpers -------------------------------
def ids(tok, text):
    return tok(text, add_special_tokens=False)["input_ids"]


def ref_first(tok, query, context):
    """Reference first segment: the joined INPUT_FORMAT string tokenized in one go."""
    return ids(tok, INPUT_FORMAT.format(query=query, context=context))


def split_output(tok, out):
    """[CLS] first [SEP] resp [SEP] -> (first, resp). Fails if the layout is wrong."""
    assert out[0] == tok.cls_token_id, "output must start with [CLS]"
    assert out[-1] == tok.sep_token_id, "output must end with [SEP]"
    seps = [i for i, t in enumerate(out) if t == tok.sep_token_id]
    assert len(seps) == 2, f"expected exactly 2 [SEP] tokens, got {len(seps)}"
    return out[1:seps[0]], out[seps[0] + 1:-1]


def is_prefix(short, long):
    return long[:len(short)] == short


def words(word, n):
    return " ".join(f"{word}{i}" for i in range(n))


QUERY = "When did Anne Frank die?"
QUERY_HEAD = f"Question: {QUERY} Context:"


# ------ module surface --------
def test_input_format_constant():
    """The format string is fixed by PHASES.md and recorded in config_snapshot.json."""
    assert INPUT_FORMAT == "Question: {query} Context: {context}"


def test_signature_takes_query():
    """mvp_check.py M0 looks for a query argument on encode_with_response_budget."""
    import inspect
    params = list(inspect.signature(Collator.encode_with_response_budget).parameters)
    assert params == ["self", "query", "context", "response"]


# short inputs: exact parity
@pytest.mark.parametrize("query,context,response", [
    ("who sat on the mat?", "the cat sat on the mat", "a dog barked"),
    ("butcher shop phone number", "passage 1:Butcher Shop - Hayward 826 B Street\n\npassage 2:Lakefield",
     "The number is (510) 889-8690."),
    ("Summarize the following news", "Water boils at 100 degrees Celsius at sea level.",
     "Water boils at 50 degrees."),
    ("Qu'est-ce que c'est ?", "Café, naïve, 25°C, 3.14", "It's 25°C."),
])
def test_short_input_parity_with_tokenizer_pair_encoding(tok, coll, query, context, response):
    """Reference: tokenizer(INPUT_FORMAT.format(query=q, context=c), response)["input_ids"],
    i.e. the tokenizer's own pair encoding with special tokens. Nothing is truncated,
    so the Collator must produce exactly the same ids."""
    expected = tok(INPUT_FORMAT.format(query=query, context=context), response)["input_ids"]
    assert coll.encode_with_response_budget(query, context, response) == expected

# long inputs: query head never truncated
def test_query_intact_on_long_context(tok, coll):
    """Reference: ref_first = tokenizer(INPUT_FORMAT.format(q, c), add_special_tokens=False).
    A 5,000-word context must be cut, the query never: the kept first segment is a prefix
    of ref_first, and decoding it still starts with 'Question: <query> Context:'."""
    context = words("ctx", 5000)
    out = coll.encode_with_response_budget(QUERY, context, "Anne Frank died in 1945.")
    first, _ = split_output(tok, out)
    assert len(out) == MAX_LEN
    assert is_prefix(first, ref_first(tok, QUERY, context))
    assert tok.decode(first).startswith(QUERY_HEAD)


def test_long_query_never_truncated(tok, coll):
    """Reference: ref_first as above. A 150-word query with a 5,000-word context:
    every query word survives (decoded first segment starts with the full query head)."""
    query = words("q", 150)
    context = words("ctx", 5000)
    out = coll.encode_with_response_budget(query, context, "short answer")
    first, _ = split_output(tok, out)
    assert is_prefix(first, ref_first(tok, query, context))
    assert tok.decode(first).startswith(f"Question: {query} Context:")


def test_long_response_protected_to_resp_max(tok, coll):
    """Reference: ref_resp = tokenizer(response, add_special_tokens=False).
    Long context + long response: the response keeps exactly its first RESP_MAX tokens,
    the context absorbs the rest of the window, total length == MAX_LEN."""
    context = words("ctx", 5000)
    response = words("resp", 1000)
    out = coll.encode_with_response_budget(QUERY, context, response)
    first, resp = split_output(tok, out)
    assert resp == ids(tok, response)[:RESP_MAX]
    assert len(out) == MAX_LEN
    assert is_prefix(first, ref_first(tok, QUERY, context))


def test_long_response_short_context_keeps_whole_context(tok, coll):
    """Reference: ref_first and ref_resp as above. Context fits, so the first segment is
    ref_first in full; the response is still capped at RESP_MAX."""
    context = "the cat sat on the mat"
    response = words("resp", 1000)
    out = coll.encode_with_response_budget(QUERY, context, response)
    first, resp = split_output(tok, out)
    assert first == ref_first(tok, QUERY, context)
    assert resp == ids(tok, response)[:RESP_MAX]


def test_short_response_kept_whole_and_context_fills_rest(tok, coll):
    """Reference: ref_first and ref_resp as above. A short response is kept whole;
    only the context is cut, and the window is filled to MAX_LEN."""
    context = words("ctx", 5000)
    response = "Anne Frank died in February 1945 at Bergen-Belsen."
    out = coll.encode_with_response_budget(QUERY, context, response)
    first, resp = split_output(tok, out)
    assert resp == ids(tok, response)
    assert len(out) == MAX_LEN
    n_special = tok.num_special_tokens_to_add(pair=True)
    assert len(first) == MAX_LEN - n_special - len(resp)
    assert is_prefix(first, ref_first(tok, QUERY, context))


# Overflow tests: small window, query head + long response cannot both fit
def test_overflow_response_gives_way_context_gets_zero(tok):
    """Small window (64, resp_max 40). Query head ~30 tokens + 200-token response cannot both
    fit at resp_max. Priority query > response > context: query intact, context gets 0 tokens
    (no 'zebra' in the first segment), response is a prefix of ref_resp cut BELOW resp_max,
    window filled to 64."""
    coll = Collator(tok, 64, 40, "legacy")
    query = words("q", 15)
    context = "zebra " * 50
    response = words("resp", 200)
    out = coll.encode_with_response_budget(query, context, response)
    first, resp = split_output(tok, out)
    assert len(out) == 64
    assert tok.decode(first).startswith(f"Question: {query} Context:")
    assert "zebra" not in tok.decode(first)
    assert is_prefix(resp, ids(tok, response))
    assert 0 < len(resp) < 40


def test_query_alone_too_long_raises(tok):
    """If the query head alone exceeds the budget, nothing valid can be built: ValueError."""
    coll = Collator(tok, 64, 40, "legacy")
    with pytest.raises(ValueError):
        coll.encode_with_response_budget(words("q", 200), "some context", "an answer")


# Collator config check
def test_collator_rejects_resp_max_not_below_budget(tok):
    """resp_max must be < max_length - special tokens, or the response could eat the window.
    test_config.yaml used 256/384 before Phase 1."""
    with pytest.raises(ValueError):
        Collator(tok, 256, 384, "fit_first")
    n_special = tok.num_special_tokens_to_add(pair=True)
    with pytest.raises(ValueError):
        Collator(tok, 256, 256 - n_special, "fit_first")
    Collator(tok, 256, 128, "fit_first")  # test_config.yaml values: must not raise



# Check that the Collator call passes query and pads the output tensors correctly.
def test_call_passes_query_and_pads(tok, coll):
    """Each item carries 'query'; output is padded tensors plus float labels and sources."""
    batch = [
        {"query": "who sat?", "context": "the cat sat on the mat", "response": "a dog barked",
         "label": 1.0, "source": "ragtruth"},
        {"query": "boiling point of water", "context": "water boils at 100 degrees",
         "response": "water boils at 50 degrees", "label": 0.0, "source": "halueval"},
    ]
    out = coll(batch)
    enc = out["enc"]
    assert enc["input_ids"].shape[0] == 2
    assert enc["input_ids"].shape == enc["attention_mask"].shape
    assert enc["labels"].dtype == torch.float
    assert out["sources"] == ["ragtruth", "halueval"]
    expected0 = tok(INPUT_FORMAT.format(query="who sat?", context="the cat sat on the mat"),
                    "a dog barked")["input_ids"]
    n0 = int(enc["attention_mask"][0].sum())
    assert enc["input_ids"][0, :n0].tolist() == expected0


# =============================== fit_first (20 Nov 2026) ===============================
# "Trim only on overflow": available = max_length - special - query part (query never cut).
#   fits (context + response <= available): nothing cut.
#   overflow: response keeps min(len, max(resp_max, available - len(context))); context gets the rest.
#   either part ending with 0 tokens: ValueError.
# The tests above keep specifying the legacy rule, which exists only to reproduce the G runs from before 20 Nov 2026.

from src.encoding import TRUNCATION_MODES, fit_first_lengths, truncation_mode


# ---- the counts alone (no tokenizer) ----
@pytest.mark.parametrize("n_ctx,n_resp,available,resp_max,expected", [
    (300, 200, 1000, 384, (300, 200)),     # fits, response under resp_max
    (300, 600, 1000, 384, (300, 600)),     # fits, response OVER resp_max: still uncut (legacy cut it to 384)
    (400, 600, 1000, 384, (400, 600)),     # exactly fills available: fits
    (5000, 1000, 1000, 384, (616, 384)),   # long context: response protected to resp_max, as legacy
    (5000, 50, 1000, 384, (950, 50)),      # long context, short response: response whole, as legacy
    (100, 2000, 1000, 384, (100, 900)),    # spare room: short context kept whole, response gets 900 > resp_max
    (500, 2000, 1000, 384, (500, 500)),    # spare room smaller: response 500 (> resp_max), context whole
    (700, 2000, 1000, 384, (616, 384)),    # available - context = 300 < resp_max: back to resp_max
])
def test_fit_first_lengths_cases(n_ctx, n_resp, available, resp_max, expected):
    assert fit_first_lengths(n_ctx, n_resp, available, resp_max) == expected


def test_fit_first_lengths_invariants():
    """Over a grid: never more than each part has, never more than available, window filled whenever cut."""
    for n_ctx in (1, 10, 383, 384, 385, 615, 616, 617, 999, 3000):
        for n_resp in (1, 10, 383, 384, 385, 616, 999, 1000, 3000):
            kc, kr = fit_first_lengths(n_ctx, n_resp, 1000, 384)
            assert 0 < kc <= n_ctx and 0 < kr <= n_resp
            assert kc + kr <= 1000
            if (kc, kr) != (n_ctx, n_resp):
                assert kc + kr == 1000
            if n_ctx + n_resp > 1000 and 1000 - n_ctx <= 384:    # long context: legacy's counts
                assert kr == min(n_resp, 384)


@pytest.mark.parametrize("n_ctx,n_resp,available,resp_max", [
    (0, 50, 1000, 384),       # empty context
    (300, 0, 1000, 384),      # empty response, even though the row fits
    (500, 2000, 384, 384),    # long query: resp_max == available, context would get 0
    (500, 2000, 300, 384),    # resp_max > available: context would go below 0
])
def test_fit_first_lengths_zero_raises(n_ctx, n_resp, available, resp_max):
    with pytest.raises(ValueError):
        fit_first_lengths(n_ctx, n_resp, available, resp_max)


# ---- the Collator with the real tokenizer ----
@pytest.fixture(scope="module")
def fit(tok):
    return Collator(tok, MAX_LEN, RESP_MAX, truncation="fit_first")


def test_collator_requires_a_known_truncation_mode(tok):
    """No default: legacy must be asked for by name, so no new code falls back to it silently."""
    with pytest.raises(TypeError):
        Collator(tok, MAX_LEN, RESP_MAX)
    assert TRUNCATION_MODES == ("fit_first", "legacy")
    with pytest.raises(ValueError):
        Collator(tok, MAX_LEN, RESP_MAX, truncation="fit")


def test_truncation_mode_from_config():
    assert truncation_mode({"truncation": "fit_first"}) == "fit_first"
    assert truncation_mode({"truncation": "legacy"}) == "legacy"
    assert truncation_mode({"max_length": 1024}) == "legacy"   # a config from before the option (old G snapshots)
    with pytest.raises(ValueError):
        truncation_mode({"truncation": "FIT_FIRST"})
    assert truncation_mode(BASE["model"]) == "fit_first"       # base.yaml default


def test_fit_first_row_that_fits_is_uncut(tok, coll, fit):
    """Short context + a response longer than resp_max, total under the window: the tokenizer's own pair
    encoding, response whole. Legacy cuts this response to resp_max."""
    context = "the cat sat on the mat"
    response = words("resp", 250)
    assert len(ids(tok, response)) > RESP_MAX                      # guard: the case is what it claims
    expected = tok(INPUT_FORMAT.format(query=QUERY, context=context), response)["input_ids"]
    assert len(expected) <= MAX_LEN
    assert fit.encode_with_response_budget(QUERY, context, response) == expected
    _, legacy_resp = split_output(tok, coll.encode_with_response_budget(QUERY, context, response))
    assert len(legacy_resp) == RESP_MAX


@pytest.mark.parametrize("query,context,response", [
    ("who sat on the mat?", "the cat sat on the mat", "a dog barked"),
    ("butcher shop phone number", "passage 1:Butcher Shop - Hayward 826 B Street\n\npassage 2:Lakefield",
     "The number is (510) 889-8690."),
    ("Qu'est-ce que c'est ?", "Café, naïve, 25°C, 3.14", "It's 25°C."),
    (QUERY, words("ctx", 300), words("resp", 120)),               # 853 tokens, response 240: under both limits
])
def test_fit_first_matches_legacy_when_legacy_cut_nothing(tok, coll, fit, query, context, response):
    legacy = coll.encode_with_response_budget(query, context, response)
    assert legacy == tok(INPUT_FORMAT.format(query=query, context=context), response)["input_ids"]  # legacy left it uncut
    assert fit.encode_with_response_budget(query, context, response) == legacy


@pytest.mark.parametrize("response", [words("resp", 1000), "Anne Frank died in February 1945 at Bergen-Belsen."])
def test_fit_first_long_context_overflow_matches_legacy(tok, coll, fit, response):
    """5,000-word context: the room left for the response is under resp_max, so both rules agree."""
    context = words("ctx", 5000)
    out = fit.encode_with_response_budget(QUERY, context, response)
    assert out == coll.encode_with_response_budget(QUERY, context, response)
    assert len(out) == MAX_LEN


def test_fit_first_spare_room_keeps_more_than_resp_max(tok, fit):
    """Short context, 2,000-word response: the context is kept whole and the response takes the rest of the
    window, more than resp_max."""
    context = "the cat sat on the mat"
    response = words("resp", 2000)
    out = fit.encode_with_response_budget(QUERY, context, response)
    first, resp = split_output(tok, out)
    assert len(out) == MAX_LEN
    assert first == ref_first(tok, QUERY, context)
    assert len(resp) > RESP_MAX
    assert resp == ids(tok, response)[:len(resp)]


def test_fit_first_context_zero_raises(tok):
    """Small window (64, resp_max 40), 15-word query: the query part leaves less than resp_max, so the
    context would end with 0 tokens (legacy allows this: test_overflow_response_gives_way_context_gets_zero)."""
    coll = Collator(tok, 64, 40, truncation="fit_first")
    with pytest.raises(ValueError):
        coll.encode_with_response_budget(words("q", 15), "zebra " * 50, words("resp", 200))


def test_fit_first_response_zero_raises(tok, fit):
    """An empty response leaves the response segment with 0 tokens."""
    with pytest.raises(ValueError):
        fit.encode_with_response_budget(QUERY, "the cat sat on the mat", "")
