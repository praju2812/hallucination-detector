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
    return Collator(tok, MAX_LEN, RESP_MAX)


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
    coll = Collator(tok, 64, 40)
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
    coll = Collator(tok, 64, 40)
    with pytest.raises(ValueError):
        coll.encode_with_response_budget(words("q", 200), "some context", "an answer")


# Collator config check
def test_collator_rejects_resp_max_not_below_budget(tok):
    """resp_max must be < max_length - special tokens, or the response could eat the window.
    test_config.yaml used 256/384 before Phase 1."""
    with pytest.raises(ValueError):
        Collator(tok, 256, 384)
    n_special = tok.num_special_tokens_to_add(pair=True)
    with pytest.raises(ValueError):
        Collator(tok, 256, 256 - n_special)
    Collator(tok, 256, 128)  # test_config.yaml values: must not raise



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
