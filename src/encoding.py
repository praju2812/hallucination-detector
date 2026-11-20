# dataclass automatically generates special methods like __init__() and __repr__() for the class based on the class attributes.

import torch
from transformers import AutoTokenizer
from typing import Any
from dataclasses import dataclass


# Detector input: first segment = this string, second segment = the response.
# Recorded in config_snapshot.json as `input_format`.
INPUT_FORMAT = "Question: {query} Context: {context}"

# How encode_with_response_budget truncates (config key model.truncation, recorded in config_snapshot.json).
#   fit_first: nothing is cut when the row fits; on overflow see fit_first_lengths. Every new run (G retrained, C, G+C).
#   legacy:    the response is always cut to resp_max, the context gets the rest. Kept ONLY to reproduce the
#              G runs trained before 20 Nov 2026 (decision, 20 Nov 2026).
TRUNCATION_MODES = ("fit_first", "legacy")


def truncation_mode(model_cfg: dict) -> str:
    """The truncation mode for a config's `model` section.

    A config without the key predates the option, so it was encoded with the legacy rule: an old G run's
    config_snapshot.json therefore still scores with legacy. New configs set the key explicitly (base.yaml: fit_first).
    Raises ValueError for any other value.
    """
    mode = model_cfg.get("truncation", "legacy")
    if mode not in TRUNCATION_MODES:
        raise ValueError(f"model.truncation must be one of {TRUNCATION_MODES}, got {mode!r}")
    return mode


def fit_first_lengths(n_context: int, n_response: int, available: int, resp_max: int) -> tuple[int, int]:
    """How many context and response tokens the fit_first rule keeps ("trim only on overflow").

    Inputs (token counts, special tokens and the query part already excluded):
        n_context:  context tokens = first-segment tokens after the query part.
        n_response: response tokens.
        available:  max_length - special tokens - query-part tokens (the query is never cut).
        resp_max:   the response's protected length (config model.resp_max).
    Output:
        (keep_context, keep_response), each a count of tokens kept from the START of that part.
    Rule:
        - if n_context + n_response <= available: nothing is cut.
        - else: the response keeps min(n_response, max(resp_max, available - n_context));
                the context gets the rest of `available`.
    Invariants (tests/test_encoding.py):
        - keep_context <= n_context, keep_response <= n_response, keep_context + keep_response <= available.
        - whenever anything is cut, keep_context + keep_response == available.
        - long context (available - n_context <= resp_max): same counts as the legacy rule.
        - spare room (short context, long response): keep_response can exceed resp_max.
    Raises:
        ValueError if keep_context or keep_response would be 0 (or below 0).
    """
    # refuse a result where either part ends with no tokens
    if n_context <= 0 or n_response <= 0:
        raise ValueError(f"fit_first_lengths needs a non-empty context and response "
                 f"(n_context={n_context}, n_response={n_response})")
    
    # the case where everything fits
    if n_context + n_response <= available:
        return n_context, n_response
    
    # on overflow, the response's share first (the max() lets it use room a short context leaves)
    keep_response = min(n_response, max(resp_max, available - n_context))

    # the context's share is what remains of `available`
    keep_context = available - keep_response

    # refuse a result where either part ends with no tokens
    if keep_context <= 0 or keep_response <= 0:
        raise ValueError(f"fit_first_lengths would keep {keep_context} context and {keep_response} response tokens; "
                         f"must keep at least 1 of each (available={available}, n_context={n_context}, "
                         f"n_response={n_response}, resp_max={resp_max})")
    return keep_context, keep_response

@dataclass
class Collator:
    """Tokenises a batch of (query, context, response) triples."""
    tokenizer: Any
    max_length: int
    resp_max: int
    truncation: str  # required, no default: callers pass truncation_mode(cfg["model"]), or "legacy" only to reproduce old G runs

    def __post_init__(self):
        if self.truncation not in TRUNCATION_MODES:
            raise ValueError(f"truncation must be one of {TRUNCATION_MODES}, got {self.truncation!r}")
        # resp_max must leave room for the query: otherwise the response alone could fill the window.
        n_special = self.tokenizer.num_special_tokens_to_add(pair=True)
        if self.resp_max >= self.max_length - n_special:
            raise ValueError(
                f"resp_max ({self.resp_max}) must be < max_length ({self.max_length}) - "
                f"{n_special} special tokens; check the config's model section")

    # The __call__ method allows an instance of the Collator class to be called as a function.
    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """
        Tokenizes a batch of (query, context, response) triples and returns a dictionary containing the tokenized inputs and labels.
        """

        # for each batch item, encode the query, context and response with respect to the response budget
        encoded = [self.encode_with_response_budget(b["query"], b["context"], b["response"]) for b in batch]

        # encoded ids are lists of token IDs, but the tokenizer.pad method expects a list of dicts with "input_ids" keys.
        features = [{"input_ids": ids} for ids in encoded]

        # pads the sequences in the batch to the same length and returns a dictionary containing the padded input IDs and attention masks.
        enc = self.tokenizer.pad(features, padding=True, return_tensors="pt")

        # tensor of labels for the batch, converted to float type for compatibility with BCEWithLogitsLoss.
        enc["labels"] = torch.tensor([b["label"] for b in batch], dtype=torch.float)

        # sources kept as plain list for per-source metric breakdown at eval time
        enc_sources = [b["source"] for b in batch]
        return {"enc": enc, "sources": enc_sources}


    def encode_with_response_budget(self, query: str, context: str, response: str) -> list[int]:
        """
        Encode a (query, context, response) triple into token IDs for a pair cross-encoder.

        Output layout:  [CLS] first_segment [SEP] response_segment [SEP]
            first_segment comes from INPUT_FORMAT.format(query=query, context=context).

        Budget = max_length - num_special_tokens_to_add(pair=True)
        self.truncation == "legacy":
            1. query part ("Question: {query} Context:"): never truncated.
            2. response: keeps up to resp_max tokens, but only what is left after the query part.
            3. context: gets whatever is left (may be 0 tokens).
        self.truncation == "fit_first": the query part is never truncated; the context and response counts come
            from fit_first_lengths (nothing cut when the row fits). Raises ValueError if either ends with 0 tokens.

        Raises ValueError if:
            - the query part alone does not fit in the budget;
            - the query part's tokens are not a prefix of the first segment's tokens
              (a tokenizer boundary change at "Context:" would otherwise make the
              query-part count wrong without any error).

        Invariants, legacy (tests/test_encoding.py):
            - short inputs: identical to tokenizer(INPUT_FORMAT.format(...), response)["input_ids"].
            - kept first segment is a PREFIX of tokenizer(INPUT_FORMAT.format(...), add_special_tokens=False).
            - kept response is a PREFIX of tokenizer(response, add_special_tokens=False), never longer than resp_max.
            - len(output) <= max_length; == max_length whenever anything was truncated.
            - only the context is cut, except in overflow (query + response > budget), where the response gives way.
        """
        # tokenize the response alone (no special tokens), as in v1.
        resp_ids = self.tokenizer(response, add_special_tokens=False)["input_ids"]

        # tokenize the first segment as ONE string built from INPUT_FORMAT (no special tokens).
        first_ids = self.tokenizer(INPUT_FORMAT.format(query=query, context=context), add_special_tokens=False)["input_ids"]

        #work out how many of those first-segment tokens belong to the query part.
        query_part_ids = self.tokenizer(INPUT_FORMAT.format(query=query, context="").rstrip(), add_special_tokens=False)["input_ids"]

        # budget arithmetic in priority order: query part, then response (<= resp_max), then context.
        budget = self.max_length - self.tokenizer.num_special_tokens_to_add(pair=True)
        if len(query_part_ids) > budget:
            raise ValueError(f"query part alone ({len(query_part_ids)} tokens) exceeds budget ({budget})")
        
        if query_part_ids != first_ids[:len(query_part_ids)]:
            raise ValueError("query part is not a prefix of the first segment; check INPUT_FORMAT")

        if self.truncation == "fit_first":
            # counts come from fit_first_lengths; each part keeps its first tokens, the query part is untouched.
            n_query = len(query_part_ids)
            keep_ctx, keep_resp = fit_first_lengths(len(first_ids) - n_query, len(resp_ids), budget - n_query,
                                                    self.resp_max)
            first_ids, resp_ids = first_ids[:n_query + keep_ctx], resp_ids[:keep_resp]
        else:
            # legacy rule, unchanged
            if len(resp_ids) > self.resp_max:
                # cut the response to resp_max
                resp_ids = resp_ids[:self.resp_max]

            # if the query part + response exceeds the budget, cut the response to fit.
            if len(query_part_ids) + len(resp_ids) > budget:
                resp_ids = resp_ids[:budget - len(query_part_ids)]

            # query and response are now within budget; if the first segment is too long, cut it to fit the remaining budget.
            if len(resp_ids) + len(first_ids) > budget:
                first_ids = first_ids[:budget - len(resp_ids)]

        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        return [cls_id] + first_ids + [sep_id] + resp_ids + [sep_id]
def build_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)

# smoke test
if __name__ == "__main__":
    import yaml
    with open("configs/base.yaml") as f:
        cfg = yaml.safe_load(f)
    tok = build_tokenizer(cfg["model"]["name"])
    collate = Collator(tok, cfg["model"]["max_length"], cfg["model"]["resp_max"], truncation_mode(cfg["model"]))
    batch = [
        {"query": "where did the cat sit?", "context": "the cat sat on the mat", "response": "a dog barked", "label": 1, "source": "ragtruth"},
        {"query": "boiling point of water", "context": "water boils at 100 degrees", "response": "water boils at 50 degrees", "label": 0, "source": "halueval"},
    ]
    out = collate(batch)
    print(out["enc"].keys())
    print("attention_mask" in out["enc"])
    print(out["enc"]["input_ids"].shape)
    print(out["enc"]["attention_mask"][0])
