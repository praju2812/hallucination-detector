# dataclass automatically generates special methods like __init__() and __repr__() for the class based on the class attributes.

import torch
from transformers import AutoTokenizer
from typing import Any
from dataclasses import dataclass


# Detector input: first segment = this string, second segment = the response.
# Recorded in config_snapshot.json as `input_format`.
INPUT_FORMAT = "Question: {query} Context: {context}"


@dataclass
class Collator:
    """Tokenises a batch of (query, context, response) triples."""
    tokenizer: Any
    max_length: int
    resp_max: int

    def __post_init__(self):
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
            1. query part ("Question: {query} Context:"): never truncated.
            2. response: keeps up to resp_max tokens, but only what is left after the query part.
            3. context: gets whatever is left (may be 0 tokens).

        Raises ValueError if:
            - the query part alone does not fit in the budget;
            - the query part's tokens are not a prefix of the first segment's tokens
              (a tokenizer boundary change at "Context:" would otherwise make the
              query-part count wrong without any error).

        Invariants (tests/test_encoding.py):
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
    collate = Collator(tok, cfg["model"]["max_length"], cfg["model"]["resp_max"])
    batch = [
        {"query": "where did the cat sit?", "context": "the cat sat on the mat", "response": "a dog barked", "label": 1, "source": "ragtruth"},
        {"query": "boiling point of water", "context": "water boils at 100 degrees", "response": "water boils at 50 degrees", "label": 0, "source": "halueval"},
    ]
    out = collate(batch)
    print(out["enc"].keys())
    print("attention_mask" in out["enc"])
    print(out["enc"]["input_ids"].shape)
    print(out["enc"]["attention_mask"][0])
