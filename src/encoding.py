# dataclass automatically generates special methods like __init__() and __repr__() for the class based on the class attributes.

import torch
from transformers import AutoTokenizer
from typing import Any
from dataclasses import dataclass



@dataclass
class Collator:
    """Tokenises a batch of (context, response) pairs."""
    tokenizer: Any
    max_length: int
    resp_max: int


    # The __call__ method allows an instance of the Collator class to be called as a function. 
    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """
        Tokenizes a batch of (context, response) pairs and returns a dictionary containing the tokenized inputs and labels.
        """

        # for each batch item, encode the context and response with respect to the response budget
        encoded = [self.encode_with_response_budget(b["context"], b["response"]) for b in batch]

        # encoded ids are lists of token IDs, but the tokenizer.pad method expects a list of dicts with "input_ids" keys.
        features = [{"input_ids": ids} for ids in encoded]

        # pads the sequences in the batch to the same length and returns a dictionary containing the padded input IDs and attention masks.
        enc = self.tokenizer.pad(features, padding=True, return_tensors="pt")

        # tensor of labels for the batch, converted to float type for compatibility with BCEWithLogitsLoss.
        enc["labels"] = torch.tensor([b["label"] for b in batch], dtype=torch.float)

        # sources kept as plain list for per-source metric breakdown at eval time
        enc_sources = [b["source"] for b in batch]
        return {"enc": enc, "sources": enc_sources}

        
    def encode_with_response_budget(self, context: str, response: str) -> list[int]:
        """
        Encode a (context, response) pair into token IDs, ensuring that the response does not exceed a specified budget.
        Returns a list of token IDs representing the encoded input.
        """

        # tokenize each side alone, no special tokens, no truncation yet
        ctx_ids  = self.tokenizer(context,  add_special_tokens=False)["input_ids"]
        resp_ids = self.tokenizer(response, add_special_tokens=False)["input_ids"]

        # ask the tokenizer how many special tokens a PAIR adds
        n_special = self.tokenizer.num_special_tokens_to_add(pair=True)
        budget = self.max_length - n_special

        # response keeps up to resp_max, but can never exceed the whole budget
        resp_keep = min(len(resp_ids), self.resp_max, budget)
        # context absorbs everything left over
        ctx_keep  = max(0, budget - resp_keep)

        # keeps the START of the answer and drops the end.
        resp_ids = resp_ids[:resp_keep]
        ctx_ids  = ctx_ids[:ctx_keep]

        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        return [cls_id] + ctx_ids + [sep_id] + resp_ids + [sep_id]


def build_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)

# smoke test
if __name__ == "__main__":
    tok = build_tokenizer("answerdotai/ModernBERT-base")
    collate = Collator(tok, 1024, 384)
    batch = [
        {"context": "the cat sat on the mat", "response": "a dog barked", "label": 1, "source": "ragtruth"},
        {"context": "water boils at 100 degrees", "response": "water boils at 50 degrees", "label": 0, "source": "halueval"},
    ]
    out = collate(batch)
    print(out["enc"].keys())
    print("attention_mask" in out["enc"])
    print(out["enc"]["input_ids"].shape)
    print(out["enc"]["attention_mask"][0])
