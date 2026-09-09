"""Data loading and collation for the hallucination detector.

Design: this is a cross-encoder. Each example is a (context, response) pair fed
to one encoder as text + text_pair. The model sees both together and predicts
whether the response is faithful to the context.

Key correctness detail below: when the pair is too long, we truncate the
CONTEXT and keep the RESPONSE intact. You cannot judge an answer you can't
fully see, so the response must never be the thing that gets cut.
"""

from __future__ import annotations
from typing import Any

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from torch.utils.data import Dataset
from utils.io import read_jsonl

class HallucinationDataset(Dataset):
    """Holds raw rows. Tokenisation happens in the collator for dynamic padding."""

    def __init__(self, path: str, cfg_data: dict):
        """
        Initialise the dataset.
        Store the raw rows and the column names for context, response, label, and source.

        Args:   
            path: Path to the JSONL file containing the data.
            cfg_data: Configuration dictionary with schema mapping.
        """
        self.rows = read_jsonl(path)
        self.context = cfg_data["context_col"]
        self.response = cfg_data["response_col"]
        self.label = cfg_data["label_col"]
        self.source = cfg_data.get("dataset_col")

    def __len__(self) -> int:
        return len(self.rows)
    
    # __getitem__ allows the dataset to be indexed like a list.
    def __getitem__(self, i: int) -> dict[str, Any]:
        """
        Get the i-th example from the dataset.
        Returns a dictionary with keys: 'context', 'response', 'label', and 'source'.
        """
        r = self.rows[i]

        # if the context is a list of strings, join them with double newlines
        context = r[self.context]
        if isinstance(context, list):
            context = "\n\n".join(context)

        return {
            "context": context,
            "response": r[self.response],
            "label": float(r[self.label]),
            "source": r.get(self.source, "unknown") if self.source else "unknown",
        }

