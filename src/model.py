"""The detector model: an encoder backbone with a binary classification head.

Single logit, not two. A single logit + BCEWithLogitsLoss gives you one clean
probability (sigmoid of the logit) per example. That probability is what your
FastAPI service will return as a hallucination score later, and it's what you
calibrate. Two-logit softmax works too, but single-logit keeps the serving path
simple and the confidence signal unambiguous.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from transformers import AutoModel


class HallucinationDetector(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)   # single logit

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.backbone(**kwargs)

        # Pooling: take the first token ([CLS]) representation.
        # DeBERTa has no trained pooler head, so we pool the last hidden state
        # ourselves. First-token pooling is the default here partly because your
        # RA clustering work already found first-token beat mean pooling on
        # SPECTER; worth checking whether that holds for this task too.
        # TODO(you): try mean pooling over non-pad tokens as an ablation and
        # compare PR-AUC. Mean pooling needs the attention_mask to exclude pads.
        pooled = out.last_hidden_state[:, 0]        # [batch, hidden]

        pooled = self.dropout(pooled)
        logits = self.head(pooled).squeeze(-1)      # [batch]
        return logits


def build_loss(pos_weight: float):
    """Binary Cross Entropy with class weighting for the imbalanced positive (hallucinated) class."""
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
