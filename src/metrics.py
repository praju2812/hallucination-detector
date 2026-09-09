"""Evaluation metrics.

PR-AUC is the primary metric because the classes are imbalanced (RAGTruth is
~30% positive). Accuracy and ROC-AUC both flatter a model on imbalanced data;
PR-AUC tracks how well you find the positive (hallucinated) class specifically.
"""

from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_fscore_support,
)


def probs_from_logits(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def ranking_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    """Threshold-free metrics. Use these for model selection."""
    return {
        "pr_auc": float(average_precision_score(labels, probs)),
        "roc_auc": float(roc_auc_score(labels, probs)),
    }


def thresholded_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    """Metrics at a specific threshold. Use these for reporting final test results."""

    # binary array if probabilities are greater than or equal to the threshold, else 0
    preds = (probs >= threshold).astype(int)

    # compute precision, recall, and F1 score 
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {"precision": float(p), "recall": float(r), "f1": float(f1), "threshold": threshold}


def best_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    """Pick the threshold that maximises F1 on the VALIDATION set.
    """
    grid = np.linspace(0.05, 0.95, 19)
    f1s = [thresholded_metrics(labels, probs, t)["f1"] for t in grid]
    return float(grid[int(np.argmax(f1s))])


def expected_calibration_error(labels: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """TODO(you): implement ECE.

    You need this because the FastAPI service reports a confidence, and a
    confidence is only meaningful if it's calibrated. Algorithm:
      1. Split [0,1] into n_bins equal-width bins by predicted prob.
      2. For each bin: mean predicted prob (confidence) and mean label (accuracy).
      3. ECE = sum over bins of (bin_size / N) * |accuracy - confidence|.
    Return the scalar. Later you can also plot a reliability diagram from the
    per-bin (confidence, accuracy) pairs.
    """
    raise NotImplementedError


def per_source_breakdown(labels, probs, sources, threshold):
    """Report metrics separately per source (ragtruth vs halueval, and later the
    niche set). This is where the transfer story lives: train general, then read
    off how the same model does on each slice."""
    out = {}
    sources = np.asarray(sources)
    for s in np.unique(sources):
        # boolean mask for examples from this source
        m = sources == s
        # skip sources with no examples (e.g., niche set in the unified train set)
        if m.sum() == 0:
            continue

        # compute metrics for this source
        row = ranking_metrics(labels[m], probs[m])
        # compute thresholded metrics for this source at the given threshold
        row.update(thresholded_metrics(labels[m], probs[m], threshold))

        # add the number of examples from this source to the row
        row["n"] = int(m.sum())
        # add the row to the output dictionary with the source as the key
        out[str(s)] = row
    return out
