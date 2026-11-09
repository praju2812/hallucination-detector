"""per_source_breakdown records the positive count per source, so prevalence
can be shown next to PR-AUC.

Run: python -m pytest tests/test_metrics.py
"""

import numpy as np

from src.metrics import per_source_breakdown


def test_per_source_breakdown_has_n_pos():
    labels = np.array([1, 0, 0, 1, 1, 0, 0, 0, 1])
    probs = np.array([0.9, 0.2, 0.4, 0.7, 0.6, 0.1, 0.3, 0.8, 0.55])
    sources = ["ragtruth"] * 5 + ["halueval"] * 4
    out = per_source_breakdown(labels, probs, sources, threshold=0.5)
    assert out["ragtruth"]["n"] == 5 and out["ragtruth"]["n_pos"] == 3
    assert out["halueval"]["n"] == 4 and out["halueval"]["n_pos"] == 1
    assert isinstance(out["ragtruth"]["n_pos"], int)  # plain int, so json.dump works


def test_per_source_breakdown_n_pos_float_labels():
    """Labels arrive as floats from the collator (BCE); n_pos must still be an int count."""
    labels = np.array([1.0, 0.0, 1.0, 0.0])
    probs = np.array([0.9, 0.1, 0.6, 0.4])
    out = per_source_breakdown(labels, probs, ["ragtruth"] * 4, threshold=0.5)
    assert out["ragtruth"]["n_pos"] == 2
    assert type(out["ragtruth"]["n_pos"]) is int
