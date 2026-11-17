"""Phase 4 (RQ1): G evaluated zero-shot on clinical data.

Core logic for scoring G checkpoints and computing the RQ1 metrics. The CLIs are
scripts/score_rq1.py (runs the model once per seed) and scripts/report_rq1.py (metrics + tables).

Vocabulary used below:
    pred      one prediction record (a dict) as written to results/rq1/<run>/predictions.jsonl:
              split, idx, source, task, group_id, label (0/1 int), prob (float), row_key, difficulty, category
    thr (a)   "ragtruth threshold": the seed's threshold tuned on RAGTruth validation in Phase 2
              (runs/<run>/metrics.json["threshold"]). Primary, because it is truly zero-shot.
    thr (b)   "clinical threshold": tuned on clean clinical validation, same grid as training. Secondary.

Spec: tests/test_rq1.py.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
from typing import Callable

import numpy as np
import torch

from src.encoding import INPUT_FORMAT
from src.metrics import best_threshold, ranking_metrics, thresholded_metrics  # reuse these, do not rewrite them

# What a reportable G checkpoint must have been trained with
EXPECTED_EXPERIMENT = "G"
EXPECTED_MAX_LENGTH = 1024
EXPECTED_RESP_MAX = 384

# Threshold (b) is tuned per source on its own validation rows, except where listed here.
# CovidQA has only 17 validation positives, so it uses pooled RAGBench validation.
POOLED_VAL = {"ragbench_covidqa": ["ragbench_pubmedqa", "ragbench_covidqa"]}


# ---- loading a G checkpoint ------------------------------------------------------------

def check_g_snapshot(snapshot: dict) -> None:
    """Raise ValueError unless `snapshot` (a run's config_snapshot.json) is a G run in the current input format.

    Checks:
        snapshot["experiment"] == EXPECTED_EXPERIMENT
        snapshot["input_format"] == INPUT_FORMAT (the format src/encoding.py builds today)
        snapshot["config"]["model"]["max_length"] == EXPECTED_MAX_LENGTH
        snapshot["config"]["model"]["resp_max"] == EXPECTED_RESP_MAX
    Returns None when all checks pass.

    Why: the checkpoint is fed by today's Collator. A run trained with another format or window would be
    scored with inputs it never saw, and nothing would crash.
    """
    # read each of the four values; a missing key must also give ValueError (not KeyError).
    try:
        experiment = snapshot["experiment"]
        input_format = snapshot["input_format"]
        max_length = snapshot["config"]["model"]["max_length"]
        resp_max = snapshot["config"]["model"]["resp_max"]
    except KeyError as e:
        raise ValueError(f"Missing key in snapshot: {e}")

    # compare each with its expected value.
    # raise ValueError naming the field, the expected value and the value found

    if experiment != EXPECTED_EXPERIMENT:
        raise ValueError(f"Expected experiment {EXPECTED_EXPERIMENT}, got {experiment}")
    if input_format != INPUT_FORMAT:
        raise ValueError(f"Expected input_format {INPUT_FORMAT}, got {input_format}")
    if max_length != EXPECTED_MAX_LENGTH:
        raise ValueError(f"Expected max_length {EXPECTED_MAX_LENGTH}, got {max_length}")
    if resp_max != EXPECTED_RESP_MAX:
        raise ValueError(f"Expected resp_max {EXPECTED_RESP_MAX}, got {resp_max}")

def _default_build_model(cfg: dict):
    """The real detector, built from the run's own config (not configs/base.yaml)."""
    from src.model import HallucinationDetector
    return HallucinationDetector(cfg["model"]["name"], cfg["model"]["dropout"])


def load_g_checkpoint(run_dir: str, device: str, build_model: Callable[[dict], object] | None = None):
    """Load a G run's best.pt for inference.

    Inputs:
        run_dir: a Phase 2 run folder containing config_snapshot.json and best.pt.
        device: "mps", "cuda" or "cpu".
        build_model: function cfg -> untrained nn.Module with the same architecture. None -> _default_build_model.
                     (Tests pass a tiny module so they do not download ModernBERT.)
    Output:
        (model, snapshot): the model on `device`, in eval mode, with best.pt's weights;
        snapshot = the parsed config_snapshot.json.
    Invariants:
        - check_g_snapshot runs BEFORE the weights are loaded (a wrong run fails fast).
        - every key in best.pt matches the model (a missing or extra key is an error, never ignored).
        - the model is in eval mode (dropout off), so the same input always gives the same prob.
    """
    # read config_snapshot.json from run_dir and validate it.
    snapshot_path = os.path.join(run_dir, "config_snapshot.json")
    if not os.path.exists(snapshot_path):
        raise FileNotFoundError(f"Missing config_snapshot.json in {run_dir}")
    with open(snapshot_path, "r") as f:
        snapshot = json.load(f)
    check_g_snapshot(snapshot)

    # build the model from snapshot["config"] (use the default builder when none is given).
    cfg = snapshot["config"]
    if build_model is None:
        build_model = _default_build_model
    model = build_model(cfg)

    # load best.pt so that it works on a machine without CUDA (the weights were saved on a T4).
    best_point_path = os.path.join(run_dir, "best.pt")
    if not os.path.exists(best_point_path):
        raise FileNotFoundError(f"Missing best.pt in {run_dir}")
    state_dict = torch.load(best_point_path, map_location="cpu", weights_only=True)  # load on CPU first, then move to device

    # put the weights into the model; make sure mismatched keys raise.
    model.load_state_dict(state_dict, strict=True)

    # move to device, switch to eval mode, return (model, snapshot).
    model.to(device)
    model.eval()
    return model, snapshot

# ---- row keys and the clean test subset ------------------------------------------------------

def assign_row_keys(rows: list[dict]) -> list[str]:
    """Stable key for every row: "<sha1 hex of 'source|group_id|answer'>#<occurrence>".

    Input:  rows from a clinical split file (dicts with source, group_id, answer), in file order.
    Output: list of keys, same length and order as rows.
    Rules:
        - base = hashlib.sha1(f"{source}|{group_id}|{answer}".encode("utf-8")).hexdigest()
        - occurrence = how many EARLIER rows in `rows` had the same base (0 for the first one).
          test_full has 3 identical CovidQA rows (decision, 17 Nov: keep all), so they become #0, #1, #2.
        - nothing else (query, context, label) goes into the key.
    Invariant: a row gets the same key in test_full and in clean test as long as none of its duplicates was
    removed before it (true today: the duplicated rows are not in clean test; derive_clean_predictions checks).
    """
    # build the base string for each row and hash it.
    bases = [hashlib.sha1(f"{r['source']}|{r['group_id']}|{r['answer']}".encode("utf-8")).hexdigest() for r in rows]
    
    # count occurrences as you go, in file order, and append "#<count>".
    keys = []
    counts = {}
    for base in bases:
        counts[base] = counts.get(base, 0) + 1
        keys.append(f"{base}#{counts[base] - 1}")
    return keys


def derive_clean_predictions(full_preds: list[dict], clean_rows: list[dict]) -> list[dict]:
    """Clean-test predictions taken from the test_full predictions, so no test row is scored twice.

    Inputs:
        full_preds: predictions for test_full (each has row_key, label, prob, ...).
        clean_rows: rows of data/clinical/test.jsonl, in file order.
    Output:
        one NEW dict per clean row, in clean file order: a copy of the matching full pred with
        split = "test_clean" and idx = position in the clean file. full_preds is not modified.
    Raises ValueError if:
        - two full preds share a row_key (the key would not identify one row);
        - a clean row has no matching full pred;
        - a clean row's label differs from its full pred's label;
        - a clean row's base key (the part before "#") occurs more than once in full_preds
          (ambiguous: which copy is the clean row?).
    """
    # index full_preds by row_key; detect repeated keys.
    key_to_pred = {}
    for pred in full_preds:
        key = pred["row_key"]
        if key in key_to_pred:
            raise ValueError(f"Duplicate row_key in full_preds: {key}")
        key_to_pred[key] = pred
    
    # compute the clean rows' keys with assign_row_keys.
    clean_keys = assign_row_keys(clean_rows)

    # check that each clean row has a matching full pred, that the labels match, and that the base key is unique in full_preds.
    base_counts = collections.Counter(k.split("#")[0] for k in key_to_pred)

    # for each clean row, find its full pred and run the checks listed above.
    for idx, (row, key) in enumerate(zip(clean_rows, clean_keys)):
        if key not in key_to_pred:
            raise ValueError(f"Clean row {idx} has no matching full pred: {key}")
        pred = key_to_pred[key]
        if row["label"] != pred["label"]:
            raise ValueError(f"Clean row {idx} label {row['label']} differs from full pred label {pred['label']}")
        base_key = key.split("#")[0]
        if base_counts[base_key] > 1:
            raise ValueError(f"Clean row {idx} base key {base_key} occurs more than once in full_preds")
        
    # build the output records (copies, new split and idx)
    clean_preds = []
    for idx, key in enumerate(clean_keys):
        pred = key_to_pred[key]
        p = dict(pred)  # shallow copy is enough because values are plain types
        p["split"] = "test_clean"
        p["idx"] = idx
        clean_preds.append(p)
    return clean_preds

# ---- thresholds and metrics ------------------------------------------------------------

def tune_clinical_thresholds(val_preds: list[dict], pooled: dict[str, list[str]] = POOLED_VAL) -> dict[str, dict]:
    """Threshold (b) per clinical source, tuned on clean clinical validation predictions.

    Input:
        val_preds: predictions for data/clinical/val.jsonl (all sources).
        pooled: source -> list of sources whose validation rows are pooled to tune it. A source that is not
                listed uses only its own rows.
    Output:
        {source: {"threshold": float, "tuned_on": sorted list of source names, "n": int, "n_pos": int}}
        one entry per source present in val_preds; n and n_pos count the rows actually used for tuning.
    Rules:
        - threshold = src.metrics.best_threshold(labels, probs): the same F1 grid (0.05..0.95) as training.
    Raises ValueError if a source listed in `pooled` is absent from val_preds, or if the rows used for a
    source contain no positive (F1 would be 0 at every threshold and argmax would pick 0.05 silently).
    """
    # find the sources present in val_preds.
    sources = sorted(set(pred["source"] for pred in val_preds))

    # for each, decide which sources' rows it is tuned on.
    source_to_tune_on = {}
    for source in sources:
        if source in pooled:
            tune_on = pooled[source]
            for s in tune_on:
                if s not in sources:
                    raise ValueError(f"Source {s} listed in pooled tuning for {source} is absent from val_preds")
            source_to_tune_on[source] = tune_on
        else:
            source_to_tune_on[source] = [source]

    # collect labels and probs for those rows; check them; call best_threshold
    source_to_result = {}
    for source, tune_on in source_to_tune_on.items():
        labels = [pred["label"] for pred in val_preds if pred["source"] in tune_on]
        probs = [pred["prob"] for pred in val_preds if pred["source"] in tune_on]
        n = len(labels)
        n_pos = int(sum(labels))
        if n_pos == 0:
            raise ValueError(f"No positive rows for tuning threshold of source {source} on {tune_on}")
        threshold = float(best_threshold(np.array(labels), np.array(probs)))
        source_to_result[source] = {"threshold": threshold, "tuned_on": sorted(tune_on), "n": n, "n_pos": n_pos}
    return source_to_result

def faithful_class_precision(labels: np.ndarray, probs: np.ndarray, threshold: float) -> float | None:
    """Precision of the FAITHFUL class: of the rows predicted faithful, the fraction truly faithful.

    Predicted faithful = prob < threshold (the complement of thresholded_metrics' rule prob >= threshold).
    Value = TN / (TN + FN), with label 0 = faithful, 1 = hallucinated.
    Returns None when no row is predicted faithful (undefined, not 0).

    Why (proposal commitment): MedHallu's faithful answers can be judged "unsupported" because the abstract's
    conclusion is withheld from the context, so low faithful-class precision is part of the construct story.
    """
    # Create a predicted-faithful mask.
    pred_faithful_mask = probs < threshold

    # count TN and FN among those rows
    true_labels = labels[pred_faithful_mask]
    tn = np.sum(true_labels == 0)
    fn = np.sum(true_labels == 1)
    
    # handle the empty case; return a plain float.
    if tn + fn == 0:
        return None
    return float(tn / (tn + fn))


def subset_metrics(preds: list[dict], thr_ragtruth: float, thr_clinical: float) -> dict:
    """All RQ1 metrics for one subset of predictions (one source, or one MedHallu difficulty/category).

    Output (plain Python types, JSON-serialisable):
        {"n": int, "n_pos": int, "prevalence": float,
         "pr_auc": float | None, "roc_auc": float | None,          # None if the subset has one class only
         "at_ragtruth_thr": {"threshold", "precision", "recall", "f1", "faithful_precision"},
         "at_clinical_thr": {... same keys ...}}
    Rules:
        - pr_auc / roc_auc from src.metrics.ranking_metrics.
        - precision / recall / f1 from src.metrics.thresholded_metrics (prob >= threshold -> hallucinated).
        - faithful_precision from faithful_class_precision.
    Raises ValueError if preds is empty or any label is not 0/1.
    """
    # arrays of labels and probs; validate.
    labels = np.array([pred["label"] for pred in preds])
    probs = np.array([pred["prob"] for pred in preds])
    if len(labels) == 0:
        raise ValueError("subset_metrics: preds is empty")
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("subset_metrics: some labels are not 0/1")
    
    # counts and prevalence
    n = len(labels)
    n_pos = int(np.sum(labels))

    if n_pos == 0 or n_pos == n:
        prevalence = 0.0 if n_pos == 0 else 1.0
        pr_auc = None
        roc_auc = None
    else:
        prevalence = n_pos / n
        # ranking metrics, guarded for a single-class subset.
        ranking = ranking_metrics(labels, probs)
        pr_auc = float(ranking["pr_auc"]) if ranking["pr_auc"] is not None else None
        roc_auc = float(ranking["roc_auc"]) if ranking["roc_auc"] is not None else None

    # the two threshold blocks (write one helper and call it twice)
    def threshold_block(threshold: float) -> dict:
        thresh_metrics = thresholded_metrics(labels, probs, threshold)
        faithful_prec = faithful_class_precision(labels, probs, threshold)
        return {
            "threshold": threshold,
            "precision": thresh_metrics["precision"],
            "recall": thresh_metrics["recall"],
            "f1": thresh_metrics["f1"],
            "faithful_precision": faithful_prec
        }

    return {
        "n": n,
        "n_pos": n_pos,
        "prevalence": prevalence,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "at_ragtruth_thr": threshold_block(thr_ragtruth),
        "at_clinical_thr": threshold_block(thr_clinical)
    }

def cluster_bootstrap_ci(labels: np.ndarray, probs: np.ndarray, groups: np.ndarray,
                         metric_fns: dict[str, Callable[[np.ndarray, np.ndarray], float]],
                         n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict[str, dict]:
    """Percentile bootstrap CIs that resample whole groups (group_id), not rows.

    Inputs:
        labels, probs, groups: aligned 1-D arrays (one entry per row).
        metric_fns: name -> f(labels, probs) -> float, e.g. PR-AUC, AUROC, F1 at a FIXED threshold.
        n_boot: number of resamples; seed: for np.random.default_rng; alpha: 0.05 -> 95% CI.
    Output:
        {name: {"point": float, "lo": float | None, "hi": float | None, "n_valid": int}}
        point = metric on the original data; lo/hi = the alpha/2 and 1 - alpha/2 percentiles of the
        resampled values; n_valid = resamples actually used.
    Rules:
        - one resample = draw len(unique groups) groups WITH replacement; take ALL rows of each drawn group
          (a group drawn twice contributes its rows twice).
        - all metrics use the same resamples.
        - a resample whose labels are all one class is skipped for every metric (not counted in n_valid).
        - n_valid == 0 -> lo and hi are None.
    Raises ValueError if the three arrays differ in length.
    Why groups: rows of one group share a question/passage, so they are not independent; resampling rows
    would make the CI too narrow.
    """
    #  validate lengths
    if not (len(labels) == len(probs) == len(groups)):
        raise ValueError("labels, probs, groups must have the same length")
    
    # map each group to the row indices it owns.
    group_to_indices = {group: np.where(groups == group)[0] for group in np.unique(groups)}

    # point estimates on the full data
    point_estimates = {name: fn(labels, probs) for name, fn in metric_fns.items()}

    # the resampling loop (one generator, created once from `seed`)
    rng = np.random.default_rng(seed)
    resampled_metrics = {name: [] for name in metric_fns}
    unique_groups = np.unique(groups)
    for _ in range(n_boot):
        # draw groups with replacement, collect their row indices, and get the resampled labels/probs
        drawn_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        resample_indices = np.concatenate([group_to_indices[g] for g in drawn_groups])
        resample_labels = labels[resample_indices]
        resample_probs = probs[resample_indices]

        # skip this resample if it has only one class; otherwise compute each metric and store it
        if len(np.unique(resample_labels)) < 2:
            continue
        for name, fn in metric_fns.items():
            resampled_metrics[name].append(fn(resample_labels, resample_probs))

    # compute percentiles; plain floats in the output.
    ci_results = {}
    for name, values in resampled_metrics.items():
        n_valid = len(values)
        if n_valid == 0:
            lo = hi = None
        else:
            lo = float(np.percentile(values, 100 * (alpha / 2)))
            hi = float(np.percentile(values, 100 * (1 - alpha / 2)))
        ci_results[name] = {
            "point": float(point_estimates[name]),
            "lo": lo,
            "hi": hi,
            "n_valid": n_valid
        }
    return ci_results

def spread(values: list[float]) -> dict:
    """Across-seed summary: {"mean", "std", "min", "max"}; std = SAMPLE standard deviation (n - 1).

    One value -> std None (undefined). Raises ValueError on an empty list or any None value
    (a None means a metric was undefined for some seed; averaging around it would hide that).
    Same convention as scripts/summarize_runs.summarize_source (decision D5).
    """
    # validate and convert to np.array
    if len(values) == 0:
        raise ValueError("spread: values is empty")
    if any(v is None for v in values):
        raise ValueError("spread: values contains None")
    
    arr = np.array(values, dtype=float)

    # compute mean, sample SD, min, max as plain floats.
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else None
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))

    return {"mean": mean, "std": std, "min": min_val, "max": max_val}