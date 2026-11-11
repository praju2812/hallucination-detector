"""Shared plumbing for the read-only supervisor analysis (scripts/analysis/).

Nothing here changes data, configs or the pipeline. Clinical data logic is not reimplemented here: Part B and the
clinical half of Part A import Prajakta's functions from scripts/build_clinical.py, pinned to BUILD_CLINICAL_SHA.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy import stats
from sklearn.metrics import roc_auc_score

from src.encoding import INPUT_FORMAT, Collator, build_tokenizer
from utils.io import read_jsonl

REPO = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO / "results" / "analysis_2026-11-11"
UNIFIED_V2 = REPO / "data" / "unified_v2"
RAGTRUTH_RAW = REPO / "data" / "raw" / "RAGTruth" / "dataset"
SPLITS = ("train", "val", "test")
EXPECTED_ROWS = {"train": 25008, "val": 2784, "test": 3800}
RUNS_GLOB = str(REPO.parent / "hd-phase2" / "runs" / "run_seed*" / "predictions.jsonl")
N_BOOT = 2000
BOOT_SEED = 0
BUILD_CLINICAL_SHA = "ef2537212375d5cdc87f7e0c88c11a70961702fd"  # commit whose scripts/build_clinical.py the clinical analysis imports
CLINICAL_SEED, CLINICAL_VAL_FRAC = 42, 0.1  # build_clinical.main() defaults


def import_build_clinical(sha: str = BUILD_CLINICAL_SHA):
    """Return (scripts.build_clinical module, full SHA), refusing if the working-tree file differs from `sha`."""
    full = subprocess.run(["git", "rev-parse", sha], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    diff = subprocess.run(["git", "diff", "--quiet", sha, "--", "scripts/build_clinical.py"], cwd=REPO)
    if diff.returncode != 0:
        raise RuntimeError(f"scripts/build_clinical.py differs from {sha}; the analysis is pinned to that commit")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # use the cached, revision-pinned datasets
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    import scripts.build_clinical as bc
    return bc, full


def clinical_rows_before_dedup(bc, raw: dict, seed: int = CLINICAL_SEED, val_frac: float = CLINICAL_VAL_FRAC) -> dict:
    """{split: {source: rows}} built with build_clinical's own functions, BEFORE any training-side removal.

    RAGBench: official splits, unlabelled rows dropped by drop_unlabelled (its guard). MedHallu: pqa_labeled = test;
    pqa_artificial split 90/10 by split_pqa_artificial (same seed and val_frac as build_clinical.main()).
    """
    out = {s: {} for s in SPLITS}
    for source in bc.RAGBENCH_SOURCES.values():
        for split in SPLITS:
            rows, _ = bc.drop_unlabelled(bc.ragbench_rows(raw[source][split], source), source, split)
            out[split][source] = rows
    art_train, art_val = bc.split_pqa_artificial(bc.medhallu_rows(raw["pqa_artificial"]), seed, val_frac)
    out["train"][bc.MEDHALLU], out["val"][bc.MEDHALLU] = art_train, art_val
    out["test"][bc.MEDHALLU] = bc.medhallu_rows(raw["pqa_labeled"])
    return out


def load_config(path: str | Path = REPO / "configs" / "base.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_tokenizer(cfg: dict | None = None):
    cfg = cfg or load_config()
    return build_tokenizer(cfg["model"]["name"])


def load_unified_v2() -> dict[str, list[dict]]:
    """{split: rows} from data/unified_v2, row counts asserted against the known totals."""
    out = {}
    for split in SPLITS:
        rows = read_jsonl(str(UNIFIED_V2 / f"{split}.jsonl"))
        if len(rows) != EXPECTED_ROWS[split]:
            raise ValueError(f"unified_v2/{split}: {len(rows)} rows, expected {EXPECTED_ROWS[split]}")
        out[split] = rows
    return out


def load_source_info() -> dict[str, dict]:
    """RAGTruth source_info.jsonl keyed by source_id."""
    return {s["source_id"]: s for s in read_jsonl(str(RAGTRUTH_RAW / "source_info.jsonl"))}


def load_responses() -> list[dict]:
    return read_jsonl(str(RAGTRUTH_RAW / "response.jsonl"))


def full_lengths(tok, rows: list[dict], batch: int = 512) -> dict[str, np.ndarray]:
    """Untruncated encoded length per row, tokenised exactly as Collator.encode_with_response_budget does.

    total = special tokens (pair) + first segment + response, where
        first segment = tokenizer(INPUT_FORMAT.format(query=q, context=c), add_special_tokens=False)
        response      = tokenizer(answer, add_special_tokens=False)
    Also returns the query-part and response lengths (the response is cut at resp_max by the encoder).
    """
    n_special = tok.num_special_tokens_to_add(pair=True)
    first_texts = [INPUT_FORMAT.format(query=r["query"], context=r["context"]) for r in rows]
    query_texts = [INPUT_FORMAT.format(query=r["query"], context="").rstrip() for r in rows]
    resp_texts = [r["answer"] for r in rows]

    def count(texts):
        out = []
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], add_special_tokens=False)["input_ids"]
            out.extend(len(x) for x in enc)
        return np.array(out, dtype=np.int64)

    first, query, resp = count(first_texts), count(query_texts), count(resp_texts)
    return {"total": first + resp + n_special, "first": first, "query": query, "response": resp,
            "n_special": n_special}


def self_check_lengths(tok, rows: list[dict], totals: np.ndarray, n: int = 200, seed: int = 0) -> int:
    """Assert full_lengths matches the real encoder with a window too large to truncate. Returns rows checked."""
    big = 10 ** 6
    coll = Collator(tok, big, big - 10)
    idx = np.random.default_rng(seed).choice(len(rows), size=min(n, len(rows)), replace=False)
    for i in idx:
        r = rows[i]
        got = len(coll.encode_with_response_budget(r["query"], r["context"], r["answer"]))
        if got != totals[i]:
            raise AssertionError(f"row {i}: full_lengths={totals[i]}, encoder={got}")
    return len(idx)


def auroc(y: np.ndarray, x: np.ndarray) -> float:
    """roc_auc_score, or NaN when only one class is present."""
    return float(roc_auc_score(y, x)) if 0 < y.sum() < len(y) else float("nan")


def group_bootstrap_ci(y: np.ndarray, x: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    """95% percentile CI for AUROC, resampling whole groups with replacement (N_BOOT resamples, seed BOOT_SEED)."""
    idx_by_group = defaultdict(list)
    for i, g in enumerate(groups):
        idx_by_group[g].append(i)
    blocks = [np.array(v) for v in idx_by_group.values()]
    rng = np.random.default_rng(BOOT_SEED)
    vals = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(blocks), size=len(blocks))
        ii = np.concatenate([blocks[k] for k in pick])
        a = auroc(y[ii], x[ii])
        if not np.isnan(a):
            vals.append(a)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def stratified_auroc(y: np.ndarray, x: np.ndarray, strata: np.ndarray) -> tuple[float, np.ndarray]:
    """AUROC of x after removing between-stratum differences.

    Returns (weighted mean of within-stratum AUROCs, weights = positives x negatives per stratum;
             within-stratum percentile rank of x, for a pooled AUROC).
    """
    num = den = 0.0
    rank = np.empty(len(x))
    for s in set(strata):
        m = strata == s
        npos, nneg = y[m].sum(), (1 - y[m]).sum()
        if npos and nneg:
            num += auroc(y[m], x[m]) * npos * nneg
            den += npos * nneg
        rank[m] = stats.rankdata(x[m]) / m.sum()
    return num / den, rank


def md_table(headers: list[str], rows: list[list]) -> str:
    def fmt(x):
        if isinstance(x, float):  # callers pass pre-formatted strings when they need other precision
            return f"{x:.1f}"
        return str(x)
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(fmt(x) for x in r) + " |" for r in rows]
    return "\n".join(lines)


def write_json(obj, name: str) -> Path:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = OUT_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=_np_default)
    return path


def _np_default(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))
