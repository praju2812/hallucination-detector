"""Phase 4 (RQ1): score one G checkpoint on clinical val + test_full (plumbing).

Clean test is NOT scored here: scripts/report_rq1.py derives it from test_full by row key.

Smoke (timing + parity with Phase 2, never reported):
    python -m scripts.score_rq1 --run-dir runs/<run_seedS_timestamp> --limit 100 --parity 100 --out-root results/rq1_smoke
Full run, one seed:
    python -m scripts.score_rq1 --run-dir runs/<run_seedS_timestamp> --parity 100
(<run_seedS_timestamp> = a G run folder trained with model.truncation: fit_first.)

Writes <out-root>/<run folder name>/predictions.jsonl and scoring.json.
Parity: re-scores the first N RAGTruth rows of the run's own val file (data/unified_v2/val.jsonl) and compares
them with Phase 2's predictions.jsonl. Gate: max |dprob| < 0.01, AUROC equal to 3 decimals,
threshold crossings <= 1% of rows. A failed gate exits with code 2 after writing scoring.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
import transformers
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.data import HallucinationDataset
from src.encoding import Collator, build_tokenizer, truncation_mode
from src.rq1 import assign_row_keys, load_g_checkpoint
from src.train import evaluate, get_device, get_device_name, get_git_dirty, get_git_sha, prediction_records
from utils.io import read_jsonl, write_jsonl

CLINICAL_SPLITS = {"val": "val.jsonl", "test_full": "test_full.jsonl"}
EXTRA_FIELDS = ("difficulty", "category")   # MedHallu breakdowns; None for RAGBench rows

PARITY_MAX_ABS = 0.01
PARITY_AUROC_DECIMALS = 3
PARITY_MAX_CROSS_FRAC = 0.01


def sha1_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def make_eval_loader(path: str, cfg: dict, tokenizer, batch_size: int, keep: list[int] | None = None) -> DataLoader:
    """Unshuffled loader over `path` (file order kept, so prediction_records can line rows up with probs).
    keep: row indices to score (in this order); None = every row. num_workers=0: no fork, no pin_memory on MPS."""
    ds = HallucinationDataset(path, cfg["data"])
    if keep is not None:
        ds.rows = [ds.rows[i] for i in keep]
    # the run's own truncation mode; a snapshot without model.truncation (pre-20 Nov legacy run) raises
    collate = Collator(tokenizer, cfg["model"]["max_length"], cfg["model"]["resp_max"], truncation_mode(cfg["model"]))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)


@torch.no_grad()
def probe_precision(model, loader, device) -> dict:
    """dtype of the weights and of the logits on one real batch; scoring never enables autocast."""
    batch = next(iter(loader))
    enc = {k: v.to(device) for k, v in batch["enc"].items() if k != "labels"}
    logits = model(**enc)
    return {
        "param_dtype": str(next(model.parameters()).dtype),
        "logits_dtype": str(logits.dtype),
        "autocast": False,
    }


def score_loader(model, loader, split, cfg, device) -> tuple[list[dict], float]:
    """Phase 2's evaluate() + prediction_records(): one record per row, labels checked against the file.
    Note: prediction_records' idx is the position among the SCORED rows (= file position unless keep is used)."""
    t0 = time.perf_counter()
    labels, probs, _ = evaluate(model, loader, device)
    secs = time.perf_counter() - t0
    recs = prediction_records(split, loader.dataset.rows, labels, probs, cfg["data"])
    return recs, secs


def parity_report(phase2_probs, new_probs, labels, threshold: float) -> dict:
    """Compare re-scored probs with Phase 2's stored probs on the same rows."""
    a = np.asarray(phase2_probs, dtype=float)
    b = np.asarray(new_probs, dtype=float)
    y = np.asarray(labels, dtype=int)
    if not (len(a) == len(b) == len(y)) or len(a) == 0:
        raise ValueError(f"parity arrays: {len(a)} phase2, {len(b)} new, {len(y)} labels")
    diff = np.abs(a - b)
    both_classes = len(np.unique(y)) == 2
    auc_a = float(roc_auc_score(y, a)) if both_classes else None
    auc_b = float(roc_auc_score(y, b)) if both_classes else None
    crossings = int(np.sum((a >= threshold) != (b >= threshold)))
    checks = {
        "max_abs_diff_lt_0.01": bool(diff.max() < PARITY_MAX_ABS),
        "auroc_equal_3dp": both_classes and round(auc_a, PARITY_AUROC_DECIMALS) == round(auc_b, PARITY_AUROC_DECIMALS),
        "crossings_le_1pct": crossings <= PARITY_MAX_CROSS_FRAC * len(a),
    }
    return {
        "n": int(len(a)),
        "n_pos": int(y.sum()),
        "threshold": float(threshold),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "auroc_phase2": auc_a,
        "auroc_new": auc_b,
        "crossings": crossings,
        "crossing_frac": crossings / len(a),
        "checks": checks,
        "passed": all(checks.values()),
    }


def run_parity(model, run_dir, snapshot, n, tokenizer, batch_size, device) -> dict:
    cfg = snapshot["config"]
    val_path = cfg["data"]["val_path"]
    rows = read_jsonl(val_path)
    src_col = cfg["data"]["dataset_col"]
    idxs = [i for i, r in enumerate(rows) if r[src_col] == "ragtruth"][:n]
    phase2 = {p["idx"]: p for p in read_jsonl(os.path.join(run_dir, "predictions.jsonl")) if p["split"] == "val"}
    missing = [i for i in idxs if i not in phase2]
    if missing:
        raise ValueError(f"Phase 2 predictions.jsonl has no val idx {missing[:5]}...")
    loader = make_eval_loader(val_path, cfg, tokenizer, batch_size, keep=idxs)
    recs, _ = score_loader(model, loader, "val", cfg, device)
    for i, r in zip(idxs, recs):
        if phase2[i]["label"] != r["label"]:
            raise ValueError(f"val idx {i}: Phase 2 label {phase2[i]['label']} != file label {r['label']}")
    with open(os.path.join(run_dir, "metrics.json")) as f:
        thr = json.load(f)["threshold"]
    rep = parity_report([phase2[i]["prob"] for i in idxs], [r["prob"] for r in recs],
                        [r["label"] for r in recs], thr)
    rep["val_path"] = val_path
    rep["val_sha1"] = sha1_file(val_path)
    return rep


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="Phase 2 G run folder (best.pt, config_snapshot.json, metrics.json)")
    ap.add_argument("--data-dir", default="data/clinical")
    ap.add_argument("--out-root", default="results/rq1")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N rows of each split (smoke)")
    ap.add_argument("--parity", type=int, default=0, help="re-score N RAGTruth val rows and compare with Phase 2")
    ap.add_argument("--batch-size", type=int, default=None, help="default: the run's train.batch_size")
    ap.add_argument("--device", default=None, help="default: cuda > mps > cpu")
    ap.add_argument("--overwrite", action="store_true", help="replace existing predictions for this run")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    run_dir = os.path.normpath(args.run_dir)
    run_name = os.path.basename(run_dir)
    out_dir = os.path.join(args.out_root, run_name)
    pred_path = os.path.join(out_dir, "predictions.jsonl")
    if os.path.exists(pred_path) and not args.overwrite:
        print(f"{pred_path} exists; pass --overwrite to replace it", file=sys.stderr)
        return 1
    if args.limit is None and os.path.normpath(args.out_root).endswith("rq1_smoke"):
        print("note: full scoring into the smoke folder", file=sys.stderr)

    device = args.device or get_device()
    ckpt = os.path.join(run_dir, "best.pt")
    print(f"run {run_name}  device {device} ({get_device_name()})  git {get_git_sha()} dirty={get_git_dirty()}")

    t_load = time.perf_counter()
    model, snapshot = load_g_checkpoint(run_dir, device)
    cfg = snapshot["config"]
    tokenizer = build_tokenizer(cfg["model"]["name"])
    load_secs = time.perf_counter() - t_load
    batch_size = args.batch_size or cfg["train"]["batch_size"]

    data_files, all_recs, timing, n_full = {}, [], {}, 0
    precision = None
    for split, fname in CLINICAL_SPLITS.items():
        path = os.path.join(args.data_dir, fname)
        all_rows = read_jsonl(path)
        n_full += len(all_rows)
        keys = assign_row_keys(all_rows)                  # keys from the WHOLE file, then cut (occurrences stay right)
        keep = list(range(min(args.limit, len(all_rows)))) if args.limit is not None else None
        loader = make_eval_loader(path, cfg, tokenizer, batch_size, keep=keep)
        rows = loader.dataset.rows
        keys = keys[:len(rows)]
        data_files[split] = {"path": path, "sha1": sha1_file(path), "rows_in_file": len(all_rows)}
        if precision is None:
            precision = probe_precision(model, loader, device)
        recs, secs = score_loader(model, loader, split, cfg, device)
        for rec, row, key in zip(recs, rows, keys):
            rec["row_key"] = key
            for f in EXTRA_FIELDS:
                rec[f] = row.get(f)
        all_recs.extend(recs)
        timing[split] = {"rows": len(rows), "seconds": round(secs, 2), "rows_per_s": round(len(rows) / secs, 3)}
        print(f"  {split}: {len(rows)} rows in {secs:.1f}s ({len(rows) / secs:.2f} rows/s)")

    scored = sum(t["rows"] for t in timing.values())
    secs = sum(t["seconds"] for t in timing.values())
    rps = scored / secs if secs else float("nan")
    eta_one = n_full / rps
    print(f"  {rps:.2f} rows/s -> full run ETA: {eta_one / 60:.1f} min per seed, {3 * eta_one / 60:.1f} min for 3 seeds "
          f"({n_full} rows per seed)")

    parity = None
    if args.parity:
        parity = run_parity(model, run_dir, snapshot, args.parity, tokenizer, batch_size, device)
        print(f"  parity on {parity['n']} RAGTruth val rows: max|d|={parity['max_abs_diff']:.2e} "
              f"mean|d|={parity['mean_abs_diff']:.2e} AUROC phase2={parity['auroc_phase2']} new={parity['auroc_new']} "
              f"crossings={parity['crossings']} -> {'PASS' if parity['passed'] else 'FAIL'} {parity['checks']}")

    write_jsonl(all_recs, pred_path)
    scoring = {
        "run": run_name,
        "seed": snapshot["seed"],
        "experiment": snapshot["experiment"],
        "truncation": truncation_mode(cfg["model"]),   # from the run snapshot (config.model.truncation)
        "checkpoint": os.path.abspath(ckpt),
        "checkpoint_sha1": sha1_file(ckpt),
        "train_git_sha": snapshot.get("git_sha"),
        "git_sha": get_git_sha(),
        "git_dirty": get_git_dirty(),
        "device": device,
        "device_name": get_device_name(),
        "precision": precision,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "batch_size": batch_size,
        "limit": args.limit,
        "data_files": data_files,
        "rows_scored": scored,
        "load_seconds": round(load_secs, 2),
        "timing": timing,
        "rows_per_s": round(rps, 3),
        "eta_full_seconds_per_seed": round(eta_one, 1),
        "parity": parity,
        "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(os.path.join(out_dir, "scoring.json"), "w") as f:
        json.dump(scoring, f, indent=2)
    print(f"wrote {pred_path} and scoring.json")
    if parity is not None and not parity["passed"]:
        print("PARITY GATE FAILED: stop and report before any full run", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
