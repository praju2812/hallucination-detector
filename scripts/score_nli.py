"""Phase 6 (M5): score the zero-shot NLI baseline on RAGTruth and the clinical sets.

Run (hypothesis format: response by default, decision 21 Nov):
    python -m scripts.score_nli                                    # or: --hypothesis question_response
Smoke (first 100 rows of each split; never reported):
    python -m scripts.score_nli --limit 100 --out-root results/nli_smoke

Scores four splits once: RAGTruth val and test (data/unified_v2, RAGTruth rows only), clinical val and
test_full (data/clinical). Clean clinical test is not scored: scripts/report_nli.py derives it from
test_full by row key, as in RQ1. Writes <out-root>/<model name>/predictions.jsonl and scoring.json.
All method logic lives in src/nli.py; this file batches pairs through the model and records provenance.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.nli import (
    ATTRIBUTION_PREFIX,
    HYP_CAP,
    MAX_LENGTH,
    MIN_WORDS,
    MODEL_ID,
    STRIDE,
    aggregate_support,
    build_hypotheses,
    cap_hypothesis_ids,
    entailment_index,
    entailment_probs,
    hallucination_score,
    normalize_preambles,
    split_response_sentences,
    window_spans,
)
from src.rq1 import assign_row_keys
from utils.io import read_jsonl, write_jsonl

# split name -> (data dir key, file name, source filter or None)
SPLITS = {
    "ragtruth_val": ("general", "val.jsonl", "ragtruth"),
    "ragtruth_test": ("general", "test.jsonl", "ragtruth"),
    "clinical_val": ("clinical", "val.jsonl", None),
    "clinical_test_full": ("clinical", "test_full.jsonl", None),
}
HYPOTHESIS_FORMATS = {"response": False, "question_response": True}   # value = include_question
EXTRA_FIELDS = ("difficulty", "category")                            # MedHallu breakdowns; None elsewhere
DETERMINISM_ROWS = 20


MODEL_REVISION = "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7"            # pinned HF commit of MODEL_ID (21 Nov)
_SHA = re.compile(r"^[0-9a-f]{40}$")


MODEL_FILES = ("config.json", "model.safetensors")                   # what AutoModel... loads
TOKENIZER_FILES = ("tokenizer_config.json", "tokenizer.json")         # what AutoTokenizer loads (fast)
DETERMINISM_TOL = 1e-4                                                # max |delta prob| between two identical passes


def snapshot_commit(path) -> str | None:
    """<commit> from an HF cache path .../snapshots/<commit>/<file>, or None (not a str, or not a snapshot path).
    The path is not resolved: snapshot files are symlinks into blobs/."""
    if not isinstance(path, str):
        return None
    parent = Path(path).parent
    return parent.name if parent.parent.name == "snapshots" else None


def cached_commit(model_id: str, filenames: tuple[str, ...], revision: str | None) -> str | None:
    """Commit of the cache snapshot holding `filenames` for model_id at `revision` (try_to_load_from_cache: no
    network). None if any file is not cached. Raises RuntimeError if the files sit in different snapshots.
    Neither model.config._commit_hash (None in transformers 5.22.2) nor the tokenizer records the commit."""
    from huggingface_hub import try_to_load_from_cache
    commits = {f: snapshot_commit(try_to_load_from_cache(model_id, f, revision=revision)) for f in filenames}
    if any(c is None for c in commits.values()):
        return None
    if len(set(commits.values())) > 1:
        raise RuntimeError(f"{model_id} files come from different snapshots: {commits}")
    return next(iter(commits.values()))


def resolve_revision(requested: str | None, model_commit: str | None, tokenizer_commit: str | None) -> str:
    """The full commit SHA both the model and the tokenizer were loaded from.

    Raises RuntimeError if either commit is missing or not a 40-hex SHA, if they differ, or if `requested` is a
    full SHA and the loaded commit is a different one. A branch or tag name is allowed and resolves to its commit.
    """
    for what, c in (("model", model_commit), ("tokenizer", tokenizer_commit)):
        if not c or not _SHA.match(c):
            raise RuntimeError(f"cannot determine the {what} revision (got {c!r}); refusing to score")
    if model_commit != tokenizer_commit:
        raise RuntimeError(f"model revision {model_commit} != tokenizer revision {tokenizer_commit}")
    if requested and _SHA.match(requested) and requested != model_commit:
        raise RuntimeError(f"requested revision {requested} but loaded {model_commit}")
    return model_commit


def model_dir_name(model_id: str) -> str:
    return model_id.rstrip("/").split("/")[-1]


def sha1_file(path: str) -> str:
    from scripts.score_rq1 import sha1_file as _sha1   # reuse the RQ1 helper
    return _sha1(path)


# ---- data ------------------------------------------------------------------------------------

def load_split(name: str, general_dir: str, clinical_dir: str) -> dict:
    """Rows of one split, with idx = 0-based line index in the file and RQ1 row keys (from the filtered rows)."""
    dir_key, fname, source = SPLITS[name]
    path = os.path.join(general_dir if dir_key == "general" else clinical_dir, fname)
    all_rows = read_jsonl(path)
    pairs = [(i, r) for i, r in enumerate(all_rows) if source is None or r["source"] == source]
    rows = [r for _, r in pairs]
    return {"name": name, "path": path, "source_filter": source, "rows_in_file": len(all_rows),
            "idx": [i for i, _ in pairs], "rows": rows, "keys": assign_row_keys(rows)}


# ---- pair construction --------------------------------------------------------------------------

class PairBuilder:
    """Builds model inputs [CLS] premise [SEP] hypothesis [SEP] from token ids.

    transformers 5 fast tokenizers no longer expose prepare_for_model, so the layout is written out here and
    verified against the tokenizer's own pair encoding on a probe pair before any scoring (verify()).
    """

    PROBE = ("The patient was given aspirin after the operation.", "Aspirin was given.")

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.cls, self.sep, self.pad = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
        self.n_special = tokenizer.num_special_tokens_to_add(pair=True)
        self.use_token_types = "token_type_ids" in getattr(tokenizer, "model_input_names", [])

    def ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def build(self, premise_ids: list[int], hyp_ids: list[int]) -> tuple[list[int], list[int]]:
        input_ids = [self.cls] + premise_ids + [self.sep] + hyp_ids + [self.sep]
        token_types = [0] * (len(premise_ids) + 2) + [1] * (len(hyp_ids) + 1)
        return input_ids, token_types

    def model_inputs(self, input_ids: list[int], token_types: list[int]) -> dict[str, list[int]]:
        """Every key the model receives for one unpadded pair, exactly as run_pairs feeds it."""
        out = {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}
        if self.use_token_types:
            out["token_type_ids"] = token_types
        return out

    def verify(self) -> dict:
        """Compare EVERY key the tokenizer returns for a probe pair with what run_pairs would feed the model.

        The key sets must be equal (a key only one side has is a mismatch, e.g. token_type_ids returned by the
        tokenizer but left out of model_input_names), and each shared key must match value for value.
        """
        a, b = self.PROBE
        ref = {k: list(v) for k, v in self.tok(a, b).items()}
        mine = self.model_inputs(*self.build(self.ids(a), self.ids(b)))
        problems = []
        if self.n_special != 3:
            problems.append(f"{self.n_special} special tokens for a pair, layout assumes 3")
        if set(ref) != set(mine):
            problems.append(f"keys differ: tokenizer {sorted(ref)}, built {sorted(mine)}")
        for k in sorted(set(ref) & set(mine)):
            if ref[k] != mine[k]:
                problems.append(f"{k} differs: tokenizer {ref[k]}, built {mine[k]}")
        if problems:
            raise RuntimeError("pair layout check failed: " + "; ".join(problems))
        return {"layout": "[CLS] premise [SEP] hypothesis [SEP]", "n_special": self.n_special,
                "keys_compared": sorted(mine), "token_type_ids": self.use_token_types, "verified": True}


def prepare_row(row: dict, builder: PairBuilder, include_question: bool, stride: int, max_length: int,
                hyp_cap: int) -> dict:
    """Sentences, capped hypothesis ids and context windows for one row. No model call.

    Sentences = split_response_sentences, then normalize_preambles (n_stripped / n_merged are its counts).
    Pair inputs are not stored (62k pairs x ~400 ids would be ~1 GB of Python ints): they are built per batch
    from "ctx", "hyps" and "spans" (pair_inputs).
    """
    sentences, fell_back = split_response_sentences(row["answer"])
    sentences, n_stripped, n_merged = normalize_preambles(sentences)
    hyps = build_hypotheses(row, sentences, include_question)
    ctx = builder.ids(row["context"])
    q_cache: dict[str, list[int]] = {}
    hyp_ids_list, spans_list, n_capped = [], [], 0
    for question, sentence in hyps:
        q_ids = [] if question is None else q_cache.setdefault(question, builder.ids(question))
        hyp_ids, capped = cap_hypothesis_ids(q_ids, builder.ids(sentence), max_tokens=hyp_cap)
        n_capped += int(capped)
        budget = max_length - len(hyp_ids) - builder.n_special
        hyp_ids_list.append(hyp_ids)
        spans_list.append(window_spans(len(ctx), budget, stride))
    return {"sentences": sentences, "fell_back": fell_back, "n_capped": n_capped, "n_stripped": n_stripped,
            "n_merged": n_merged, "ctx": ctx, "hyps": hyp_ids_list, "spans": spans_list}


def n_pairs(p: dict) -> int:
    return sum(len(s) for s in p["spans"])


def pair_inputs(p: dict, j: int, k: int, builder: PairBuilder) -> tuple[list[int], list[int]]:
    s, e = p["spans"][j][k]
    return builder.build(p["ctx"][s:e], p["hyps"][j])


def pair_length(p: dict, j: int, k: int, builder: PairBuilder) -> int:
    s, e = p["spans"][j][k]
    return (e - s) + len(p["hyps"][j]) + builder.n_special


# ---- model ---------------------------------------------------------------------------------------

def run_pairs(model, lengths: list[int], get_inputs, batch_size: int, device: str, ent_idx: int,
              pad_id: int, use_token_types: bool, progress_every: int = 0) -> tuple[np.ndarray, float]:
    """P(entailment) for items 0..n-1, in item order; get_inputs(i) -> (input_ids, token_type_ids).

    Items are batched longest first (similar lengths share a batch, little padding; an OOM shows up at once).
    """
    import torch

    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    out = np.empty(len(lengths), dtype=np.float64)
    t0 = time.perf_counter()
    with torch.inference_mode():
        for b, start in enumerate(range(0, len(order), batch_size)):
            idx = order[start:start + batch_size]
            width = max(lengths[i] for i in idx)
            ids = torch.full((len(idx), width), pad_id, dtype=torch.long)
            mask = torch.zeros((len(idx), width), dtype=torch.long)
            tts = torch.zeros((len(idx), width), dtype=torch.long)
            for r, i in enumerate(idx):
                input_ids, tt = get_inputs(i)
                n = len(input_ids)
                ids[r, :n] = torch.tensor(input_ids)
                mask[r, :n] = 1
                tts[r, :n] = torch.tensor(tt)
            kwargs = {"input_ids": ids.to(device), "attention_mask": mask.to(device)}
            if use_token_types:
                kwargs["token_type_ids"] = tts.to(device)
            logits = model(**kwargs).logits.float().cpu().numpy()
            out[idx] = entailment_probs(logits, ent_idx)
            if progress_every and (b + 1) % progress_every == 0:
                done = min(start + batch_size, len(order))
                el = time.perf_counter() - t0
                print(f"    {done}/{len(order)} pairs  {done / el:.1f} pairs/s", flush=True)
    return out, time.perf_counter() - t0


def score_prepared(prepared: list[dict], model, builder: PairBuilder, ent_idx: int, batch_size: int, device: str,
                   progress_every: int = 0) -> tuple[list[dict], dict]:
    """Run every pair of the prepared rows through the model and aggregate per row."""
    where = [(r, j, k) for r, p in enumerate(prepared) for j, spans in enumerate(p["spans"]) for k in range(len(spans))]
    lengths = [pair_length(prepared[r], j, k, builder) for r, j, k in where]
    get_inputs = lambda i: pair_inputs(prepared[where[i][0]], where[i][1], where[i][2], builder)
    probs, secs = run_pairs(model, lengths, get_inputs, batch_size, device, ent_idx, builder.pad,
                            builder.use_token_types, progress_every)
    support = [[[None] * len(spans) for spans in p["spans"]] for p in prepared]
    for (r, j, k), pr in zip(where, probs):
        support[r][j][k] = float(pr)
    results = []
    for p, sup in zip(prepared, support):
        row_support = aggregate_support(sup)
        results.append({"prob": hallucination_score(row_support),
                        "sentence_support": [max(w) for w in sup],
                        "n_sentences": len(p["sentences"]), "n_windows": max(len(s) for s in p["spans"]),
                        "fell_back": p["fell_back"], "n_capped": p["n_capped"]})
    return results, {"pairs": len(where), "seconds": round(secs, 2)}


def compare_passes(a: list[float], b: list[float], tol: float = DETERMINISM_TOL) -> dict:
    """Two passes over the same rows agree if max |a - b| <= tol AND no ranking change.

    A ranking change is a pair (i, j) whose first-pass scores differ by more than 2 * tol and whose order is
    reversed in the second pass ((a_i - a_j) * (b_i - b_j) < 0). Swaps between closer pairs are numerical noise
    and ignored. `identical` (exact equality) is recorded for information.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    max_diff = float(np.max(np.abs(a - b))) if len(a) else 0.0
    da, db = a[:, None] - a[None, :], b[:, None] - b[None, :]
    swaps = (np.abs(da) > 2 * tol) & (da * db < 0)
    n_changes = int(np.triu(swaps, k=1).sum())
    return {"n": int(len(a)), "max_abs_diff": max_diff, "tolerance": tol, "within_tolerance": max_diff <= tol,
            "ranking_changes": n_changes, "ranking_unchanged": n_changes == 0,
            "identical": bool(np.array_equal(a, b)), "passed": max_diff <= tol and n_changes == 0}


def determinism_check(prepared: list[dict], model, builder, ent_idx, batch_size, device) -> dict:
    """Score the same rows twice, the same way, and compare the probs (compare_passes)."""
    a, _ = score_prepared(prepared, model, builder, ent_idx, batch_size, device)
    b, _ = score_prepared(prepared, model, builder, ent_idx, batch_size, device)
    return compare_passes([x["prob"] for x in a], [y["prob"] for y in b])


def make_records(split: dict, results: list[dict]) -> list[dict]:
    recs = []
    for i, res in enumerate(results):
        row = split["rows"][i]
        rec = {"split": split["name"], "idx": split["idx"][i], "source": row["source"], "task": row.get("task"),
               "group_id": row["group_id"], "label": int(row["label"]), "prob": res["prob"], "row_key": split["keys"][i]}
        rec.update({f: row.get(f) for f in EXTRA_FIELDS})
        rec.update({k: res[k] for k in ("n_sentences", "n_windows", "fell_back", "n_capped", "sentence_support")})
        recs.append(rec)
    return recs


def split_counts(split: dict, prepared: list[dict]) -> dict:
    out = defaultdict(lambda: {"rows": 0, "sentences": 0, "pairs": 0, "fallback_rows": 0, "capped_hypotheses": 0,
                               "stripped_prefixes": 0, "merged_colon_lines": 0})
    for row, p in zip(split["rows"], prepared):
        c = out[row["source"]]
        c["rows"] += 1
        c["sentences"] += len(p["sentences"])
        c["pairs"] += n_pairs(p)
        c["fallback_rows"] += int(p["fell_back"])
        c["capped_hypotheses"] += p["n_capped"]
        c["stripped_prefixes"] += p["n_stripped"]
        c["merged_colon_lines"] += p["n_merged"]
    return dict(out)


# ---- main ------------------------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hypothesis", default="response", choices=sorted(HYPOTHESIS_FORMATS),
                    help="response only (default, decision 21 Nov), or question + response for QA rows")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--revision", default=MODEL_REVISION,
                    help="HF revision for tokenizer and model (default: the pinned commit)")
    ap.add_argument("--splits", default=",".join(SPLITS))
    ap.add_argument("--general-dir", default="data/unified_v2")
    ap.add_argument("--clinical-dir", default="data/clinical")
    ap.add_argument("--out-root", default="results/nli")
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--max-length", type=int, default=MAX_LENGTH)
    ap.add_argument("--hyp-cap", type=int, default=HYP_CAP)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default=None, help="default: cuda > mps > cpu")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N rows of each split (smoke)")
    ap.add_argument("--progress-every", type=int, default=50, help="print progress every N batches (0 = off)")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from src.train import get_device, get_device_name, get_git_dirty, get_git_sha

    args = parse_args(argv)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    unknown = [s for s in splits if s not in SPLITS]
    if unknown:
        print(f"unknown split(s) {unknown}; choose from {list(SPLITS)}", file=sys.stderr)
        return 1
    out_dir = os.path.join(args.out_root, model_dir_name(args.model))
    pred_path = os.path.join(out_dir, "predictions.jsonl")
    if os.path.exists(pred_path) and not args.overwrite:
        print(f"{pred_path} exists; pass --overwrite to replace it", file=sys.stderr)
        return 1
    include_question = HYPOTHESIS_FORMATS[args.hypothesis]
    device = args.device or get_device()
    print(f"model {args.model}  hypothesis={args.hypothesis}  device {device} ({get_device_name()})  "
          f"git {get_git_sha()} dirty={get_git_dirty()}")

    t_load = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, revision=args.revision, dtype=torch.float32)
    model.to(device).eval()
    load_secs = time.perf_counter() - t_load
    revision = resolve_revision(args.revision, cached_commit(args.model, MODEL_FILES, args.revision),
                                cached_commit(args.model, TOKENIZER_FILES, args.revision))  # raises: nothing scored
    ent_idx = entailment_index(model.config.id2label)
    builder = PairBuilder(tokenizer)
    pair_layout = builder.verify()
    print(f"  loaded in {load_secs:.1f}s  revision {revision}  id2label {model.config.id2label}  entailment={ent_idx}")

    all_recs, data_files, counts, timing = [], {}, {}, {}
    pairs_full = pairs_scored = 0
    first_prepared = None
    for name in splits:
        sp = load_split(name, args.general_dir, args.clinical_dir)
        t_prep = time.perf_counter()
        prepared = [prepare_row(r, builder, include_question, args.stride, args.max_length, args.hyp_cap)
                    for r in sp["rows"]]                       # whole split: full-split counts and the ETA
        prep_secs = time.perf_counter() - t_prep
        counts[name] = split_counts(sp, prepared)
        pairs_full += sum(n_pairs(p) for p in prepared)
        n = len(sp["rows"]) if args.limit is None else min(args.limit, len(sp["rows"]))
        sp_cut = {**sp, "rows": sp["rows"][:n], "idx": sp["idx"][:n], "keys": sp["keys"][:n]}
        if first_prepared is None:
            first_prepared = prepared[:DETERMINISM_ROWS]
        print(f"  {name}: scoring {n} of {len(sp['rows'])} rows "
              f"({sum(n_pairs(p) for p in prepared[:n])} pairs; prep {prep_secs:.1f}s)", flush=True)
        results, st = score_prepared(prepared[:n], model, builder, ent_idx, args.batch_size, device, args.progress_every)
        all_recs.extend(make_records(sp_cut, results))
        pairs_scored += st["pairs"]
        timing[name] = {"rows": n, **st, "pairs_per_s": round(st["pairs"] / st["seconds"], 2) if st["seconds"] else None}
        data_files[name] = {"path": sp["path"], "sha1": sha1_file(sp["path"]), "source_filter": sp["source_filter"],
                            "rows_in_file": sp["rows_in_file"], "rows_expected": len(sp["rows"])}
        print(f"    {st['pairs']} pairs in {st['seconds']:.1f}s", flush=True)

    secs = sum(t["seconds"] for t in timing.values())
    pps = pairs_scored / secs if secs else float("nan")
    eta = pairs_full / pps if pps == pps and pps > 0 else None
    print(f"  {pps:.1f} pairs/s; full run = {pairs_full} pairs -> ETA {eta / 60:.1f} min" if eta else "  no timing")

    det = determinism_check(first_prepared, model, builder, ent_idx, args.batch_size, device)
    print(f"  determinism on {det['n']} rows: max|d|={det['max_abs_diff']:.2e} (tol {det['tolerance']:.0e}), "
          f"ranking unchanged={det['ranking_unchanged']} -> {'PASS' if det['passed'] else 'FAIL'}")

    write_jsonl(all_recs, pred_path)
    param = next(model.parameters())
    scoring = {
        "model": args.model,
        "model_dir": model_dir_name(args.model),
        "revision": revision,
        "revision_requested": args.revision,
        "id2label": {str(k): v for k, v in model.config.id2label.items()},
        "entailment_index": ent_idx,
        "type_vocab_size": getattr(model.config, "type_vocab_size", None),
        "hypothesis": args.hypothesis,
        "include_question": include_question,
        "max_length": args.max_length,
        "stride": args.stride,
        "hyp_cap": args.hyp_cap,
        "min_words": MIN_WORDS,
        "splitter": {"name": "pysbd", "version": importlib.metadata.version("pysbd"), "newline_split_first": True},
        "preambles": {"attribution_prefix": ATTRIBUTION_PREFIX.pattern, "flags": "IGNORECASE",
                      "colon_lines": "merged into the next sentence (the previous one if last)",
                      "counts": "counts[split][source].stripped_prefixes / merged_colon_lines"},
        "pair_layout": pair_layout,
        "aggregation": "score = 1 - min over sentences of max over windows of P(entailment)",
        "device": device,
        "device_name": get_device_name(),
        "precision": {"param_dtype": str(param.dtype), "autocast": False},
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "git_sha": get_git_sha(),
        "git_dirty": get_git_dirty(),
        "batch_size": args.batch_size,
        "limit": args.limit,
        "data_files": data_files,
        "counts": counts,
        "rows_scored": len(all_recs),
        "pairs_scored": pairs_scored,
        "pairs_full": pairs_full,
        "load_seconds": round(load_secs, 2),
        "timing": timing,
        "pairs_per_s": round(pps, 2) if pps == pps else None,
        "eta_full_seconds": round(eta, 1) if eta else None,
        "determinism": det,
        "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(os.path.join(out_dir, "scoring.json"), "w") as f:
        json.dump(scoring, f, indent=2)
    print(f"wrote {pred_path} and scoring.json")
    if not det["passed"]:
        print("DETERMINISM CHECK FAILED: stop and report before using these scores", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
