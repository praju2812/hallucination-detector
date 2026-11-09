"""what the encoder does to each split and source.

For every (split, source): rows, % with context truncated, rows where the context gets 0 tokens,
rows where the response is cut by resp_max (by design), rows where the response is cut BELOW
min(own length, resp_max) (overflow: query + response did not fit), and query length p50/p99 in tokens.

Definitions (all from the real Collator.encode_with_response_budget output):
    ref_first   = tokenizer(INPUT_FORMAT.format(query, context), add_special_tokens=False)
    query_part  = tokenizer(INPUT_FORMAT.format(query, context="").rstrip(), add_special_tokens=False)
                  i.e. the "Question: {query} Context:" head
    ctx truncated      : kept first segment shorter than ref_first
    ctx 0 tokens       : kept first segment no longer than query_part
    resp > resp_max    : response longer than resp_max (cut to resp_max, expected)
    resp overflow cut  : kept response shorter than min(response length, resp_max)
At max_length 1024 the last two "must be 0" columns (ctx 0 tokens, resp overflow cut) are checked.

Run:  python -m scripts.report_truncation --config configs/base.yaml --data-dir data/unified_v2
      ... --per-row results/truncation_flags_v2.jsonl   # also write one flag line per row (Phase 2)

--per-row lines: {split, idx, source, group_id, ctx_truncated}; idx = row position in the split file,
the same definition predictions.jsonl uses, so the two files join on (split, idx).
"""

from __future__ import annotations
import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

sys.path.append(str(Path(__file__).parent.parent))
from src.encoding import INPUT_FORMAT, Collator, build_tokenizer
from utils.io import read_jsonl, write_jsonl

SPLITS = ("train", "val", "test")


def per_row_record(split: str, idx: int, row: dict, stats: dict, cfg_data: dict) -> dict:
    """One line of the --per-row flags file."""
    return {
        "split": split,
        "idx": int(idx),
        "source": row[cfg_data["dataset_col"]],
        "group_id": row.get("group_id"),
        "ctx_truncated": bool(stats["ctx_trunc"]),
    }


def split_output(out: list[int], sep_id: int) -> tuple[list[int], list[int]]:
    """[CLS] first [SEP] resp [SEP] -> (first, resp)."""
    i = out.index(sep_id)
    return out[1:i], out[i + 1:-1]


def row_stats(coll: Collator, tok, q: str, c: str, r: str) -> dict:
    out = coll.encode_with_response_budget(q, c, r)
    first, resp = split_output(out, tok.sep_token_id)
    ids = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    ref_first = ids(INPUT_FORMAT.format(query=q, context=c))
    query_part = ids(INPUT_FORMAT.format(query=q, context="").rstrip())
    n_resp = len(ids(r))
    return {
        "ctx_trunc": len(first) < len(ref_first),
        "ctx_zero": len(first) <= len(query_part),
        "resp_capped": n_resp > coll.resp_max,
        "resp_overflow": len(resp) < min(n_resp, coll.resp_max),
        "q_bare": len(ids(q)),
        "q_head": len(query_part),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--data-dir", default="data/unified_v2")
    ap.add_argument("--limit", type=int, default=None, help="first N rows per split (quick look only, not for the report)")
    ap.add_argument("--per-row", default=None, help="also write per-row truncation flags (jsonl) to this path")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    d, m = cfg["data"], cfg["model"]
    tok = build_tokenizer(m["name"])
    coll = Collator(tok, m["max_length"], m["resp_max"])

    groups = defaultdict(list)
    flags = []
    for sp in SPLITS:
        path = os.path.join(args.data_dir, f"{sp}.jsonl")
        rows = read_jsonl(path)[: args.limit]
        for i, r in enumerate(rows):
            s = row_stats(coll, tok, r[d["query_col"]], r[d["context_col"]], r[d["response_col"]])
            groups[(sp, r[d["dataset_col"]])].append(s)
            if args.per_row:
                flags.append(per_row_record(sp, i, r, s, d))
        print(f"  scored {path}: {len(rows)} rows", file=sys.stderr)

    if args.per_row:
        write_jsonl(flags, args.per_row)
        print(f"  wrote {len(flags)} per-row flags to {args.per_row}", file=sys.stderr)

    print(f"\nconfig {args.config}: max_length {m['max_length']}, resp_max {m['resp_max']}; data {args.data_dir}"
          + (f"; LIMIT {args.limit} rows/split (not for the report)" if args.limit else ""))
    print(f"input format: {INPUT_FORMAT!r} [SEP] response\n")
    print("| split | source | n | ctx truncated | ctx 0 tokens | resp > resp_max | resp overflow cut "
          "| query tok p50 / p99 | query head tok p50 / p99 |")
    print("|---|---|---|---|---|---|---|---|---|")
    must_be_zero = 0
    for (sp, src), st in sorted(groups.items(), key=lambda kv: (SPLITS.index(kv[0][0]), kv[0][1])):
        n = len(st)
        col = lambda k: sum(x[k] for x in st)
        qb = np.percentile([x["q_bare"] for x in st], [50, 99])
        qh = np.percentile([x["q_head"] for x in st], [50, 99])
        must_be_zero += col("ctx_zero") + col("resp_overflow")
        print(f"| {sp} | {src} | {n} | {col('ctx_trunc') / n:.1%} ({col('ctx_trunc')}) | {col('ctx_zero')} "
              f"| {col('resp_capped')} | {col('resp_overflow')} | {qb[0]:.0f} / {qb[1]:.0f} | {qh[0]:.0f} / {qh[1]:.0f} |")

    if m["max_length"] == 1024:
        status = "PASS" if must_be_zero == 0 else "FAIL"
        print(f"\n[{status}] ctx 0 tokens + resp overflow cut = {must_be_zero} (must be 0 at max_length 1024)")


if __name__ == "__main__":
    main()
