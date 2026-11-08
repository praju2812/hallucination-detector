#!/usr/bin/env python3
"""
progress_check.py: audit repo state against the proposal (read-only).

Nothing in the repo is written or modified. Run from the repo root:

    python scripts/progress_check.py
    python scripts/progress_check.py --data-checks                # + reproduce the 19 Oct data measurements
    python scripts/progress_check.py --data-checks --truncation   # + combined truncation (needs the tokenizer)

Status legend:
    PASS        matches the proposal
    DIFFERS     does not match: find out which number is right before it goes in the dissertation
    NOT YET     expected to be missing today (e.g. the query field)
    MISSING     file/column not found, so the check could not run (reason shown)
    ERROR       the check crashed (message shown); other checks still run
    INFO        printed for you to read, no pass/fail

ASSUMPTIONS (field names I could not confirm without your files; a wrong one shows up as MISSING):
    unified data : columns label (1 = hallucinated), source, task, group_id
    RAGTruth raw : response.jsonl (source_id, labels[list of dicts with implicit_true], quality)
                   source_info.jsonl (source_id, task_type in {QA, Summary}, source)
    RAGBench     : question, documents (list), response, adherence_score (True = supported)
    MedHallu     : Question, Knowledge, Ground Truth, Hallucinated Answer
    PubMedQA     : question, long_answer
    Raw datasets are read from --raw-dir first (parquet/jsonl/csv), then from the Hugging Face cache.
"""
import argparse
import glob
import importlib
import inspect
import json
import os
import re
import sys
from itertools import combinations

import pandas as pd

# ------------------------------------------------------------------ proposal values
EXPECTED_SPLITS = {"train": 25008, "val": 2784, "test": 3800}
EXPECTED_RAGTRUTH_HALLU = 3410
EXPECTED_IMPLICIT_ONLY = 501
EXPECTED_RAGTRUTH_RATES = {"QA": 29.1, "CNN/DM": 30.9, "Recent News": 27.6}  # percent
EXPECTED_RAGBENCH = {
    "pubmedqa": {"sizes": {"train": 19600, "validation": 2450, "test": 2450},
                 "rates": {"train": 24.5, "validation": 30.0, "test": 29.6}},
    "covidqa": {"sizes": {"train": 1252, "validation": 267, "test": 246},
                "rate_overall": 14.7, "test_positives": 39},
}
EXPECTED_TRUNC_PUBMEDQA = {"train": 0.58, "validation": 0.41, "test": 0.37}  # percent
RESP_MAX, MAX_LEN, N_SPECIAL = 384, 1024, 3  # detector budget as measured on 19 Oct

SPLIT_ALIASES = {"train": {"train"}, "val": {"val", "valid", "validation", "dev"}, "test": {"test"}}
TABLE_EXTS = (".jsonl", ".json", ".csv", ".parquet")

# ------------------------------------------------------------------ result bookkeeping
RESULTS = []


def record(section, check, expected, got, status=None, tol=0.1):
    """Store one result. Numbers compare exactly for ints, within tol for floats."""
    if status is None:
        if got is None:
            status = "MISSING"
        elif (isinstance(expected, (int, float)) and isinstance(got, (int, float))
              and not isinstance(expected, bool) and not isinstance(got, bool)):
            use_tol = tol if (isinstance(expected, float) or isinstance(got, float)) else 0
            status = "PASS" if abs(expected - got) <= use_tol else "DIFFERS"
        else:
            status = "PASS" if expected == got else "DIFFERS"
    RESULTS.append((section, check, expected, got, status))


def guarded(section, fn, *args):
    """Run a check; if it crashes, record ERROR and carry on with the others."""
    try:
        return fn(*args)
    except Exception as e:  # noqa: BLE001 (audit script: never stop on one failure)
        record(section, fn.__name__, "runs cleanly", f"{type(e).__name__}: {e}", status="ERROR")
        return None


def print_summary():
    print("\n" + "=" * 100 + "\nSUMMARY\n" + "=" * 100)
    w = max((len(r[1]) for r in RESULTS), default=10)
    for section, check, expected, got, status in RESULTS:
        print(f"[{status:<10}] {section:<14} {check:<{w}}  expected={expected!s:<22} got={got!s}")
    counts = {}
    for r in RESULTS:
        counts[r[4]] = counts.get(r[4], 0) + 1
    print("\n" + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))


# ------------------------------------------------------------------ generic loading helpers
def files_under(root, exts=TABLE_EXTS):
    return sorted(f for f in glob.glob(os.path.join(root, "**", "*"), recursive=True)
                  if os.path.isfile(f) and f.lower().endswith(exts))


def name_tokens(path):
    return {t for t in re.split(r"[^a-z0-9]+", path.lower()) if t}


def read_table(path):
    p = path.lower()
    if p.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    if p.endswith(".parquet"):
        return pd.read_parquet(path)
    if p.endswith(".csv"):
        return pd.read_csv(path)
    if p.endswith(".json"):
        try:
            with open(path, encoding="utf-8") as f:
                obj = json.load(f)
        except json.JSONDecodeError:  # JSON lines saved as .json (e.g. HaluEval)
            return pd.read_json(path, lines=True)
        return pd.DataFrame(obj)
    raise ValueError(f"unsupported file type: {path}")


def norm(text):
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def as_text(x):
    """Knowledge/documents may be a string or a list/array of passages."""
    if isinstance(x, str):
        return x
    try:
        return " ".join(str(p) for p in list(x))
    except TypeError:
        return str(x)


def as_passages(x):
    if isinstance(x, str):
        return [x]
    try:
        return [str(p) for p in list(x)]
    except TypeError:
        return [str(x)]


def src_kind(s):
    s = str(s).lower()
    if "ragtruth" in s:
        return "ragtruth"
    if "halueval" in s:
        return "halueval"
    return s


def flatten(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten(v, f"{prefix}.{k}" if prefix else str(k))
    else:
        yield prefix, obj


def keys_matching(flat, needles):
    return {k: v for k, v in flat if any(n in k.lower() for n in needles)}


# ------------------------------------------------------------------ 1. unified general data
def load_unified(data_dir):
    found = {}
    all_files = files_under(data_dir)
    for split, aliases in SPLIT_ALIASES.items():
        cands = [f for f in all_files if name_tokens(os.path.basename(f)) & aliases]
        if not cands:
            continue
        cands.sort(key=lambda f: (f.count(os.sep), len(f)))
        if len(cands) > 1:
            print(f"  [warn] several {split} files, using {cands[0]}: {cands}")
        found[split] = (cands[0], read_table(cands[0]))
    return found


def check_unified_general(data_dir):
    """Proposal 3.2: split sizes, leakage, prevalence."""
    sec = "3.2 general"
    print(f"\n--- Unified general data: {data_dir}")
    if not os.path.isdir(data_dir):
        record(sec, "unified data dir", data_dir, None, status="MISSING")
        return None
    splits = load_unified(data_dir)
    for split in EXPECTED_SPLITS:
        if split in splits:
            path, df = splits[split]
            print(f"  {split}: {path}  ({len(df)} rows; columns: {list(df.columns)})")
            record(sec, f"{split} rows", EXPECTED_SPLITS[split], len(df))
        else:
            record(sec, f"{split} rows", EXPECTED_SPLITS[split], None, status="MISSING")
    if not splits:
        return None

    all_ = pd.concat([df.assign(_split=s) for s, (_, df) in splits.items()], ignore_index=True)
    missing_cols = {"label", "source", "group_id"} - set(all_.columns)
    if missing_cols:
        record(sec, "required columns", "label, source, group_id", f"missing {sorted(missing_cols)}",
               status="MISSING")
        return splits
    all_["_src"] = all_["source"].map(src_kind)
    all_["label"] = all_["label"].astype(int)

    print("\n  Rows per source x split:")
    print("  " + pd.crosstab(all_["_src"], all_["_split"]).to_string().replace("\n", "\n  "))

    # Leakage: key on (source, group_id) because RAGTruth and HaluEval were split separately,
    # so the same raw id in two sources is not the same context.
    keys = {s: set(zip(g["_src"], g["group_id"].astype(str))) for s, g in all_.groupby("_split")}
    for a, b in combinations(sorted(keys), 2):
        record(sec, f"group leakage {a}/{b}", 0, len(keys[a] & keys[b]))
    shared_raw = all_.groupby(all_["group_id"].astype(str))["_src"].nunique()
    n_cross = int((shared_raw > 1).sum())
    record(sec, "group_ids reused across sources", "INFO", n_cross, status="INFO")
    if n_cross:
        print(f"  note: {n_cross} group_id values occur in both sources. Harmless for leakage (sources were "
              "split separately) but namespacing them (e.g. 'ragtruth:123') would be safer.")

    record(sec, "label values", [0, 1], sorted(int(v) for v in all_["label"].unique()))

    print("\n  Prevalence (% hallucinated) per source x split:")
    prev = all_.pivot_table(index="_src", columns="_split", values="label", aggfunc="mean") * 100
    print("  " + prev.round(1).to_string().replace("\n", "\n  "))

    rt = all_[all_["_src"] == "ragtruth"]
    record(sec, "RAGTruth hallucinated (all splits)", EXPECTED_RAGTRUTH_HALLU, int(rt["label"].sum()))

    if "task" in rt.columns:
        print("\n  RAGTruth prevalence per task (all splits):")
        for task, g in rt.groupby("task"):
            rate = round(g["label"].mean() * 100, 1)
            print(f"    {task}: {rate}%  (n={len(g)})")
            if str(task).lower() == "qa":
                record(sec, "RAGTruth QA rate %", EXPECTED_RAGTRUTH_RATES["QA"], rate)
        # CNN/DM vs Recent News needs a column naming the summary source; look for one.
        sub_col = next((c for c in rt.columns if rt[c].dtype == object and c not in ("context", "answer", "query")
                        and rt[c].astype(str).str.contains("cnn", case=False).any()), None)
        if sub_col:
            for name, g in rt.groupby(sub_col):
                rate = round(g["label"].mean() * 100, 1)
                if name in EXPECTED_RAGTRUTH_RATES:
                    record(sec, f"RAGTruth {name} rate %", EXPECTED_RAGTRUTH_RATES[name], rate)
        else:
            record(sec, "RAGTruth CNN/DM vs Recent News", "29.1/30.9/27.6",
                   "no summary-source column in unified data; see --data-checks (raw)", status="INFO")
    else:
        record(sec, "task column", "present", None, status="MISSING")
    return splits


# ------------------------------------------------------------------ 2. query field
def check_query_field(splits, repo):
    """Proposal 3.3: is the question in the input yet?"""
    sec = "3.3 query"
    print("\n--- Query field")
    if splits and "train" in splits:
        df = splits["train"][1]
        has = "query" in df.columns
        record(sec, "'query' column in unified train", True, has, status="PASS" if has else "NOT YET")
        if has:
            filled = df["query"].fillna("").astype(str).str.strip().ne("")
            for src, frac in filled.groupby(df["source"].map(src_kind)).mean().items():
                record(sec, f"query non-empty ({src})", 1.0, round(float(frac), 3), tol=0.0)

    for rel in ("scripts/build_dataset.py", "src/encoding.py"):
        path = os.path.join(repo, rel)
        if not os.path.isfile(path):
            record(sec, f"'query' in {rel}", "yes", None, status="MISSING")
            continue
        with open(path, encoding="utf-8") as f:
            hits = [(i + 1, line.rstrip()) for i, line in enumerate(f) if re.search(r"\bquery\b", line)]
        record(sec, f"'query' in {rel}", "yes", "yes" if hits else "no",
               status="PASS" if hits else "NOT YET")
        for ln, text in hits[:5]:
            print(f"  {rel}:{ln}: {text.strip()}")

    if repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        mod = importlib.import_module("src.encoding")
        fn = getattr(mod, "encode_with_response_budget")
        params = list(inspect.signature(fn).parameters)
        print(f"  encode_with_response_budget{inspect.signature(fn)}")
        has_q = any(p in ("query", "question") for p in params)
        record(sec, "encoder takes a query argument", True, has_q, status="PASS" if has_q else "NOT YET")
    except Exception as e:  # noqa: BLE001
        record(sec, "encoder signature", "importable", f"{type(e).__name__}: {e}", status="MISSING")


# ------------------------------------------------------------------ 3. training runs
def check_runs(checkpoint_dirs):
    """Proposal 3.10: what training runs exist?"""
    sec = "3.10 runs"
    print("\n--- Training runs")
    run_dirs = sorted({p for d in checkpoint_dirs if os.path.isdir(d)
                       for p in glob.glob(os.path.join(d, "**", "run_seed*"), recursive=True) if os.path.isdir(p)})
    if not run_dirs:
        record(sec, "run_seed* folders", "any", 0, status="INFO")
        print(f"  none under {checkpoint_dirs}. Kaggle runs only appear here if you downloaded their outputs.")

    n_test = 0
    for rd in run_dirs:
        m = re.search(r"run_seed(\d+)_?(\d{8}_\d{6})?", os.path.basename(rd))
        seed, stamp = (m.group(1), m.group(2)) if m else ("?", "?")
        snap = _load_json(os.path.join(rd, "config_snapshot.json"))
        metrics = _load_json(os.path.join(rd, "metrics.json"))
        flat_snap = list(flatten(snap)) if isinstance(snap, dict) else []
        sha = keys_matching(flat_snap, ("sha", "git"))
        data = {k: v for k, v in keys_matching(flat_snap, ("data",)).items() if isinstance(v, str)}
        fmt = keys_matching(flat_snap, ("query", "input_format"))
        print(f"\n  {rd}\n    seed={seed} stamp={stamp} git={sha or 'NOT RECORDED'}")
        print(f"    data={data or 'NOT RECORDED'}  input_format={fmt or 'NOT RECORDED (assume evidence-only)'}")
        if isinstance(metrics, dict):
            flat_m = list(flatten(metrics))
            test_keys = keys_matching(flat_m, ("test",))
            prauc = {k: v for k, v in flat_m if "pr_auc" in k.lower()}
            if test_keys or prauc:
                n_test += 1
            print(f"    metrics keys: {sorted({k.split('.')[0] for k, _ in flat_m})}")
            for k, v in list(prauc.items())[:8]:
                print(f"      {k} = {v}")
        else:
            print("    metrics.json: NOT FOUND")
    record(sec, "runs with test metrics", "none locked (proposal)", f"{n_test} of {len(run_dirs)}", status="INFO")

    stale = sorted({p for d in checkpoint_dirs + ["."] if os.path.isdir(d)
                    for p in glob.glob(os.path.join(d, "**", "test_results"), recursive=True) if os.path.isdir(p)})
    for p in stale:
        print(f"  note: {p} exists. In September it held stale broken numbers; check it before trusting it.")


def _load_json(path):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ 4. raw data checks (optional)
def find_file(root, filename):
    hits = sorted(glob.glob(os.path.join(root, "**", filename), recursive=True))
    return hits[0] if hits else None


def load_splits(raw_dir, hints, exclude, hf_name, hf_config):
    """Local files whose path contains all hints (and none of exclude), else the HF cache."""
    local = [f for f in files_under(raw_dir, (".parquet", ".jsonl", ".csv"))
             if all(h in f.lower() for h in hints) and not any(x in f.lower() for x in exclude)]
    if local:
        parts = {}
        for f in local:
            toks = name_tokens(os.path.basename(f))
            split = ("validation" if toks & {"validation", "val", "dev"} else
                     "test" if "test" in toks else "train")
            parts.setdefault(split, []).append(read_table(f))
        return {s: pd.concat(dfs, ignore_index=True) for s, dfs in parts.items()}, "local"
    from datasets import load_dataset  # needs the HF cache or network
    ds = load_dataset(hf_name, hf_config)
    return {s: ds[s].to_pandas() for s in ds}, "hf"


def check_ragtruth_raw(raw_dir):
    sec = "3.2 RAGTruth"
    resp, src = find_file(raw_dir, "response.jsonl"), find_file(raw_dir, "source_info.jsonl")
    if not resp or not src:
        record(sec, "raw response/source_info", "found", None, status="MISSING")
        return
    r, s = pd.read_json(resp, lines=True), pd.read_json(src, lines=True)
    m = r.merge(s[["source_id", "task_type", "source"]], on="source_id", how="left")
    m = m[m["task_type"].isin(["QA", "Summary"])].copy()
    m["hallu"] = m["labels"].map(lambda L: isinstance(L, list) and len(L) > 0)
    m["implicit_only"] = m["labels"].map(
        lambda L: isinstance(L, list) and len(L) > 0 and all(bool(x.get("implicit_true")) for x in L))
    print(f"\n--- RAGTruth raw (QA + Summary): {len(m)} responses")
    if "quality" in m.columns:
        print("  by quality:", m["quality"].value_counts().to_dict(),
              "(if the builder filters on quality, the unified counts will differ from these)")
    record(sec, "hallucinated responses (raw)", EXPECTED_RAGTRUTH_HALLU, int(m["hallu"].sum()))
    record(sec, "implicit_true-only (raw)", EXPECTED_IMPLICIT_ONLY, int(m["implicit_only"].sum()))
    qa = m[m["task_type"] == "QA"]
    record(sec, "QA rate % (raw)", EXPECTED_RAGTRUTH_RATES["QA"], round(qa["hallu"].mean() * 100, 1))
    summ = m[m["task_type"] == "Summary"]
    print("  Summary sources found:", summ["source"].value_counts().to_dict())
    for name in ("CNN/DM", "Recent News"):
        g = summ[summ["source"] == name]
        record(sec, f"{name} rate % (raw)", EXPECTED_RAGTRUTH_RATES[name],
               round(g["hallu"].mean() * 100, 1) if len(g) else None)


def ragbench_label(df):
    for col in ("adherence_score", "overall_supported"):
        if col in df.columns:
            return ~df[col].astype(bool), col
    raise KeyError(f"no support column in {list(df.columns)}")


def check_ragbench(raw_dir):
    sec = "3.2 RAGBench"
    out = {}
    for cfg, exp in EXPECTED_RAGBENCH.items():
        splits, origin = load_splits(raw_dir, ("ragbench", cfg), (), "rungalileo/ragbench", cfg)
        print(f"\n--- RAGBench {cfg} ({origin}): splits {sorted(splits)}")
        out[cfg] = splits
        pos_total = n_total = 0
        for split, n_exp in exp["sizes"].items():
            df = splits.get(split)
            record(sec, f"{cfg} {split} rows", n_exp, None if df is None else len(df))
            if df is None:
                continue
            hallu, col = ragbench_label(df)
            rate = round(hallu.mean() * 100, 1)
            print(f"  {split}: {rate}% hallucinated (label from '{col}'), positives={int(hallu.sum())}")
            pos_total, n_total = pos_total + int(hallu.sum()), n_total + len(df)
            if "rates" in exp:
                record(sec, f"{cfg} {split} rate %", exp["rates"][split], rate)
            if cfg == "covidqa" and split == "test":
                record(sec, "covidqa test positives", exp["test_positives"], int(hallu.sum()))
        if cfg == "covidqa" and n_total:
            record(sec, "covidqa overall rate %", exp["rate_overall"], round(100 * pos_total / n_total, 1))
    return out


def check_medhallu(raw_dir, ragbench):
    sec = "3.2 MedHallu"
    mh_lab, o1 = load_splits(raw_dir, ("medhallu", "pqa_labeled"), (), "UTAustin-AIHealth/MedHallu", "pqa_labeled")
    mh_art, o2 = load_splits(raw_dir, ("medhallu", "pqa_artificial"), (), "UTAustin-AIHealth/MedHallu",
                             "pqa_artificial")
    pq, o3 = load_splits(raw_dir, ("pubmedqa", "pqa_labeled"), ("medhallu", "ragbench"), "qiaojin/PubMedQA",
                         "pqa_labeled")
    lab, art, pqd = (pd.concat(d.values(), ignore_index=True) for d in (mh_lab, mh_art, pq))
    print(f"\n--- MedHallu ({o1}/{o2}): pqa_labeled={len(lab)}, pqa_artificial={len(art)}; PubMedQA ({o3})={len(pqd)}")
    record(sec, "pqa_labeled rows", 1000, len(lab))
    record(sec, "pqa_artificial rows", 9000, len(art))

    pq_by_q = {norm(q): la for q, la in zip(pqd["question"], pqd["long_answer"])}
    joined = [(k, gt, pq_by_q[norm(q)]) for q, k, gt in zip(lab["Question"], lab["Knowledge"], lab["Ground Truth"])
              if norm(q) in pq_by_q]
    record(sec, "pqa_labeled joined to PubMedQA", 1000, len(joined))
    record(sec, "contexts containing conclusion", 0, sum(norm(la) in norm(as_text(k)) for k, _, la in joined))
    record(sec, "ground truth == conclusion", 1000, sum(norm(gt) == norm(la) for _, gt, la in joined))

    if not ragbench or "pubmedqa" not in ragbench:
        record(sec, "overlap with RAGBench", "measured", None, status="MISSING")
        return lab, art
    rb = ragbench["pubmedqa"]
    mh_q = {norm(q) for q in pd.concat([lab["Question"], art["Question"]])}
    shared_q = set()
    print("  shared questions / shared context passages per (MedHallu config, RAGBench split):")
    max_docs = 0
    for cfg_name, mh in (("pqa_labeled", lab), ("pqa_artificial", art)):
        mh_docs = {norm(p) for k in mh["Knowledge"] for p in as_passages(k)}
        mh_qs = {norm(q) for q in mh["Question"]}
        for split, df in rb.items():
            rb_qs = {norm(q) for q in df["question"]}
            rb_docs = {norm(d) for docs in df["documents"] for d in as_passages(docs)}
            sq, sd = mh_qs & rb_qs, mh_docs & rb_docs
            shared_q |= (mh_q & rb_qs)
            max_docs = max(max_docs, len(sd))
            print(f"    {cfg_name} x {split}: questions={len(sq)}, passages={len(sd)}")
    record(sec, "distinct shared questions", 1, len(shared_q))
    record(sec, "max shared passages per pair", "<= 7", max_docs, status="PASS" if max_docs <= 7 else "DIFFERS")
    print("  note: if passage counts look off, MedHallu's Knowledge and RAGBench's documents may be chunked "
          "differently from the 19 Oct check; compare the per-pair numbers, not just the max.")
    return lab, art


def check_truncation(ragbench, medhallu):
    """Old evidence-only format (no query), to reproduce the 19 Oct numbers."""
    sec = "3.3 truncation"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    budget = MAX_LEN - N_SPECIAL

    def pct_truncated(contexts, responses):
        c = tok(list(contexts), add_special_tokens=False)["input_ids"]
        r = tok(list(responses), add_special_tokens=False)["input_ids"]
        n = sum(len(ci) > budget - min(len(ri), RESP_MAX) for ci, ri in zip(c, r))
        return round(100 * n / max(len(c), 1), 2)

    print("\n--- Combined truncation (evidence-only format, budget as on 19 Oct)")
    for cfg in ("pubmedqa", "covidqa"):
        for split, df in (ragbench or {}).get(cfg, {}).items():
            ctx = ["\n\n".join(as_passages(d)) for d in df["documents"]]
            got = pct_truncated(ctx, df["response"].astype(str))
            exp = EXPECTED_TRUNC_PUBMEDQA.get(split) if cfg == "pubmedqa" else 0.0
            record(sec, f"{cfg} {split} %", exp, got, tol=0.05)
    if medhallu:
        for cfg_name, mh in zip(("pqa_labeled", "pqa_artificial"), medhallu):
            ctx = [as_text(k) for k in mh["Knowledge"]]
            for col in ("Ground Truth", "Hallucinated Answer"):
                record(sec, f"MedHallu {cfg_name} {col} %", 0.0, pct_truncated(ctx, mh[col].astype(str)), tol=0.05)
    print("  note: RAGBench documents are joined with blank lines here. The clinical builder doesn't exist yet, "
          "so re-measure with its real joining rule (and the query) after Task 1.")


def check_data_checks(raw_dir, truncation):
    """Optional: reproduce the 19 Oct measurements with your own code."""
    if not os.path.isdir(raw_dir):
        record("raw", "raw data dir", raw_dir, None, status="MISSING")
    guarded("3.2 RAGTruth", check_ragtruth_raw, raw_dir)
    rb = guarded("3.2 RAGBench", check_ragbench, raw_dir)
    mh = guarded("3.2 MedHallu", check_medhallu, raw_dir, rb)
    if truncation:
        guarded("3.3 truncation", check_truncation, rb, mh)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".", help="repo root")
    ap.add_argument("--data-dir", default="data/unified", help="unified general data (relative to repo)")
    ap.add_argument("--raw-dir", default="data/raw", help="raw datasets (relative to repo)")
    ap.add_argument("--checkpoints", nargs="+", default=["checkpoints"], help="dirs holding run_seed* folders")
    ap.add_argument("--data-checks", action="store_true", help="also reproduce the 19 Oct data measurements")
    ap.add_argument("--truncation", action="store_true", help="with --data-checks: combined truncation")
    a = ap.parse_args()

    repo = os.path.abspath(a.repo)
    at = lambda p: p if os.path.isabs(p) else os.path.join(repo, p)  # noqa: E731

    splits = guarded("3.2 general", check_unified_general, at(a.data_dir))
    guarded("3.3 query", check_query_field, splits, repo)
    guarded("3.10 runs", check_runs, [at(c) for c in a.checkpoints])
    if a.data_checks:
        check_data_checks(at(a.raw_dir), a.truncation)
    print_summary()


if __name__ == "__main__":
    main()