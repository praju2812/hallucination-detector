#!/usr/bin/env python3
"""
mvp_check.py: read-only audit of the dissertation MVP. Nothing in the repo is written.

Run from the repo root:
    python scripts/mvp_check.py
    python scripts/mvp_check.py --runs-dir ~/Downloads/kaggle_runs     # also scan downloaded Kaggle outputs
    python scripts/mvp_check.py --no-query                             # if you decide to drop the query field

MVP = what the plan says is never cut:
    M0  query field restored              (proposal commitment; --no-query skips it)
    M1  G locked: >=3 seeds, test metrics, valid config
    M2  clinical dataset built, clean splits
    M3  RQ1: G evaluated on every clinical test source, >=3 seeds
    M4  RQ2: C and G+C trained, >=3 seeds each, tested on general + clinical
    M5  NLI baseline scored on general + clinical test sets

Status: PASS | NOT YET | DIFFERS (needs a look) | MISSING (could not check, reason shown) | INFO

PATH CONVENTIONS for artefacts that do not exist yet. Change them here if you prefer others;
the point is that "done" is defined before the work starts.
"""
import argparse, ast, gzip, hashlib, json, os, re, statistics, subprocess, sys
from pathlib import Path

GENERAL_V1 = Path("data/unified")          # existing; your earlier validation findings used it
GENERAL_V2 = Path("data/unified_v2")       # with the query field
CLINICAL = Path("data/clinical")           # proposed: train/val/test.jsonl, same schema as unified
RQ1_DIR = Path("results/rq1")              # proposed: one json per G seed, {"seed":..,"test":{source:{metrics}}}
NLI_DIR = Path("results/nli")              # proposed: {"test":{source:{metrics}}}
RUN_ROOTS = ["checkpoints", "runs", "outputs"]
SPLITS = ["train", "val", "test"]
SCHEMA = ["query", "context", "answer", "label", "source", "task", "group_id"]
CLINICAL_KEYS = ["covidqa", "medhallu", "pubmedqa"]  # matched as substrings of source names, in this order
MIN_SEEDS = 3
SELF_FILES = {"mvp_check.py", "progress_check.py"}  # never count the audit scripts as evidence
EXPECTED_MAX_LENGTH = 1024

RESULTS = {}
STATE = {}


def say(status, msg):
    print(f"  [{status:<7}] {msg}")


def section(title):
    print(f"\n=== {title} ===")


def clinical_key(source):
    s = str(source).lower()
    return next((k for k in CLINICAL_KEYS if k in s), None)


def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_splits(d):
    out = {}
    for sp in SPLITS:
        for name in (f"{sp}.jsonl", f"{sp}.jsonl.gz"):      # plain or gzip-compressed
            p = d / name
            if p.exists():
                if sp in out:
                    raise ValueError(f"both {sp}.jsonl and {sp}.jsonl.gz in {d}: keep one")
                out[sp] = read_jsonl(p)
    return out


def find_key(obj, key):
    """First value for `key` anywhere in a nested dict/list."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = find_key(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_key(v, key)
            if r is not None:
                return r
    return None


def metric(m, names=("pr_auc", "prauc", "average_precision", "ap")):
    if not isinstance(m, dict):
        return None
    for n in names:
        if n in m:
            return m[n]
    return None


def test_block(metrics):
    """Per-source test metrics. Accepts {"per_source": {src: {...}}} (src/train.py format),
    {"test": {src: {...}}} or {src: {...}} at top level. Never returns the blended "overall"."""
    if not isinstance(metrics, dict):
        return {}
    if isinstance(metrics.get("per_source"), dict):
        return {k: v for k, v in metrics["per_source"].items() if isinstance(v, dict)}
    t = metrics.get("test", metrics)
    if isinstance(t, dict):
        t = {k: v for k, v in t.items() if k != "overall"}
    return {k: v for k, v in t.items() if isinstance(v, dict)} if isinstance(t, dict) else {}


def ctx_hash(text):
    return hashlib.sha1(" ".join(str(text).lower().split()).encode()).hexdigest()


def git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=20).stdout.rstrip()
    except Exception as e:
        return f"ERR {e}"


def grep(pattern, roots=("src", "scripts", "tests")):
    hits = []
    rx = re.compile(pattern)
    for root in roots:
        for p in Path(root).rglob("*.py") if Path(root).exists() else []:
            if p.name in SELF_FILES:
                continue
            try:
                for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
                    if rx.search(line):
                        hits.append(f"{p}:{i}: {line.strip()[:90]}")
            except Exception:
                pass
    return hits


# ---------------------------------------------------------------------------
def check_repo():
    section("Repo state")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    say("INFO", f"branch {branch}, HEAD {git('rev-parse', '--short', 'HEAD')}, last commit {git('log', '-1', '--format=%cs %s')}")
    dirty = [l for l in git("status", "--porcelain").splitlines() if l]
    say("INFO" if not dirty else "DIFFERS", f"{len(dirty)} uncommitted change(s)" + (": " + ", ".join(l[3:] for l in dirty[:6]) if dirty else ""))
    ahead = git("rev-list", "--count", "@{u}..HEAD")
    if ahead.isdigit():
        say("INFO" if ahead == "0" else "DIFFERS", f"{ahead} commit(s) not pushed (Kaggle clones from GitHub)")
    else:
        say("MISSING", "no upstream branch set, so unpushed commits can't be counted")
    cfg = Path("configs/base.yaml")
    if cfg.exists():
        m = re.search(r"max_length:\s*(\d+)", cfg.read_text())
        v = int(m.group(1)) if m else None
        say("PASS" if v == EXPECTED_MAX_LENGTH else "DIFFERS", f"configs/base.yaml max_length = {v}")
    else:
        say("MISSING", "configs/base.yaml not found")


def check_m0(require_query):
    section("M0  Query field")
    if not require_query:
        say("INFO", "skipped (--no-query): state the old input format as a deviation in the dissertation")
        RESULTS["M0"] = "SKIPPED"
        return
    ok = True
    v1, v2 = load_splits(GENERAL_V1), load_splits(GENERAL_V2)
    if not v2:
        say("NOT YET", f"{GENERAL_V2}/ has no split files")
        ok = False
    else:
        for sp in SPLITS:
            rows = v2.get(sp, [])
            if not rows:
                say("MISSING", f"{GENERAL_V2}/{sp}.jsonl")
                ok = False
                continue
            by_src = {}
            for r in rows:
                s = r.get("source", "?")
                n, filled = by_src.get(s, (0, 0))
                by_src[s] = (n + 1, filled + bool(str(r.get("query") or "").strip()))
            parts = ", ".join(f"{s} {f}/{n}" for s, (n, f) in sorted(by_src.items()))
            full = all(f == n for n, f in by_src.values())
            say("PASS" if full else "DIFFERS", f"{sp}: non-empty query per source: {parts}")
            ok &= full
            if sp in v1:
                a, b = v1[sp], rows
                same_len = len(a) == len(b)
                same_cols = same_len and all(
                    x.get(k) == y.get(k) for x, y in zip(a, b) for k in ("label", "group_id", "source", "context", "answer"))
                say("PASS" if same_cols else "DIFFERS",
                    f"{sp}: v2 vs v1 rows {len(b)} vs {len(a)}; label/group_id/source/context/answer identical in order: {same_cols}")
                ok &= same_cols
    enc = Path("src/encoding.py")
    if enc.exists():
        tree = ast.parse(enc.read_text())
        args = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "encode_with_response_budget":
                args = [a.arg for a in node.args.args]
        has_q = bool(args) and any("query" in a or "question" in a for a in args)
        say("PASS" if has_q else "NOT YET", f"encode_with_response_budget args: {args}")
        ok &= has_q
    else:
        say("MISSING", "src/encoding.py not found")
        ok = False
    hard = grep(r"Collator\(\s*\w+\s*,\s*1024\s*,\s*384\s*\)")
    say("DIFFERS" if hard else "PASS", f"hard-coded Collator(tok, 1024, 384) calls: {len(hard)}" + (" -> " + hard[0] if hard else ""))
    tests = grep(r"query", roots=("tests",))
    say("PASS" if tests else "NOT YET", f"tests mentioning query: {len(tests)}")
    ok &= bool(tests)
    RESULTS["M0"] = "PASS" if ok else "NOT YET"


def scan_runs(extra_dirs):
    runs = []
    roots = [Path(r) for r in RUN_ROOTS] + [Path(os.path.expanduser(d)) for d in extra_dirs]
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in (".venv", ".git", "__pycache__")]
            if "config_snapshot.json" in filenames:
                d = Path(dirpath).resolve()
                if d in seen:
                    continue
                seen.add(d)
                snap = json.load(open(d / "config_snapshot.json"))
                mp = d / "metrics.json"
                metrics = json.load(open(mp)) if mp.exists() else None
                runs.append(dict(dir=d, snap=snap, metrics=metrics))
    return runs


def data_paths(snap):
    """data_dir if recorded, else the train/val/test paths src/train.py already snapshots."""
    vals = [find_key(snap, k) for k in ("data_dir", "data_dirs", "train_path", "val_path", "test_path")]
    flat = []
    for v in vals:
        if isinstance(v, list):
            flat += [str(x) for x in v]
        elif v:
            flat.append(str(v))
    return " ".join(flat)


def classify(snap, require_query=True):
    exp = find_key(snap, "experiment")
    if exp:
        return str(exp), False
    dd = find_key(snap, "train_path") or find_key(snap, "data_dir") or find_key(snap, "data_dirs") or ""
    dd = " ".join(map(str, dd)) if isinstance(dd, list) else str(dd)
    gen, clin = "unified" in dd, "clinical" in dd
    if gen and clin:
        return "G+C", True
    if clin:
        return "C", True
    if gen:
        return ("G" if ("unified_v2" in dd or not require_query) else "G-oldformat"), True
    return "?", True


def run_valid(snap, require_query):
    ml = find_key(snap, "max_length")
    dd = data_paths(snap)
    problems = []
    if ml != EXPECTED_MAX_LENGTH:
        problems.append(f"max_length={ml}")
    if require_query and "unified/" in dd and "unified_v2" not in dd:
        problems.append("old input format (data/unified)")
    if "tiny" in dd:
        problems.append("tiny smoke-test data")
    return problems


def summarise(runs, label, sources):
    for src in sources:
        vals = []
        for r in runs:
            tb = test_block(r["metrics"])
            hit = next((v for k, v in tb.items() if (clinical_key(k) == src if src in CLINICAL_KEYS else k.lower() == src)), None)
            pr = metric(hit)
            if pr is not None:
                vals.append(float(pr))
        if vals:
            sd = statistics.stdev(vals) if len(vals) > 1 else float("nan")
            say("INFO", f"{label} test PR-AUC {src}: mean {statistics.mean(vals):.3f} +/- {sd:.3f} (n={len(vals)})")


def check_runs(extra_dirs, require_query):
    section("M1  Tooling and G multi-seed")
    seed_arg = grep(r"""add_argument\(\s*['\"]--seed""")
    say("PASS" if seed_arg else "MISSING", f"--seed CLI flag: {seed_arg[0] if seed_arg else 'not found'}")
    summ = Path("scripts/summarize_runs.py").exists()
    say("PASS" if summ else "MISSING", "scripts/summarize_runs.py " + ("present" if summ else "not found"))
    sha = grep(r"def get_git_sha")
    say("PASS" if sha else "MISSING", f"get_git_sha: {sha[0] if sha else 'not found'}")

    runs = scan_runs(extra_dirs)
    say("INFO", f"{len(runs)} run folder(s) with config_snapshot.json found under {RUN_ROOTS + list(extra_dirs)}")
    groups = {}
    for r in sorted(runs, key=lambda r: r["dir"].name):
        exp, inferred = classify(r["snap"], require_query)
        r["exp"] = exp
        r["problems"] = run_valid(r["snap"], require_query)
        r["has_test"] = bool(test_block(r["metrics"]))
        groups.setdefault(exp, []).append(r)
        say("INFO", f"{r['dir'].name}: exp={exp}{' (inferred from data paths)' if inferred else ''}, "
                    f"seed={find_key(r['snap'], 'seed')}, sha={str(find_key(r['snap'], 'git_sha'))[:8]}, "
                    f"test metrics={'yes' if r['has_test'] else 'no'}"
                    + (f", INVALID: {'; '.join(r['problems'])}" if r["problems"] else ""))

    def good(exp, need_sources):
        ok_runs = []
        for r in groups.get(exp, []):
            if r["problems"] or not r["has_test"]:
                continue
            tb = test_block(r["metrics"])
            have = {clinical_key(k) or k.lower() for k in tb}
            if all(s in have for s in need_sources):
                ok_runs.append(r)
        seeds = {find_key(r["snap"], "seed") for r in ok_runs}
        return ok_runs, seeds

    g_runs, g_seeds = good("G", ["ragtruth"])
    status = "PASS" if (seed_arg and summ and len(g_seeds) >= MIN_SEEDS) else "NOT YET"
    say(status, f"valid G runs with RAGTruth test metrics: {len(g_runs)}, distinct seeds {sorted(map(str, g_seeds))} (need {MIN_SEEDS})")
    shas = {str(find_key(r["snap"], "git_sha")) for r in g_runs}
    if len(shas) > 1:
        say("DIFFERS", f"G seeds ran on {len(shas)} different commits: check nothing changed between them")
    summarise(g_runs, "G", ["ragtruth", "halueval"])
    RESULTS["M1"] = status
    STATE.update(g_runs=g_runs, groups=groups, good=good)


def check_m3_m4():
    g_runs, groups, good = STATE["g_runs"], STATE["groups"], STATE["good"]
    section("M3  RQ1: G on clinical test sets")
    rq1_seeds = {}
    for r in g_runs:
        for k in test_block(r["metrics"]):
            if clinical_key(k):
                rq1_seeds.setdefault(clinical_key(k), set()).add(find_key(r["snap"], "seed"))
    if RQ1_DIR.exists():
        for p in RQ1_DIR.glob("*.json"):
            d = json.load(open(p))
            for k in test_block(d):
                if clinical_key(k):
                    rq1_seeds.setdefault(clinical_key(k), set()).add(d.get("seed", p.stem))
    ok = True
    for src in CLINICAL_KEYS:
        n = len(rq1_seeds.get(src, set()))
        say("PASS" if n >= MIN_SEEDS else "NOT YET", f"{src}: G seeds evaluated = {n}")
        ok &= n >= MIN_SEEDS
    RESULTS["M3"] = "PASS" if ok else "NOT YET"

    section("M4  RQ2: C and G+C")
    ok = True
    for exp in ("C", "G+C"):
        rr, seeds = good(exp, ["ragtruth"] + CLINICAL_KEYS)
        say("PASS" if len(seeds) >= MIN_SEEDS else "NOT YET",
            f"{exp}: valid runs tested on RAGTruth + all clinical sources = {len(rr)}, seeds {sorted(map(str, seeds))}")
        summarise(rr, exp, ["ragtruth"] + CLINICAL_KEYS)
        ok &= len(seeds) >= MIN_SEEDS
    unknown = groups.get("?", [])
    if unknown:
        say("DIFFERS", f"{len(unknown)} run(s) could not be classified: record 'experiment' in config_snapshot.json")
    RESULTS["M4"] = "PASS" if ok else "NOT YET"


def check_m2():
    section("M2  Clinical dataset")
    sp = load_splits(CLINICAL)
    if not sp:
        say("NOT YET", f"{CLINICAL}/ has no split files")
        RESULTS["M2"] = "NOT YET"
        return
    ok = True
    for name in SPLITS:
        rows = sp.get(name, [])
        if not rows:
            say("MISSING", f"{CLINICAL}/{name}.jsonl(.gz)")
            ok = False
            continue
        missing = [c for c in SCHEMA if c not in rows[0]]
        counts = {}
        for r in rows:
            s = r.get("source", "?")
            n, pos = counts.get(s, (0, 0))
            counts[s] = (n + 1, pos + (r.get("label") == 1))
        say("PASS" if not missing else "DIFFERS", f"{name}: missing columns {missing}" if missing else f"{name}: schema ok")
        say("INFO", f"{name}: " + ", ".join(f"{s} n={n} pos={100 * p / n:.1f}%" for s, (n, p) in sorted(counts.items())))
        ok &= not missing
    test_srcs = {clinical_key(r.get("source")) for r in sp.get("test", [])}
    lacking = [k for k in CLINICAL_KEYS if k not in test_srcs]
    say("PASS" if not lacking else "DIFFERS", f"clinical test sources present; missing: {lacking}")
    ok &= not lacking
    groups = {n: {r.get("group_id") for r in rows} for n, rows in sp.items()}
    leak = sum(len(groups[a] & groups[b]) for a, b in [("train", "val"), ("train", "test"), ("val", "test")] if a in groups and b in groups)
    say("PASS" if leak == 0 else "DIFFERS", f"group_id shared across clinical splits: {leak}")
    ok &= leak == 0
    train_ctx = {ctx_hash(r.get("context")) for r in sp.get("train", [])}
    train_ctx |= {ctx_hash(r.get("context")) for r in load_splits(GENERAL_V2 if GENERAL_V2.exists() else GENERAL_V1).get("train", [])}
    test_ctx = [ctx_hash(r.get("context")) for r in sp.get("test", [])]
    clash = sum(h in train_ctx for h in test_ctx)
    say("PASS" if clash == 0 else "DIFFERS",
        f"clinical test rows whose full context also appears in a train set: {clash} (whole-context match only; passage-level overlap needs your own dedup check)")
    ok &= clash == 0
    RESULTS["M2"] = "PASS" if ok else "NOT YET"


def check_m5():
    section("M5  NLI baseline")
    files = list(NLI_DIR.glob("*.json")) if NLI_DIR.exists() else []
    if not files:
        say("NOT YET", f"no results in {NLI_DIR}/")
        RESULTS["M5"] = "NOT YET"
        return
    have = set()
    for p in files:
        tb = test_block(json.load(open(p)))
        have |= {clinical_key(k) or k.lower() for k, v in tb.items() if metric(v) is not None}
    need = ["ragtruth"] + CLINICAL_KEYS
    lacking = [s for s in need if s not in have]
    say("PASS" if not lacking else "NOT YET", f"NLI PR-AUC present for {sorted(have)}; missing {lacking}")
    RESULTS["M5"] = "PASS" if not lacking else "NOT YET"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", action="append", default=[], help="extra folder to scan for run outputs")
    ap.add_argument("--no-query", action="store_true", help="the query field was dropped")
    a = ap.parse_args()
    if not Path("src").exists():
        sys.exit("Run from the repo root (no src/ here).")
    for fn in (check_repo, lambda: check_m0(not a.no_query),
               lambda: check_runs(a.runs_dir, not a.no_query), check_m2, check_m3_m4, check_m5):
        try:
            fn()
        except Exception as e:
            say("ERROR", f"{type(e).__name__}: {e}")
    section("MVP summary")
    for m in ("M0", "M1", "M2", "M3", "M4", "M5"):
        print(f"  {m}: {RESULTS.get(m, 'ERROR')}")


if __name__ == "__main__":
    main()