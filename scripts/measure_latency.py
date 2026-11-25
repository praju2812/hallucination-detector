"""Phase 7 (RQ3): per-request latency and cost of one scoring method on a fixed seeded sample.

One request = one full call, text in -> result out (tokenisation, NLI sentence splitting and windowing, every
forward pass), with a device sync before the clock stops. Batch size 1, one request at a time. Model loading is
timed once (startup_s); warm-up requests are run first and not recorded.

Sample (decision, 24 Nov): 50 RAGTruth QA + 50 RAGTruth Summary rows (data/unified_v2/test.jsonl, as they are) and
100 rows of each clean clinical test source (data/clinical/test.jsonl), chosen by sha1("<seed>:<row_key>") order.
It is written once to results/latency/sample_seed<seed>.jsonl and reused by every method (G, NLI, later the judge),
so latencies are comparable row for row.

Cost (AE-1 §3.6, after Luna): cost per 1,000 requests = hourly price / (qps * 3600) * 1000, qps = measured
throughput at batch size 1. Monthly cost = monthly instance price * target qps / qps per instance, only when
--target-qps is given. Prices are parameters with a stated source, never looked up by this script.

Usage (repo root):
    python -m scripts.measure_latency --method g --device cpu --checkpoint ../Hallucination_detector/runs/run_seed43_20261121_000845/best.pt --tag m2-cpu
    python -m scripts.measure_latency --method nli --device cpu --tag m2-cpu
    python -m scripts.measure_latency --print-kaggle          # T4 command list (Prajakta runs it)
Output: results/latency/<method>_<tag>.json (summary) and <method>_<tag>_requests.jsonl (one line per request).

Spec: tests/test_measure_latency.py.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np

from src.rq1 import assign_row_keys
from utils.io import read_jsonl, write_jsonl

SAMPLE_STRATA = {("ragtruth", "qa"): 50, ("ragtruth", "summary"): 50, ("ragbench_pubmedqa", None): 100,
                 ("ragbench_covidqa", None): 100, ("medhallu", None): 100}
GENERAL_TEST = "data/unified_v2/test.jsonl"
CLINICAL_TEST = "data/clinical/test.jsonl"          # clean clinical test (rule 6)
OUT_DIR = "results/latency"
SAMPLE_FIELDS = ("row_key", "source", "task", "group_id", "query", "context", "answer", "label", "split_file")
P95_METHOD = "numpy.percentile(q=95), linear interpolation"


# ---- sample (plumbing) --------------------------------------------------------------------------

def _stratum(row: dict) -> tuple[str, str | None]:
    return (row["source"], row.get("task")) if row["source"] == "ragtruth" else (row["source"], None)


def _hash_order(seed: int, key: str) -> str:
    return hashlib.sha1(f"{seed}:{key}".encode("utf-8")).hexdigest()   # hard rule 3: sha1, never hash()


def build_latency_sample(general_rows: list[dict], clinical_rows: list[dict], seed: int) -> list[dict]:
    """The stratified latency sample (SAMPLE_STRATA), in stratum order, sha1 order within a stratum.

    Row keys come from src.rq1.assign_row_keys over each whole file, so they match the RQ1 / Phase 6 keys.
    Raises ValueError if a stratum has fewer rows than it needs.
    """
    out = []
    for split_file, rows in (("general_test", general_rows), ("clinical_test", clinical_rows)):
        keys = assign_row_keys(rows)
        by_stratum: dict = {}
        for r, k in zip(rows, keys):
            by_stratum.setdefault(_stratum(r), []).append((k, r))
        for stratum, n in SAMPLE_STRATA.items():
            if (stratum[0] == "ragtruth") != (split_file == "general_test"):
                continue
            pool = by_stratum.get(stratum, [])
            if len(pool) < n:
                raise ValueError(f"stratum {stratum}: {len(pool)} rows in {split_file}, sample needs {n}")
            pool = sorted(pool, key=lambda kr: _hash_order(seed, kr[0]))[:n]
            for k, r in pool:
                out.append({"row_key": k, "source": r["source"], "task": r.get("task"),
                            "group_id": r.get("group_id"), "query": r.get("query"), "context": r["context"],
                            "answer": r["answer"], "label": r["label"], "split_file": split_file})
    order = {s: i for i, s in enumerate(SAMPLE_STRATA)}
    out.sort(key=lambda r: order[_stratum(r)])                 # stable: sha1 order kept within a stratum
    return out


def sha1_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def load_or_create_sample(path: str, general_path: str, clinical_path: str, seed: int) -> tuple[list[dict], dict]:
    """Load the saved sample, or build and save it if it does not exist yet. Never resamples an existing file.

    The first line of the file is a header {"_header": {...seed, sources, sha1 of the data files}}.
    Raises ValueError if the saved sample was drawn with another seed.
    """
    created = False
    if not os.path.exists(path):
        rows = build_latency_sample(read_jsonl(general_path), read_jsonl(clinical_path), seed)
        header = {"_header": {"seed": seed, "strata": {f"{s}/{t}": n for (s, t), n in SAMPLE_STRATA.items()},
                              "general_test": general_path, "general_sha1": sha1_file(general_path),
                              "clinical_test": clinical_path, "clinical_sha1": sha1_file(clinical_path)}}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        write_jsonl([header] + rows, path)
        created = True
    lines = read_jsonl(path)
    header, rows = lines[0].get("_header"), lines[1:]
    if header is None:
        raise ValueError(f"{path}: first line is not a sample header")
    if header["seed"] != seed:
        raise ValueError(f"{path} was drawn with seed {header['seed']}, not {seed}; use another --sample path")
    return rows, {"path": path, "sha1": sha1_file(path), "created": created, "n_rows": len(rows), **header}


# ---- hardware and sync (plumbing) ---------------------------------------------------------------

def _sysctl(name: str) -> str | None:
    try:
        return subprocess.check_output(["sysctl", "-n", name], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _linux_cpu_model() -> str | None:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def hardware_info(device: str) -> dict:
    """What the numbers were measured on, as found (not mapped to any cloud instance)."""
    import torch

    def version(pkg):
        try:
            return importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            return None

    mac = platform.system() == "Darwin"

    def mac_int(name):
        v = _sysctl(name) if mac else None
        return int(v) if v and v.isdigit() else None

    mem = mac_int("hw.memsize")
    info = {
        "device": device,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": (_sysctl("machdep.cpu.brand_string") if mac else _linux_cpu_model()) or platform.processor(),
        "logical_cores": os.cpu_count(),
        "physical_cores": mac_int("hw.physicalcpu"),
        "performance_cores": mac_int("hw.perflevel0.physicalcpu"),
        "efficiency_cores": mac_int("hw.perflevel1.physicalcpu"),
        "memory_gb": round(mem / 2**30, 1) if mem else None,
        "torch_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
        "transformers_version": version("transformers"),
        "python_version": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" and torch.cuda.is_available() else None,
        "kaggle": "KAGGLE_KERNEL_RUN_TYPE" in os.environ,
    }
    return info


def make_sync(device: str) -> Callable[[], None]:
    """Wait for queued GPU work, so the clock measures the finished request."""
    import torch
    if device == "cuda":
        return torch.cuda.synchronize
    if device == "mps":
        return torch.mps.synchronize
    return lambda: None


# ---- methods (plumbing) -------------------------------------------------------------------------

@dataclass
class MethodHandle:
    fn: Callable[[dict], dict]            # one sample row -> score result
    meta: dict                            # model_id, threshold, ...
    lengths: Callable[[dict], dict]       # one sample row -> input-size fields, computed OUTSIDE the clock


def load_method(method: str, args) -> MethodHandle:
    from src.inference import load_detector, load_nli, score, score_nli

    if method == "g":
        det = load_detector(args.run_dir, checkpoint=args.checkpoint, device=args.device)

        def g_lengths(r):
            return {"n_tokens": len(det.collator.encode_with_response_budget(r["query"], r["context"], r["answer"]))}

        return MethodHandle(fn=lambda r: score(r["query"], r["context"], r["answer"], det),
                            meta={"model_id": det.model_id, "threshold": det.threshold,
                                  "experiment": det.snapshot["experiment"], "run_dir": args.run_dir},
                            lengths=g_lengths)
    if method == "nli":
        nli = load_nli(device=args.device)

        def nli_lengths(r):
            return {"n_context_tokens": len(nli.builder.ids(r["context"])),
                    "n_response_tokens": len(nli.builder.ids(r["answer"]))}

        return MethodHandle(fn=lambda r: score_nli(r["query"], r["context"], r["answer"], nli),
                            meta={"model_id": nli.model_id, "threshold": nli.threshold}, lengths=nli_lengths)
    raise ValueError(f"unknown method {method!r}")


# ---- timing, statistics and cost (TEMPLATES: Prajakta writes the bodies) -------------------------

def time_requests(fn: Callable[[dict], dict], rows: list[dict], warmup: int, sync: Callable[[], None]) -> list[dict]:
    """Time one full call of `fn` per row, batch size 1, after `warmup` untimed calls.

    Inputs:
        fn:     row -> result dict (may contain "n_pairs" for NLI).
        rows:   the latency sample (each has row_key, source, task).
        warmup: number of untimed calls first, on rows[:warmup] (results thrown away). 0 allowed.
        sync:   called after every timed fn call, BEFORE the clock stops (GPU work is asynchronous).
    Output: one record per row, in row order:
        {"row_key", "source", "task", "seconds": float, "n_pairs": result.get("n_pairs")}
    Invariants (tests/test_measure_latency.py):
        - fn is called len(rows) + warmup times; every row is timed exactly once.
        - after warm-up the call order is fn, sync, fn, sync, ...
    """
    # validate warmup.
    if warmup < 0 or warmup > len(rows):
        raise ValueError(f"warmup must be between 0 and {len(rows)}, got {warmup}")

    # warm-up: untimed calls on the first rows, results thrown away
    for row in rows[:warmup]:
        fn(row)
        sync()

    # every row timed exactly once, in order
    records = []
    for row in rows:
        start = time.perf_counter()
        result = fn(row)
        sync()                                   # wait for GPU work before stopping the clock
        seconds = time.perf_counter() - start
        records.append({"row_key": row["row_key"], "source": row["source"], "task": row.get("task"),
                        "seconds": seconds, "n_pairs": result.get("n_pairs")})
    return records

def latency_stats(seconds: list[float]) -> dict:
    """Summary of per-request latencies.

    Output: {"n", "median_s", "p95_s", "mean_s", "total_s", "throughput_rps"}
        throughput_rps = n / total_s (requests served one at a time at batch size 1). This is the qps of the cost formula.
    p95: P95_METHOD. Raises ValueError on an empty list. Plain Python floats / int in the dict.
    """
    # validate; convert to a numpy array.
    if not seconds:
        raise ValueError("latency_stats: empty seconds list")

    # convert to a numpy array for the statistics.
    seconds_arr = np.array(seconds, dtype=float)

    # compute each statistic.
    n = len(seconds_arr)
    median_s = np.median(seconds_arr)
    p95_s = np.percentile(seconds_arr, 95)
    mean_s = np.mean(seconds_arr)
    total_s = np.sum(seconds_arr)
    throughput_rps = n / total_s if total_s > 0 else 0

    return {"n": int(n), "median_s": float(median_s), "p95_s": float(p95_s),
            "mean_s": float(mean_s), "total_s": float(total_s), "throughput_rps": float(throughput_rps)}

def summarize_latency(records: list[dict]) -> dict:
    """Latency per source, per RAGTruth task and pooled.

    Output: {"pooled": latency_stats(all),
             "per_source": {source: latency_stats(...)},
             "per_task": {"ragtruth_qa": ..., "ragtruth_summary": ...}   (RAGTruth rows only),
             "p95_method": P95_METHOD}
    """
    # pooled
    pooled_stats = latency_stats([r["seconds"] for r in records])

    # group seconds by source.
    per_source_stats = {}
    for r in records:
        source = r["source"]
        per_source_stats.setdefault(source, []).append(r["seconds"])
    per_source_stats = {source: latency_stats(seconds) for source, seconds in per_source_stats.items()}

    # group RAGTruth seconds by task; key "ragtruth_<task>".
    per_task = {}
    for r in records:
        if r["source"] == "ragtruth":
            per_task.setdefault(f"ragtruth_{r['task']}", []).append(r["seconds"])
    per_task_stats = {k: latency_stats(v) for k, v in per_task.items()}

    return {
        "pooled": pooled_stats,
        "per_source": per_source_stats,
        "per_task": per_task_stats,
        "p95_method": P95_METHOD
    }

def cost_per_1000(price_per_hour: float, qps: float) -> float:
    """Cost of 1,000 requests on one instance (Luna): price_per_hour / (qps * 3600) * 1000.

    qps = measured throughput at batch size 1 (latency_stats(...)["throughput_rps"]).
    Raises ValueError if qps <= 0 or price_per_hour < 0.
    """
    # validate.
    if qps <= 0:
        raise ValueError(f"qps must be > 0, got {qps}")
    if price_per_hour < 0:
        raise ValueError(f"price_per_hour must be >= 0, got {price_per_hour}")
    
    #compute requests per hour, then the formula.
    requests_per_hour = qps * 3600
    cost = (price_per_hour / requests_per_hour) * 1000
    return cost

def monthly_cost(monthly_price: float, target_qps: float | None, qps_per_instance: float) -> float | None:
    """Monthly cost to serve `target_qps`: monthly_price * target_qps / qps_per_instance (AE-1 §3.6).

    Returns None when target_qps is None (no target given). Raises ValueError if qps_per_instance <= 0.
    """
    # no target -> None.
    if target_qps is None:
        return None
    
    # validate.
    if qps_per_instance <= 0:
        raise ValueError(f"qps_per_instance must be > 0, got {qps_per_instance}")

    if target_qps < 0:
        raise ValueError(f"target_qps must be >= 0, got {target_qps}")
    
    if monthly_price < 0:
        raise ValueError(f"monthly_price must be >= 0, got {monthly_price}")

    # valid formula: monthly_price * target_qps / qps_per_instance.
    cost = monthly_price * target_qps / qps_per_instance
    return cost


# ---- CLI (plumbing) -----------------------------------------------------------------------------

KAGGLE_COMMANDS = """\
# Kaggle notebook, GPU T4 x1. Inputs: this repo cloned from GitHub; the seed 43 best.pt as a Kaggle dataset.
# Paths are absolute under /kaggle/working/ (hard rule 9). The sample file is the committed one (never resampled).
!git clone https://github.com/<you>/<repo>.git /kaggle/working/repo
%cd /kaggle/working/repo
!pip install -q transformers==5.22.2 pysbd==0.3.4
!nvidia-smi
!python -m scripts.measure_latency --method g --device cuda --tag t4 \\
    --checkpoint /kaggle/input/<dataset>/best.pt \\
    --out /kaggle/working/results/latency --sample /kaggle/working/repo/results/latency/sample_seed42.jsonl \\
    --price-per-hour <USD> --price-source "<provider, instance, page, date>"
!python -m scripts.measure_latency --method nli --device cuda --tag t4 \\
    --out /kaggle/working/results/latency --sample /kaggle/working/repo/results/latency/sample_seed42.jsonl \\
    --price-per-hour <USD> --price-source "<provider, instance, page, date>"
# Download /kaggle/working/results/latency/*.json(l) into results/latency/ locally.
"""


def parse_args(argv=None):
    from src.inference import DEFAULT_RUN_DIR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["g", "nli"])
    ap.add_argument("--device", default=None, help="default: cuda > mps > cpu")
    ap.add_argument("--tag", default=None, help="hardware tag in the file name, e.g. m2-cpu, t4 (default: device)")
    ap.add_argument("--run-dir", default=DEFAULT_RUN_DIR, help="served run (G, C or G+C) for --method g")
    ap.add_argument("--checkpoint", default=os.environ.get("HD_CHECKPOINT"),
                    help="weights path if not <run-dir>/best.pt (default: $HD_CHECKPOINT)")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42, help="sample seed")
    ap.add_argument("--threads", type=int, default=None, help="torch.set_num_threads (CPU); recorded either way")
    ap.add_argument("--general-test", default=GENERAL_TEST)
    ap.add_argument("--clinical-test", default=CLINICAL_TEST)
    ap.add_argument("--sample", default=None, help="sample file (default: <out>/sample_seed<seed>.jsonl)")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--price-per-hour", type=float, default=None, help="USD per instance-hour (a stated parameter)")
    ap.add_argument("--price-source", default=None, help="where the price comes from (required with a price)")
    ap.add_argument("--monthly-price", type=float, default=None, help="USD per instance-month (default: hourly * 730)")
    ap.add_argument("--target-qps", type=float, default=None, help="monthly cost only when given")
    ap.add_argument("--print-kaggle", action="store_true", help="print the Kaggle T4 command list and exit")
    args = ap.parse_args(argv)
    if not args.print_kaggle and not args.method:
        ap.error("--method is required")
    if args.price_per_hour is not None and not args.price_source:
        ap.error("--price-source is required with --price-per-hour")
    return args


def print_table(summary: dict) -> None:
    def row(name, s):
        print(f"  {name:<20} n={s['n']:>4}  median {s['median_s'] * 1000:8.1f} ms  p95 {s['p95_s'] * 1000:8.1f} ms  "
              f"throughput {s['throughput_rps']:7.2f} req/s")
    row("pooled", summary["pooled"])
    for k, s in summary["per_source"].items():
        row(k, s)
    for k, s in summary["per_task"].items():
        row(k, s)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.print_kaggle:
        print(KAGGLE_COMMANDS)
        return 0

    import torch
    from src.train import get_device, git_state

    git = git_state()                                   # read once, before anything is written
    device = args.device or get_device()
    args.device = device
    if args.threads:
        torch.set_num_threads(args.threads)
    tag = args.tag or device
    sample_path = args.sample or os.path.join(args.out, f"sample_seed{args.seed}.jsonl")
    rows, sample_info = load_or_create_sample(sample_path, args.general_test, args.clinical_test, args.seed)
    print(f"sample {sample_path} ({len(rows)} rows, sha1 {sample_info['sha1'][:10]}, "
          f"{'created' if sample_info['created'] else 'reused'})")

    hw = hardware_info(device)
    print(f"method {args.method}  device {device}  {hw['cpu_model']}  gpu {hw['gpu']}  threads {hw['torch_threads']}")
    t0 = time.perf_counter()
    handle = load_method(args.method, args)
    make_sync(device)()
    startup_s = time.perf_counter() - t0
    print(f"  loaded in {startup_s:.1f}s  {handle.meta}")

    records = time_requests(handle.fn, rows, args.warmup, make_sync(device))
    by_key = {r["row_key"]: r for r in rows}
    for rec in records:
        rec.update(handle.lengths(by_key[rec["row_key"]]))      # input sizes, outside the timed loop
    summary = summarize_latency(records)
    print_table(summary)

    cost = None
    if args.price_per_hour is not None:
        qps = summary["pooled"]["throughput_rps"]
        monthly_price = args.monthly_price if args.monthly_price is not None else args.price_per_hour * 730
        cost = {"price_per_hour": args.price_per_hour, "price_source": args.price_source,
                "qps_batch1": qps, "cost_per_1000": cost_per_1000(args.price_per_hour, qps),
                "per_source_cost_per_1000": {k: cost_per_1000(args.price_per_hour, s["throughput_rps"])
                                             for k, s in summary["per_source"].items()},
                "monthly_price": monthly_price,
                "monthly_price_basis": "given" if args.monthly_price is not None else "hourly * 730",
                "target_qps": args.target_qps,
                "monthly_cost": monthly_cost(monthly_price, args.target_qps, qps),
                "formula": "price_per_hour / (qps * 3600) * 1000; monthly = monthly_price * target_qps / qps "
                           "(AE-1 §3.6, Luna)"}
        print(f"  cost per 1,000 requests: ${cost['cost_per_1000']:.5f} at ${args.price_per_hour}/h "
              f"({args.price_source})")

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, f"{args.method}_{tag}")
    result = {
        "method": args.method, **handle.meta, "tag": tag, "hardware": hw, "startup_s": round(startup_s, 3),
        "sample": {k: v for k, v in sample_info.items() if k != "created"}, "warmup": args.warmup, "batch_size": 1,
        "timing": "full call text in -> result out, device sync before stop; load and warm-up excluded",
        "summary": summary, "cost": cost, **git,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join([os.path.basename(sys.executable), "-m", "scripts.measure_latency"] + (argv or sys.argv[1:])),
    }
    with open(stem + ".json", "w") as f:
        json.dump(result, f, indent=2)
    write_jsonl(records, stem + "_requests.jsonl")
    print(f"wrote {stem}.json and {stem}_requests.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
