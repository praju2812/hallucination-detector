"""Write the smoke-run config (Phase 2 decision D1; Phase 5 adds C and G+C).

The --base config with ONLY these changed: every data path -> a tiny file of the same family, epochs -> 1, out_dir.
    data/unified_v2/... -> data/unified_v2/tiny.jsonl
    data/clinical/...   -> <out-dir>/clinical_tiny.jsonl, written here: 10 positive + 10 negative rows per source,
                           the first ones in data/clinical/val.jsonl order (so selection has both classes)
List-valued paths stay lists of the same length (G+C trains on [general tiny, clinical tiny]).
max_length, batch size and fp16 stay as in the base config, so the smoke run also tests GPU memory at the
reported-run settings. mvp_check.py marks any run on tiny data INVALID, so it never counts as a seed.

Run (Kaggle):
    python -m scripts.make_smoke_config --out /kaggle/working/smoke.yaml --out-dir /kaggle/working/smoke
    python -m scripts.make_smoke_config --base configs/c.yaml --out /kaggle/working/smoke.yaml --out-dir /kaggle/working/smoke
"""

from __future__ import annotations
import argparse
import copy
import os

import yaml

from utils.io import read_jsonl, write_jsonl

TINY = "data/unified_v2/tiny.jsonl"
CLINICAL_PREFIX = "data/clinical/"
CLINICAL_TINY_FROM = "data/clinical/val.jsonl"
CLINICAL_TINY_NAME = "clinical_tiny.jsonl"     # "tiny" in the path is what mvp_check.py looks for
PATH_KEYS = ("train_path", "val_path", "test_path")


def _is_clinical(path: str) -> bool:
    return str(path).startswith(CLINICAL_PREFIX)


def _all_paths(data: dict) -> list[str]:
    out = []
    for k in PATH_KEYS:
        v = data.get(k)
        out += [v] if isinstance(v, str) else list(v or [])
    out += list((data.get("test_sets") or {}).values())
    return out


def needs_clinical_tiny(cfg: dict) -> bool:
    return any(_is_clinical(p) for p in _all_paths(cfg["data"]))


def make_smoke_config(base: dict, data_path: str, out_dir: str, epochs: int = 1,
                      clinical_tiny: str | None = None) -> dict:
    """Return a copy of `base` with each data path replaced by its family's tiny file (see module docstring)."""
    if needs_clinical_tiny(base) and clinical_tiny is None:
        raise ValueError("the base config has data/clinical/ paths: pass clinical_tiny")

    def tiny(path: str) -> str:
        return clinical_tiny if _is_clinical(path) else data_path

    cfg = copy.deepcopy(base)
    d = cfg["data"]
    for k in PATH_KEYS:
        if k not in d:
            continue
        d[k] = tiny(d[k]) if isinstance(d[k], str) else [tiny(p) for p in d[k]]
    if "test_sets" in d:
        d["test_sets"] = {name: tiny(p) for name, p in d["test_sets"].items()}
    cfg["train"]["epochs"] = epochs
    cfg["train"]["out_dir"] = out_dir
    return cfg


def write_clinical_tiny(src_path: str, out_path: str, n_per_class: int = 10) -> list[dict]:
    """First n_per_class positive and n_per_class negative rows of every source in src_path (file order),
    sources in sorted order; rows unchanged. Raises if a source has fewer rows of a class. Returns the rows written."""
    rows = read_jsonl(src_path)
    out = []
    for src in sorted({r["source"] for r in rows}):
        for label in (0, 1):
            picked = [r for r in rows if r["source"] == src and int(r["label"]) == label][:n_per_class]
            if len(picked) < n_per_class:
                raise ValueError(f"{src_path}: {src} has only {len(picked)} rows with label {label}")
            out += picked
    # restore file order within each source, so the tiny file reads like a slice of the real one
    order = {id(r): i for i, r in enumerate(rows)}
    out.sort(key=lambda r: (r["source"], order[id(r)]))
    write_jsonl(out, out_path)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="configs/base.yaml")
    ap.add_argument("--out", required=True, help="where to write the smoke yaml")
    ap.add_argument("--data", default=TINY)
    ap.add_argument("--out-dir", required=True, help="run output root, e.g. /kaggle/working/smoke")
    ap.add_argument("--clinical-from", default=CLINICAL_TINY_FROM, help="rows for the clinical tiny file")
    args = ap.parse_args(argv)

    with open(args.base) as f:
        base = yaml.safe_load(f)
    clinical_tiny = None
    if needs_clinical_tiny(base):
        clinical_tiny = os.path.join(args.out_dir, CLINICAL_TINY_NAME)
        n = len(write_clinical_tiny(args.clinical_from, clinical_tiny))
        print(f"wrote {clinical_tiny}: {n} rows from {args.clinical_from}")
    cfg = make_smoke_config(base, args.data, args.out_dir, clinical_tiny=clinical_tiny)
    with open(args.out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"wrote {args.out}: data={args.data}" + (f" + {clinical_tiny}" if clinical_tiny else "")
          + f", epochs=1, out_dir={args.out_dir}; everything else = {args.base}")


if __name__ == "__main__":
    main()
