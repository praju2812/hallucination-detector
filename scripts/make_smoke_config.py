"""Write the Phase 2 smoke-run config (decision D1).

configs/base.yaml with ONLY these changed: train/val/test paths -> tiny.jsonl, epochs -> 1, out_dir.
max_length, batch size and fp16 stay as in base.yaml, so the smoke run also tests GPU memory at the
reported-run settings. mvp_check.py marks any run on tiny data INVALID, so it never counts as a seed.

Run (Kaggle):
    python -m scripts.make_smoke_config --out /kaggle/working/smoke.yaml --out-dir /kaggle/working/smoke
"""

from __future__ import annotations
import argparse
import copy

import yaml

TINY = "data/unified_v2/tiny.jsonl"


def make_smoke_config(base: dict, data_path: str, out_dir: str, epochs: int = 1) -> dict:
    cfg = copy.deepcopy(base)
    for k in ("train_path", "val_path", "test_path"):
        cfg["data"][k] = data_path
    cfg["train"]["epochs"] = epochs
    cfg["train"]["out_dir"] = out_dir
    return cfg


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="configs/base.yaml")
    ap.add_argument("--out", required=True, help="where to write the smoke yaml")
    ap.add_argument("--data", default=TINY)
    ap.add_argument("--out-dir", required=True, help="run output root, e.g. /kaggle/working/smoke")
    args = ap.parse_args(argv)

    with open(args.base) as f:
        base = yaml.safe_load(f)
    cfg = make_smoke_config(base, args.data, args.out_dir)
    with open(args.out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"wrote {args.out}: data={args.data}, epochs=1, out_dir={args.out_dir}; everything else = {args.base}")


if __name__ == "__main__":
    main()
