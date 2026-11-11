import gzip
import io
import json
import os
from typing import Any


def _open_text(path: str, mode: str):
    """Text handle for `path`; a path ending in .gz is gzip-compressed.

    Writing uses mtime=0 so the same rows always give the same bytes (git sees no change on a rebuild).
    """
    if not path.endswith(".gz"):
        return open(path, mode, encoding="utf-8")
    if mode == "r":
        return gzip.open(path, "rt", encoding="utf-8")
    raw = open(path, "wb")
    gz = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    text = io.TextIOWrapper(gz, encoding="utf-8")
    text._raw_file = raw  # closed by close_text below
    return text


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with _open_text(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = _open_text(path, "w")
    try:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    finally:
        f.close()
        raw = getattr(f, "_raw_file", None)
        if raw is not None:
            raw.close()
