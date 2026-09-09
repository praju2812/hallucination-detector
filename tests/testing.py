import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.data import HallucinationDataset, build_tokenizer
import numpy as np

from src.data import HallucinationDataset  # or read source_info directly
from utils.io import read_jsonl


qa = [s for s in read_jsonl("data/raw/RAGTruth/dataset/source_info.jsonl")
      if s["task_type"] == "QA"]
print(type(qa[1]["source_info"]), qa[1]["source_info"].keys()
      if isinstance(qa[0]["source_info"], dict) else "str")

cfg = {"context_col":"context","response_col":"answer",
       "label_col":"label","dataset_col":"source"}
ds = HallucinationDataset("data/unified/train.jsonl", cfg)
tok = build_tokenizer("answerdotai/ModernBERT-base")

lens = []
for i in range(len(ds)):
    c, a = ds[i]["context"], ds[i]["response"]
    if not isinstance(c, str) or not isinstance(a, str):
        continue
    lens.append(len(tok(c, a)["input_ids"]))
lens = np.array(lens)
print("skipped", len(ds) - len(lens))
print("median", int(np.median(lens)), "p95", int(np.percentile(lens, 95)),
      "frac>1024", round((lens > 1024).mean(), 3))



rows = []
with open("data/unified/train.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

lengths = [len(tok(r["answer"], add_special_tokens=False)["input_ids"]) for r in rows]
print(np.percentile(lengths, [50, 90, 95, 99]), max(lengths))