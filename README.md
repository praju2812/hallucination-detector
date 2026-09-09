# Hallucination Detector

Supervised cross-encoder that classifies an LLM answer as faithful or hallucinated,
given the context it was meant to be grounded in. The trained classifier is compared
against reference-free monitoring signals (self-consistency, semantic entropy, NLI
grounding) and an LLM-as-judge baseline.

## Data

Trained on a unified dataset built from two public sources:

- **RAGTruth** — real, human-annotated hallucination spans. Joins `response.jsonl` and
  `source_info.jsonl` on `source_id`. Roughly 30% positive (hallucinated).
- **HaluEval** — synthetic QA pairs, 50/50 balanced.

`scripts/build_dataset.py` turns the raw downloads into a unified schema
(`context`, `answer`, `label`, `source`) and writes train/val/test splits to
`data/unified/`. 

Raw datasets are not committed; download them from their original
repositories and run the build script to reproduce the splits. Splitting is
group-based (GroupShuffleSplit) so the same context never spans train and test,
which prevents leakage.

Final splits: 25,008 train / 2,784 val / 3,800 test.

## Model

`answerdotai/ModernBERT-base`, `max_length: 1024`. The encoder was chosen by
measurement, not default: a token-length scan on the unified training set showed
RoBERTa-base at 512 truncated ~21% of pairs, while ModernBERT at 1024 truncates ~5%.

## Design decisions

- **Cross-encoder, single logit.** One probability per example (sigmoid of the logit),
  which is what the serving layer will return and what gets calibrated.
- **Reserve-a-response-budget truncation.** The response is protected up to a token
  ceiling (`resp_max`), and the context absorbs the truncation. You can't judge an
  answer you can't fully see, so the response is the last thing cut. The response-length
  distribution (p99 = 321 tokens) set `resp_max` at 384. Encoding lives in one place
  (`src/encoding.py`) so training and future serving tokenise identically.
- **PR-AUC for model selection**, not accuracy, because the classes are imbalanced.
  `pos_weight` in BCE up-weights the positive class.
- **Threshold chosen on val, reported on test.** Never picked on test.

## Niche-agnostic by design

Trained on RAGTruth/HaluEval. A domain-specific evaluation set can be added as a new
`source` and evaluated as a held-out slice, or `test_path` pointed at it, with no model
code changes. The domain is an evaluation lens, not a retrain.

## Structure

```
src/          data.py, encoding.py, model.py, metrics.py, train.py
scripts/      build_dataset.py
configs/      base.yaml, test_config.yaml
data/unified/ train / val / test splits
```

## Run

```bash
pip install -r requirements.txt
python -m src.train --config configs/base.yaml
```

`configs/test_config.yaml` runs a fast overfit sanity check on a tiny slice, used to
confirm the training loop is wired correctly before committing to a full run.

## Next phase

FastAPI endpoint returning the calibrated score, request logging, confidence
monitoring, and Docker containerisation. Planned, not yet built.