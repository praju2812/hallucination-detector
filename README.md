# Hallucination Detector

Supervised cross-encoder that classifies an LLM answer as faithful or hallucinated,
given the context it was meant to be grounded in. This is the trained method in the
dissertation's comparison; the reference-free signals (self-consistency, semantic
entropy, NLI grounding) and LLM-as-judge sit alongside it in the wider study.

## What is scaffolded vs yours to write

Filled in: config, seeding, tokenisation with correct truncation, model wrapper,
optimiser/scheduler, checkpointing on best val PR-AUC, ranking + thresholded metrics.

Yours to write (marked `TODO(you)`): the training step body, the eval forward loop,
the multi-passage context join, ECE, mean-pooling ablation, per-source printout.

## Design decisions already baked in

- **Cross-encoder, single logit.** One probability per example (sigmoid of the logit),
  which is what the FastAPI service will return and what you calibrate.
- **Truncate context, keep response whole** (`truncation="only_first"`). You can't judge
  an answer you can't fully see.
- **PR-AUC for model selection**, not accuracy, because the classes are imbalanced.
- **Threshold chosen on val, reported on test.** Never pick it on test.

## Decisions you still need to make

1. **Encoder + context length.** Default is `deberta-v3-base` at 512 tokens. RAGTruth
   contexts can exceed 512, so some context gets truncated. If that truncation is
   dropping grounding evidence, switch to a long-context encoder (ModernBERT, Longformer)
   and raise `max_length`. Check how often you truncate before deciding.
2. **Manual loop vs HF Trainer.** This scaffold is a manual loop so you see every step.
   If you'd rather move fast, `transformers.Trainer` with a `compute_metrics` gives you
   the same result with less code. Either is defensible in the write-up.

## Niche-agnostic by design

Train on RAGTruth/HaluEval now. When the topic is agreed, add the niche dataset as a
new `source` and point `test_path` at it, or evaluate it as a separate held-out slice.
No model code changes. The niche is an evaluation lens, not a retrain.

## Run

```bash
pip install -r requirements.txt
python -m src.train --config configs/base.yaml
```

## Next phase (not in this scaffold)

FastAPI endpoint returning the calibrated score, request logging, confidence
monitoring, Docker. Build that once the model trains and the metrics hold.
