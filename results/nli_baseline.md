# NLI baseline: zero-shot DeBERTa NLI

Model MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli, no training. Score = 1 - min over response sentences of max over context windows of P(entailment). One deterministic run, so cells are single values (no seed spread). Threshold (a) = tuned on RAGTruth validation, applied to every source (primary, zero-shot). Threshold (b) = tuned on each clinical source's own validation rows (secondary). Same grid as training.

## Run

| model | revision | hypothesis | windows | hypothesis cap | splitter | device | precision | scoring git sha | dirty | determinism |
|---|---|---|---|---|---|---|---|---|---|---|
| MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli | b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7 | response | 512 tokens, stride 128 | 256 tokens | pysbd 0.3.4 (newlines first, < 3 words merged) | cuda (Tesla T4) | torch.float32, autocast=False | f63ebec | False | 20 rows, max abs diff 0.0e+00 (tol 1e-04, ranking unchanged True; PASS) |

## Thresholds

| threshold | used for | tuned on (validation) | positives / rows | value |
|---|---|---|---|---|
| (a) | every source (primary) | RAGTruth | 296 / 984 | 0.30 |
| (b) | PubMedQA | PubMedQA | 145 / 526 | 0.65 |
| (b) | CovidQA | CovidQA, PubMedQA | 162 / 614 | 0.65 |
| (b) | MedHallu | MedHallu | 869 / 1738 | 0.50 |

## RAGTruth test

For RAGTruth, (a) and (b) are the same threshold. The QA and Summary rows are a breakdown of the RAGTruth row (min over sentences may penalise long responses).

| source | prevalence (n_pos / n) | PR-AUC | AUROC | P @ (a) | R @ (a) | F1 @ (a) | F1 @ (b) | F1 if every row flagged |
|---|---|---|---|---|---|---|---|---|
| RAGTruth | 20.2% (364 / 1800) | 0.3919 | 0.7186 | 0.2941 | 0.8434 | 0.4361 | 0.4361 | 0.3364 |
| RAGTruth QA | 17.8% (160 / 900) | 0.3491 | 0.7088 | 0.2413 | 0.8625 | 0.3770 | 0.3770 | 0.3019 |
| RAGTruth Summary | 22.7% (204 / 900) | 0.5882 | 0.7954 | 0.3581 | 0.8284 | 0.5000 | 0.5000 | 0.3696 |

## Clinical test: full

| source | prevalence (n_pos / n) | PR-AUC | AUROC | P @ (a) | R @ (a) | F1 @ (a) | F1 @ (b) | F1 if every row flagged |
|---|---|---|---|---|---|---|---|---|
| PubMedQA | 29.6% (725 / 2450) | 0.3968 | 0.6311 | 0.3466 | 0.7917 | 0.4822 | 0.4645 | 0.4567 |
| CovidQA | 15.9% (39 / 246) | 0.2703 | 0.7251 | 0.3086 | 0.6410 | 0.4167 | 0.2564 | 0.2737 |
| MedHallu | 50.0% (1000 / 2000) | 0.5990 | 0.6394 | 0.5520 | 0.9440 | 0.6967 | 0.6972 | 0.6667 |

## Clinical test: clean

| source | prevalence (n_pos / n) | PR-AUC | AUROC | P @ (a) | R @ (a) | F1 @ (a) | F1 @ (b) | F1 if every row flagged |
|---|---|---|---|---|---|---|---|---|
| PubMedQA | 28.4% (148 / 522) | 0.4116 | 0.6244 | 0.3279 | 0.8108 | 0.4669 | 0.4381 | 0.4418 |
| CovidQA | 19.0% (20 / 105) | 0.3006 | 0.7065 | 0.3243 | 0.6000 | 0.4211 | 0.2162 | 0.3200 |
| MedHallu | 50.0% (999 / 1998) | 0.5989 | 0.6392 | 0.5521 | 0.9439 | 0.6967 | 0.6973 | 0.6667 |

## Notes

- Hypothesis format: response (question in the hypothesis for QA rows: no; Summary rows never get a question).
- Sentences: newline split first, then pysbd; pieces under 3 words merged into a neighbour; a response with nothing left is scored whole. Hypotheses capped at 256 tokens (question shortened first). Fallback rows and capped hypotheses: none in any split.
- Threshold (b) tuning rows: PubMedQA on PubMedQA (145 / 526 positives); CovidQA on CovidQA, PubMedQA (162 / 614 positives); MedHallu on MedHallu (869 / 1738 positives).
- MedHallu validation is pqa_artificial; MedHallu test is pqa_labeled.
- Clean-test predictions are taken from the test_full predictions by row key (no row scored twice).
- Scoring: torch 2.11.0+cu130, transformers 5.22.2, 49756 (premise window, hypothesis) pairs.
