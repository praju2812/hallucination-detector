# Results

Index of `results/`, grouped by research question. Scripts run from the repo root as modules
(`python -m scripts.<name>`). "To come" marks outputs not yet produced.

Training run folders (`config_snapshot.json`, `metrics.json`, `predictions.jsonl`, `val_history.json`) live in
`runs/`, not here; they are written by `src/train.py`. Checkpoints are not in the repo.

`<G run>` below is one of the three fit_first G runs: `run_seed42_20261120_225509`, `run_seed43_20261121_000845`,
`run_seed44_20261121_012213`. `<model>` is `DeBERTa-v3-large-mnli-fever-anli-ling-wanli`.

## 1. Data and input length

| File or folder | Contents | Produced by |
|---|---|---|
| `phase1_truncation.md` | Truncation at 1,024 tokens on the general data (`data/unified_v2`, query field included), per split and source. Written before the truncation mode was recorded in the header; the tables match a fit_first re-run. | Printed output of `scripts/report_truncation.py` |
| `truncation_clinical.md` | The same truncation table on the clinical data (`data/clinical`), including test_full. | Printed output of `scripts/report_truncation.py` |
| `truncation_flags_v2.jsonl` | One line per general row (31,592): `split`, `idx`, `source`, `group_id`, `ctx_truncated`. Joins `predictions.jsonl` on (`split`, `idx`). Identical when regenerated under fit_first. | `scripts/report_truncation.py --per-row` |
| `analysis/encoder_change_count.md` | Rows whose encoding changed from the legacy rule to fit_first, per source and split (general and clinical). | `scripts/analysis/encoder_change_count.py` at commit 966bc8d (removed with the legacy mode) |

## 2. Analyses of G

| File or folder | Contents | Produced by |
|---|---|---|
| `analysis_2026-11-11.md`, `analysis_2026-11-11/` | Frozen, from before the encoder change. Input lengths (general and clinical), test-side overlap, Summary length cue, question cue. Later runs never write here. | `scripts/analysis/a_lengths.py`, `a_lengths_clinical.py`, `b_test_overlap.py`, `c_summary_length_cue.py`, `d_question_cue.py` |
| `analysis_2026-11-21_G-fit-first/` | Current. A3 and C4 (the parts that depend on G's predictions) re-run on the fit_first G, with printed output; its README lists the parts that are unchanged. | `scripts/analysis/a_lengths.py`, `scripts/analysis/c_summary_length_cue.py` |

## 3. G training

| File or folder | Contents | Produced by |
|---|---|---|
| `phase2_G.md` | G (fit_first), seeds 42 to 44: per-source test metrics as mean +/- std, with RAGTruth-validation thresholds. | `scripts/summarize_runs.py runs --out results/phase2_G.md`, run when `runs/` held only the G runs |
| `kaggle_logs/wall_times_G.json`, `kaggle_logs/wall_times_C.json` | Training wall time in seconds per seed, for G and C. `wall_times_GC.json` (G+C): to come. | Written by the Kaggle training notebook during a run and copied here; no repo script writes them |

## 4. RQ1: G zero-shot on clinical data

| File or folder | Contents | Produced by |
|---|---|---|
| `rq1/<G run>/predictions.jsonl`, `rq1/<G run>/scoring.json` | G scores per seed on clinical val and test_full, with scoring provenance (truncation mode included). | `scripts/score_rq1.py --run-dir runs/<G run>` |
| `rq1/<G run>/metrics.json`, `rq1/<G run>.json` | Per-seed metrics on full and clean clinical test (full detail; summary). | `scripts/report_rq1.py` |
| `rq1_G.md` | RQ1 tables across seeds, at thresholds tuned on (a) RAGTruth validation and (b) clinical validation. | `scripts/report_rq1.py` |

## 5. RQ2: C and G+C

| File or folder | Contents | Produced by |
|---|---|---|
| `runs/run_seed42_20261124_184829`, `run_seed43_20261124_204658`, `run_seed44_20261124_224636` | C run folders, seeds 42 to 44 (24 Nov, commit 525c33c). In `runs/`, not here. | `src/train.py` on Kaggle |
| G+C run folders | To come. | `src/train.py` on Kaggle |
| `rq2/<run>/metrics.json`, `rq2/<run>.json` | To come. Per-run metrics for G, C and G+C on clinical and general test (full detail; summary). | `scripts/report_rq2.py`, once G+C is in |
| `rq2.md` | To come. G, C and G+C tables. | `scripts/report_rq2.py`, once G+C is in |

## 6. RQ3: accuracy

| File or folder | Contents | Produced by |
|---|---|---|
| `nli/<model>/predictions.jsonl`, `nli/<model>/scoring.json` | Zero-shot NLI scores on RAGTruth val and test and clinical val and test_full, with model revision and scoring provenance. | `scripts/score_nli.py` |
| `nli/<model>/metrics.json`, `nli/<model>.json` | NLI metrics per source (full detail; summary). | `scripts/report_nli.py` |
| `nli_baseline.md` | NLI baseline tables. | `scripts/report_nli.py` |
| LLM judge | To come. | To come |

## 7. RQ3: latency and cost

| File or folder | Contents | Produced by |
|---|---|---|
| `latency/sample_seed42.jsonl` | The fixed 400-row latency sample shared by every method: 50 RAGTruth QA, 50 RAGTruth Summary, 100 per clean clinical test source. | `scripts/measure_latency.py` (first run writes it; later runs reuse it) |
| `latency/g_m2-cpu.json`, `latency/g_m2-cpu_requests.jsonl` | G (seed 43 run) on an Apple M2 CPU: summary (median, p95, throughput, hardware, exact command) and one line per request. No price given, so no cost. | `scripts/measure_latency.py --method g --device cpu --tag m2-cpu` (command recorded in the JSON) |
| NLI latency | To come. | `scripts/measure_latency.py --method nli` |
| G and NLI on the Kaggle T4, with cost | To come. | `scripts/measure_latency.py` (command list from `--print-kaggle`) |
| Judge latency and cost | To come. | To come |
