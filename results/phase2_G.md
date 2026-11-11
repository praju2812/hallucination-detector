# G test metrics (runs)

3 run(s). Spread = mean ± sample std [min, max] across seeds. F1 uses each run's RAGTruth-validation threshold.

## Runs

| run folder | seed | git sha | dirty | device | threshold | peak val epoch |
|---|---|---|---|---|---|---|
| run_seed42_20261110_120515 | 42 | 6294bb2 | False | Tesla T4 | 0.1 | 1 |
| run_seed43_20261110_135827 | 43 | 6294bb2 | False | Tesla T4 | 0.39999999999999997 | 1 |
| run_seed44_20261110_151337 | 44 | 6294bb2 | False | Tesla T4 | 0.2 | 1 |

## Per source (test)

| source | prevalence (n_pos / n) | PR-AUC | F1 @ val threshold | AUROC | runs |
|---|---|---|---|---|---|
| halueval | 50.0% (1000 / 2000) | 0.9990 ± 0.0004 [0.9988, 0.9995] | 0.9915 ± 0.0010 [0.9905, 0.9925] | 0.9987 ± 0.0006 [0.9983, 0.9994] | 3 |
| ragtruth | 20.2% (364 / 1800) | 0.4965 ± 0.0532 [0.4471, 0.5529] | 0.4631 ± 0.0370 [0.4376, 0.5055] | 0.7709 ± 0.0348 [0.7383, 0.8076] | 3 |

## RAGTruth test: context truncated vs untruncated

Per seed:

| run folder | subset | prevalence (n_pos / n) | PR-AUC | F1 | AUROC |
|---|---|---|---|---|---|
| run_seed42_20261110_120515 | ragtruth_truncated | 27.5% (74 / 269) | 0.4506 | 0.5000 | 0.6960 |
| run_seed42_20261110_120515 | ragtruth_untruncated | 18.9% (290 / 1531) | 0.5734 | 0.5069 | 0.8232 |
| run_seed43_20261110_135827 | ragtruth_truncated | 27.5% (74 / 269) | 0.4010 | 0.4025 | 0.6249 |
| run_seed43_20261110_135827 | ragtruth_untruncated | 18.9% (290 / 1531) | 0.5137 | 0.4577 | 0.7869 |
| run_seed44_20261110_151337 | ragtruth_truncated | 27.5% (74 / 269) | 0.4084 | 0.4537 | 0.6663 |
| run_seed44_20261110_151337 | ragtruth_untruncated | 18.9% (290 / 1531) | 0.4556 | 0.4321 | 0.7476 |

Across seeds:

| subset | prevalence (n_pos / n) | PR-AUC | F1 @ val threshold | AUROC | runs |
|---|---|---|---|---|---|
| ragtruth_truncated | 27.5% (74 / 269) | 0.4200 ± 0.0268 [0.4010, 0.4506] | 0.4521 ± 0.0488 [0.4025, 0.5000] | 0.6624 ± 0.0357 [0.6249, 0.6960] | 3 |
| ragtruth_untruncated | 18.9% (290 / 1531) | 0.5142 ± 0.0589 [0.4556, 0.5734] | 0.4656 ± 0.0380 [0.4321, 0.5069] | 0.7859 ± 0.0378 [0.7476, 0.8232] | 3 |
