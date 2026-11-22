# G test metrics (runs)

3 run(s). Spread = mean ± sample std [min, max] across seeds. F1 uses each run's RAGTruth-validation threshold.

## Runs

| run folder | seed | git sha | dirty | device | threshold | peak val epoch |
|---|---|---|---|---|---|---|
| run_seed42_20261120_225509 | 42 | 266b0b3 | False | Tesla T4 | 0.2 | 1 |
| run_seed43_20261121_000845 | 43 | 266b0b3 | False | Tesla T4 | 0.35 | 1 |
| run_seed44_20261121_012213 | 44 | 266b0b3 | False | Tesla T4 | 0.2 | 1 |

## Per source (test)

| source | prevalence (n_pos / n) | PR-AUC | F1 @ val threshold | AUROC | runs |
|---|---|---|---|---|---|
| halueval | 50.0% (1000 / 2000) | 0.9993 ± 0.0002 [0.9990, 0.9995] | 0.9925 ± 0.0005 [0.9920, 0.9930] | 0.9990 ± 0.0004 [0.9986, 0.9994] | 3 |
| ragtruth | 20.2% (364 / 1800) | 0.4813 ± 0.0280 [0.4491, 0.4992] | 0.4674 ± 0.0158 [0.4492, 0.4780] | 0.7609 ± 0.0148 [0.7439, 0.7696] | 3 |

## RAGTruth test: context truncated vs untruncated

Per seed:

| run folder | subset | prevalence (n_pos / n) | PR-AUC | F1 | AUROC |
|---|---|---|---|---|---|
| run_seed42_20261120_225509 | ragtruth_truncated | 27.5% (74 / 269) | 0.4219 | 0.4569 | 0.6633 |
| run_seed42_20261120_225509 | ragtruth_untruncated | 18.9% (290 / 1531) | 0.5131 | 0.4850 | 0.7816 |
| run_seed43_20261121_000845 | ragtruth_truncated | 27.5% (74 / 269) | 0.4113 | 0.4398 | 0.6480 |
| run_seed43_20261121_000845 | ragtruth_untruncated | 18.9% (290 / 1531) | 0.5157 | 0.4840 | 0.7862 |
| run_seed44_20261121_012213 | ragtruth_truncated | 27.5% (74 / 269) | 0.4195 | 0.4963 | 0.6838 |
| run_seed44_20261121_012213 | ragtruth_untruncated | 18.9% (290 / 1531) | 0.4548 | 0.4351 | 0.7519 |

Across seeds:

| subset | prevalence (n_pos / n) | PR-AUC | F1 @ val threshold | AUROC | runs |
|---|---|---|---|---|---|
| ragtruth_truncated | 27.5% (74 / 269) | 0.4176 ± 0.0056 [0.4113, 0.4219] | 0.4643 ± 0.0290 [0.4398, 0.4963] | 0.6650 ± 0.0180 [0.6480, 0.6838] | 3 |
| ragtruth_untruncated | 18.9% (290 / 1531) | 0.4945 ± 0.0344 [0.4548, 0.5157] | 0.4680 ± 0.0285 [0.4351, 0.4850] | 0.7732 ± 0.0186 [0.7519, 0.7862] | 3 |
