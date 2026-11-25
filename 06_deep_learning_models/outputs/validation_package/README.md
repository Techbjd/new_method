# Validation package -- generated 2026-11-23

[GENERATED] by `06_deep_learning_models/validation_package.py` in 53.7 s.

## Scientific status / caveats

- **Fast nested-CV AUC/AP/BalAcc/MCC numbers are sanity checks only.** They are
  not validation of hits and not prospective performance.
- **The official 129-compound scaffold-disjoint test set was never used for
  selection.** All hyperparameter and threshold choices were made in inner
  GroupKFold folds drawn only from the 516 train+val compounds. The test was
  touched only to *re-score already-existing* fast prediction CSVs, once,
  descriptively.
- **No external test set exists yet.** There is no prospective validation set
  in this repository. Nothing here upgrades a predicted active to a hit.
- The official split leaves train and val sharing scaffolds (test is disjoint
  from both). This is documented in `nested_cv_summary.json`; all CV folds here
  use Murcko-scaffold GroupKFold.

## What WAS run

1. Rebuilt the exact 464/52/129 Murcko scaffold split from
   `build_notebooks.py SPLIT_SRC` (seed 57) and asserted test isolation.
2. Repeated nested scaffold CV on train+val only:
   GroupKFold by Murcko scaffold, 5 folds x 3 seeded repeats (seeds 57,58,59; scaffold-label permutation keeps folds disjoint)
   GroupKFold by Murcko scaffold, 3 folds; hyperparameter and probability-threshold selection from inner OOF AUC (tie: AP) only
   Models: ecfp_logreg, ecfp_random_forest, descriptor_logreg.
   Inner selection counts: {"ecfp_logreg": 2, "ecfp_random_forest": 13}.
   Selected-model fold counts: ecfp_logreg: n=2, ecfp_random_forest: n=13.
3. Re-scored the preserved default-23 snapshots in
   `outputs/validation_package/source_fast_predictions/` (falling back to
   `outputs/dl01..dl05/fast/test_predictions.csv` only if a snapshot is absent;
   no retraining): number of test actives, ROC AUC, precision@10, EF1%
   (top 2 of 129), BEDROC(alpha=20), and 2000-draw fixed-seed bootstrap 95%
   CIs for AUC, precision@10 and BEDROC.
4. Wrote this README plus the CSV/JSON files listed below.

## What was NOT run

- No retraining or re-tuning of any DL_01..DL_05 deep-learning model.
- No new scoring of the official test set for the purpose of choosing a model;
  no comparison-based "best model" selection from test enrichment.
- No external/prospective test set, no experimental assay, no applicability
  domain expansion, no hit validation claim.
- No changes to `build_notebooks.py`, notebooks, docs, source data or archives.

## Exact commands

```bash
cd "/home/bijay/Desktop/already read paper/new_method/06_deep_learning_models"
python3 validation_package.py

# optional, e.g. 2 repeats x 3 outer folds
VALID_REPEATS=2 VALID_FOLDS=3 python3 validation_package.py
```

Dependencies: installed numpy, pandas, scikit-learn and rdkit only. No network.

## Outputs (all under `outputs/validation_package/`)

- `nested_cv_folds.csv` -- one row per (repeat, outer fold, model family);
  includes inner AUC/AP, the inner-only-selected flag, chosen parameters and
  threshold, and outer-fold AUC/AP/BalAcc/MCC.
- `nested_cv_summary.json` -- mean +/- std per model across outer folds, plus
  selected-by-inner-only summary (sanity check only).
- `fast_test_enrichment.csv` -- one row per available score column of each
  existing fast test prediction file, with point estimates and bootstrap CIs;
  every row is flagged `descriptive_only=True`, `used_for_model_selection=False`.
- `fast_test_enrichment_summary.json` -- grouped by notebook, including
  warnings for skipped files/schemas.
- `README.md` -- this file.

## Honest limitations

- If a prediction CSV has an unexpected schema it is skipped with a printed
  warning and recorded in `fast_test_enrichment_summary.json`.
- AUC/BEDROC are undefined for single-class label vectors and are reported as
  empty (NaN) rather than fabricated.
- EF1% on 129 compounds uses only 2 top-ranked molecules; treat it as a
  noisy descriptive statistic.
