# 01_traditional_ml — Traditional ML Layer

One runnable file per model. Fingerprint/descriptor featurization and scaffold
splitting are shared in `models/featurize.py` and `models/ml_common.py`; every
model file then stays small and comparable.

## Per-model files (`models/`)

| File | Model | Shared tracking/ensembles appear in |
|---|---|---|
| `rf_model.py` | Random Forest (Morgan/MACCS) | Arjun Part 6/8 origin |
| `extra_trees_model.py` | Extremely Randomized Trees | part21 table |
| `hist_gb_model.py` | Histogram Gradient Boosting | part21 table |
| `adaboost_model.py` | AdaBoost | part21 table |
| `xgboost_model.py` | XGBoost | part21 table |
| `lightgbm_model.py` | LightGBM | part21 table |
| `catboost_model.py` | CatBoost | part21 table |
| `svm_model.py` | SVM (SVC/SVR RBF) | part21 (excluded from screening vote) |
| `ridge_model.py` | Ridge on 6 RDKit descriptors | Part 20 leak-safe baseline |
| `knn_model.py` | k-Nearest Neighbours (Tanimoto-style) | part21 table |

Each file prints a smoke run with `python models/<name>.py --smoke`.

## Provenance

- Arjun sir's Parts 1–8 (ChEMBL extract → RO5 filter → EDA → fingerprints →
  PCA/t-SNE → 20-classifier catalog → COCONUT prep → RF screening) live in his
  separate local clone `../Natural_MDM2_Inhibitor_Discovery_using_ML/`
  (intentionally not tracked here). The canonical data products are already
  promoted to `../00_data/`.
- `models/*` were written by the user (2026-10-29) to give every model its own
  copy-paste-runnable file, matching Arjun's per-part teaching style.
- Classification outputs use `predict_proba` (softmax/logistic internal), and
  regression outputs are pIC50 predictions; both tracks share one 80/10/10
  Murcko-scaffold split + 5-fold group CV (seed 57).
