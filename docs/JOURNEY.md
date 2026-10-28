# Project Journey

This is the short, factual history of the MDM2 project. It is not a lab notebook;
it is the map of how the pipeline evolved and why the current Part_20 code looks
the way it does.

> **Correction (authoritative):** Older diary entries say "3,568 compounds." The
> modelling lineage is **2,571 raw ChEMBL records -> 1,933 unique -> 645 RO5
> compounds** (343 active / 302 inactive). The detailed correction and the
> claims not to repeat are in `docs/DISCOVERY_NOTES.md`; remaining work is in
> `docs/IMPROVEMENTS.md`.

## 1. Origin

- The project started from the public repository
  `https://github.com/arjunpahi/Natural_MDM2_Inhibitor_Discovery_using_ML.git`.
- The original work was cloned on **2026-09-07**.
- The original repository is preserved unchanged at
  `Natural_MDM2_Inhibitor_Discovery_using_ML/`.
- The original pipeline had 10 parts: Parts 1-8 traditional ML, Part 9 a GNN
  built from scratch with DeepChem `GraphConvMol`, and Part 10 a pretrained GIN
  using the MolCLR architecture.

## 2. Arjun sir's traditional-ML phase (Parts 1-8)

- ChEMBL lineage: **2,571 raw records -> 1,933 unique -> 645 RO5 compounds**
  (343 active / 302 inactive). The old "3,568 compounds" figure mixed in a
  separate ChEMBL table; see `docs/DISCOVERY_NOTES.md`.
- Lipinski/RO5 filtering produced the working table of **645 compounds**
  (343 active, 302 inactive), now stored at
  `Part_2/ro5_properties_filtered.csv`.
- Fingerprints: MACCS keys and Morgan fingerprints. The final MDM2 pipeline
  settled on Morgan radius 3, 512 bits, because the same fingerprint can be used
  for training and for screening.
- Part 6 compared 20 classifiers (RF, SVM, KNN, logistic regression, boosting,
  etc.). The strongest RF run reported ROC-AUC **0.942**, F1 **0.906**, MCC
  **0.813**, accuracy **0.913**.
- Part 8 screened the COCONUT library and reported **116 predicted inhibitors**.
- The associated paper is Budha et al., *Adv. Theory Simul.* 2025,
  DOI `10.1002/adts.202501502`; its reported RF numbers were ROC-AUC 0.960,
  F1 0.943, MCC 0.876, with 116 hits and 2 leads.

## 3. GNN phase (Parts 9-10)

- Part 9 replaced the classifier with a GCN built from scratch
  (DeepChem `GraphConvMol`). Test accuracy **0.915**, F1 **0.923**, AUC
  **0.952**, MCC **0.830**; 5-fold CV AUC **0.945 ± 0.019**.
- Part 10 used a pretrained GIN/MolCLR-style architecture. Test accuracy
  **0.899**, F1 **0.913**, AUC **0.947**, MCC **0.807**, sensitivity
  **0.986**; 5-fold CV AUC **0.937 ± 0.017**.
- Y-scrambling gave roughly **0.5 AUC**, which is the expected null.
- The original problem was **classification** (active/inactive), not potency
  regression.

## 4. Screening and model extensions (Parts 11-19)

- Part 11 applied the GNNs to the COCONUT screening library.
- Part 12 added AttentiveFP.
- Part 13 added a hybrid GCN + GAT model.
- Part 14 tried ensemble screening.
- Part 15 explored small-data deep learning.
- Part 17 explored pretraining and fine-tuning, including regression variants.
- Part 18 became the GNN triage/clean-training codebase
  (`common.py`, `screen_triage.py`, `train_gcn_fixed.py`,
  `train_gin_fixed.py`).
- Part 19 (`part19/first.ipynb`) was an intermediate experiment.
- The root `Part_9` and `Part_10` folders contain the user's HFooladi-style
  reimplementations, separate from Arjun sir's original notebooks.

## 5. The regression correction (Part 20)

The final problem was reframed as **pIC50 regression**, because a binary
active/inactive label hides potency and creates a misleading "hit" list.

Part_20_Fixed introduced:

- a leak-free scaffold split: **464 train / 52 validation / 129 test**;
- a Ridge(6 RDKit descriptors) baseline plus GNN residual (delta-ML);
- retrained GCN, GIN, AttentiveFP, and HFooladi-GIN without reusing pretrained
  weights that had seen the labels;
- validation-only early stopping, calibration, and primary-model selection;
- test evaluation only once.

The first Kaggle run exposed a serious presentation problem: the model was
extrapolating over most of the library. The raw top-100 had **0/100** compounds
inside the applicability domain, and the apparent "hits" were mostly duplicated
stereoisomers/analogs. That led to the strict-screening rewrite:

- strict AD cutoff `0.52`;
- nearest active training neighbour `>= 0.40`;
- positive active-vs-inactive margin `>= 0.02`;
- core-model agreement (range `<= 0.70`, std `<= 0.40`);
- parent + Murcko-scaffold deduplication;
- a cap of 12 diverse parents;
- explicit label: **low-confidence hypotheses, not validated hits**.

The final ensemble is a simple mean of calibrated GCN + GIN + AttentiveFP.
NNLS is retained only as a labelled diagnostic because fitting weights on 52
validation compounds was unstable.

## 6. Robust evaluation added later

The final evaluation cell adds:

- 5-repeat scaffold CV for two fast baselines: Ridge(6 descriptors) and
  Ridge(Morgan r3/512);
- official-test precision@10, EF1%, and BEDROC (alpha=20);
- bootstrap confidence intervals for R2/RMSE/MAE;
- validation-only bootstrap model-selection stability, reporting how often each
  model wins;
- HFooladi-GIN and ChemBERTa marked as sensitivity analyses, never as the
  primary selection.

The repeated-CV loop does **not** retrain the full GNNs across folds. GNNs are
evaluated on the single official test split with bootstrap intervals. That
limitation is deliberate and must be stated in any report.

## 7. What this cleanup changed

- Added a project skill at `.opencode/skills/mdm2-project/SKILL.md`.
- Added a non-destructive `organized/` copy of code, notebooks, docs, and the
  three essential data files.
- Kept every original folder and file in place.
- Added `MANIFEST.csv` with source path, destination, size, and SHA-256 for every
  copied file.
- Added provenance labels so it is clear what came from Arjun sir, what was
  adapted, and what is user work.

## 8. Lessons that matter

1. A high test R2 is not enough if the screening library is outside the training
   domain.
2. A top-100 list is not a hit list.
3. Stereoisomers and close analogs are not independent discoveries.
4. Model selection on 52 compounds is noisy; bootstrap it.
5. Never select a model because its test score is higher. Use validation only.
6. Keep the raw data and original code; organize by copying, not moving.
