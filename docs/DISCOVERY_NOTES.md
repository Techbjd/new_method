# Discovery Notes and Corrections

This file is the detailed, source-checked record of the long journey: data lineage,
model history, screening collapse, silent bugs, environment failures, and claims that
must not be repeated. It supplements `JOURNEY.md` and corrects simplified numbers in
earlier notes. Sources: `diary.md`, the Arjun-sir repository, per-part READMEs, notebook
outputs pasted during the work, and the later Part_20 audit.

> Rule: the original folders are archive. Nothing here authorizes deleting or moving
> original files. Curated copies live under `organized/`; provenance is in
> `organized/MANIFEST.csv` and `PROVENANCE.md`.

## 1. Data lineage and label drift

- Raw ChEMBL `CHEMBL5023` IC50 records: **2,571**.
- After unique-compound handling: **1,933**.
- After longest-fragment cleaning and Lipinski rule-of-five filtering (≤1 violation):
  **645** compounds, **343 active / 302 inactive**.
- `Part_1/bioactivity_data_2.csv` is a separate **3,567-row** table, not the modelling
  set. The diary's "3,568" is therefore misleading if read as the model table.
- Labels drift between notebooks: "active" is sometimes IC50 ≤100 nM, sometimes
  ≤1,000 nM, and sometimes pIC50 ≥6. "Inactive" sometimes includes pIC50 6–7.
  Every notebook must state its exact label rule; do not compare metrics across
  different label definitions.
- Action: keep one canonical 645-row table with a documented `pIC50` and `label`
  definition; record the ChEMBL snapshot date and filtering script.

## 2. Traditional ML and the base paper

- Arjun sir's traditional-ML track used Morgan r3/512 and MACCS fingerprints with
  RF/SVM/KNN-style classifiers.
- Best reported RF baseline: test AUROC **0.960**, F1 **0.943**, MCC **0.876**;
  SVC reached AUROC **0.963**.
- COCONUT funnel: 695,119 → **154,647** rows in this repository. The paper says
  154,627; the 20-row difference is canonicalisation drift.
- Screening: 3,713 predicted active; **116** compounds at RF probability >0.6.
- The paper's leads `CNP0492204.2` and `CNP0385629.2`, docking scores −10.0/−9.6
  kcal/mol, and MD/MM-GBSA results are **not reproduced in this repository**.
- The strongest defensible result in the project is still the traditional RF baseline,
  not the later GNN screening claims.

## 3. First GNN phase (Parts 9–10)

- Scratch GCN: test AUROC **0.952**, F1 **0.923**, MCC **0.830**.
- GIN: test AUROC **0.947**, F1 **0.913**, MCC **0.807**; 5-fold CV AUROC
  **0.937 ± 0.017**.
- The "MolCLR pretrained" GIN was not real pretrained fine-tuning: only **28/63**
  layers matched, so the model was effectively trained from scratch.
- GIN y-randomisation reached **0.822**, so it is not a clean honesty check.
- Do not describe this phase as successful pretrained-GNN transfer.

## 4. Screening collapse and the architecture chase

- COCONUT screening called **54.1%** of the library active for GCN, close to the
  training prior, and **25.2%** for GIN.
- Expanding to 4,441 ChEMBL records (4,103 active / 338 inactive, 92% active) made
  prior collapse worse.
- AttentiveFP reached test AUROC **0.924**; hybrid GCN+GAT reached **0.956**;
  Part 14 added a soft-vote ensemble.
- Higher test AUROC did not translate into useful screening. The bottleneck was domain
  shift and prior collapse, not another architecture.

## 5. Part_16 and Part_17

- `Part_16/` (FP-GNN hybrid: Morgan 2048 + MACCS 167 + 3-layer GCN, EF/BEDROC,
  random/scaffold splits) exists only in the Arjun diary; there is no folder or commit
  in this repository.
- Part_17 v1 regression pretraining failed badly: best validation R² **−0.28**,
  final **−0.788**, RMSE **1.92**.
- Classification fine-tuning in the same phase reported **0.979 vs 0.970** random-test
  AUC and **930 vs 6,264** hits at p>0.85, despite the failed regression. Do not quote
  those hit counts as validated.
- The scaffold "1.000 AUC" came from only **134 compounds / 60 actives** and is
  uninformative.
- Later masked-atom work: random-test **0.9723 pretrained vs 0.9733 scratch** — no
  gain. It silently fell back to the 645 set as "ChEMBL" because the 4,441-row file
  (`P4441`) was missing.
- Masked-atom pretraining improved only **+0.016 AUC** (0.935 vs 0.919) and
  **+0.029 R²**; T1=1 and predicted-active rate 0.373.
- The earlier "test-selected" **0.938 AUC / −0.241 R²** result must not be used.

## 6. Part_18 triage

- The 78-dim graph builder is intentional in Part_18.
- GCN: **0.9726 AUC**; GIN: **0.9465 AUC / 0.883 F1 / 0.738 MCC**.
- Temperature was fitted on the held-out 129 test compounds. Part_18's calibrated
  probabilities are therefore leakage-contaminated if reused for claims.
- Part_18 `.pth` checkpoints are **not** warm-started in Part_20 for this reason.

## 7. Part_20 honest regression and screening

- Split: 464 train / 52 validation / 129 test, deterministic scaffold split; test
  touched once.
- Test R²: Ridge **0.410**, GCN **0.649**, GIN **0.671**, ensemble **0.666**,
  AttentiveFP **0.453**, HFooladi GIN **0.584**. ChemBERTa was skipped because
  `transformers` was unavailable in that run.
- Test bias **−0.525 pIC50**; calibration slope **0.696**; low-pIC50 bin bias
  **−1.059**; worst error **3.43 pIC50** (`CHEMBL3703675`).
- Screening library: 154,587 rows. Only **886 (0.57%)** were in the legacy AD at
  max Tanimoto ≥0.40 to the 464 training compounds.
- Legacy tiers: T1=103, T2=24,901, T3=129,583. Top T1 included
  `CNP0423424.1` 8.42, `CNP0071615.1` 8.30, `CNP0234138.1/.2` 8.29.
- Raw predictions of 10.2–10.5 were all out of domain.
- Screen-to-train median similarity was **0.217**, so roughly **99.4%** of COCONUT is
  extrapolation. Collapse stereoisomer duplicates before any assay.
- The 47/103 T1 compounds are a threshold artifact, not validated hits.

## 8. Post-diary refactor

- Commit `ecf1521` (2026-10-24) added AttentiveFP/HFooladi/ChemBERTa voters,
  validation-only calibration, simple mean of GCN+GIN+AttentiveFP, stricter AD and
  active-margin gates, model-agreement filtering, parent/scaffold deduplication, and
  `shortlist_low_confidence.csv`.
- Old T1/T2/top-100 outputs are demoted to diagnostics.
- Notebook cells 19–23 were **unexecuted** at the time of that audit, and the old
  `run_manifest.json` / `hits_T1.csv` predate the refactor. Do not treat stale outputs
  as current.

## 9. Silent scientific bugs that were fixed

- `GroupShuffleSplit(test_size=0.20)` produced **391/44/210** (33% test) instead of
  464/52/129.
- Test-vs-all-645 Tanimoto included self-matches (`maxT=1.000`); fixed to compare
  against the 464 training compounds only.
- Y-randomisation shuffled scaled deltas after fitting, producing fake R² 0.29–0.49;
  fixed by shuffling raw pIC50 and refitting Ridge/scaler.
- The DL featurizer sliced 124 atom features to 78, invalidating old checkpoints.
- `AUGMENT_N=2` duplicated isomorphic graphs.
- `FULL_TRAIN=True` "Test R²" 0.902/0.884/0.860 was really a training monitor.
- Classification softmax saturation produced ~62k near-identical "hits".
- The GCN is stereo-blind: predictions were identical across stereoisomers to three
  decimals.

## 10. Environment and runtime failures

- Colab vs Kaggle path differences (`/content`, `%cd`, clone fallback).
- GitHub PAT/auth failures.
- `rdkit-pypi` / CUDA dependency clashes.
- `NameError: DirectedMPNN`.
- XO `gnn_probs` and `gnn_test_metrics(..., thr=...)` issues.
- Function shadowing in `len(diverse)` / `xo.head`.
- Missing `dmpnn_losses` / `HAVE_CB`.
- `KeyError: medchem_pass` / `pred_chemberta` from skipped or emptied cells.
- Fixes: path splits, skipping auth cells, conda-forge RDKit, guarded columns/asserts,
  a restored ChemBERTa cell, explicit run-order messages.
- Kaggle CPU-100%/GPU-idle was fixed with multiprocessing, batch 512/768, and dual-GPU
  threads.

## 11. Claims not to repeat

- "MolCLR fine-tuning" — it was scratch training.
- GIN y-randomisation 0.24–0.82 as proof of signal.
- Scaffold 1.000 AUC as evidence.
- Part_17 0.979 / 930 hits without the failed regression and 645-row fallback.
- Masked pretraining as clearly "better" from +0.016 AUC.
- FULL_TRAIN 0.902/0.884/0.860 as test R².
- 62k / 24,683 hits as real hits.
- Part_18 calibrated probabilities.
- The 116 RF hits as validated inhibitors.
- Exact 154,627 as the repo row count (repo: 154,647).

## 12. Dead ends and open questions

- Part_16 never landed.
- `Part_12` is a naming collision: the AttentiveFP folder vs the expanded-GNN diary
  entry.
- XO side experiment: Morgan RF 0.955 vs GNN 0.71–0.75, all GNN MCC 0.000, and
  0 diverse hits.
- Docking into 4HG7 was never run; there is no wet-lab validation.
- Assay-type audit, scaffold-singleton fix, and repeated CV for all models remain
  deferred.

## 13. New-contributor rules

1. Start from the canonical 645-row table and document the label rule.
2. Never mix 78-dim Part_18 checkpoints with 124-dim DL checkpoints.
3. Use deterministic scaffold splits; early-stop on validation only.
4. Touch the test split once.
5. Compute AD against the 464 training compounds only.
6. Treat Tanimoto, active-neighbour, margin, and model-agreement gates as mandatory.
7. Report enrichment and confidence bands, not only R².
8. Call the shortlist low-confidence hypotheses, never hits.
9. Keep originals untouched; work in copies under `organized/`.
10. Do not revive a failed branch without a clean, documented rerun.
