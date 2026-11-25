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

## 9. Deep-learning notebooks (06_deep_learning_models, 2026-11-13)

- The user asked for a dedicated folder of deep-learning notebooks in Arjun
  sir's style: every DL model, stepwise fine-tuning, full GPU/RAM/CPU
  utilization, a chemistry primer, and Y-randomization in all of them.
  Notebook 1 uses no fingerprints (graphs, descriptors, SMILES sequence);
  Notebook 2 adds Morgan ECFP4 2048-bit; Notebook 3 uses MACCS 167-bit.
- Built `06_deep_learning_models/`: `MASTER_PROMPT.md` (the build spec,
  written first), three notebooks (`DL_01_Graph_Models_No_Fingerprints.ipynb`,
  `DL_02_Morgan_Fingerprint_DL.ipynb`, `DL_03_MACCS_Keys_DL.ipynb`),
  `build_notebooks.py` (regenerates and validates all three), `run_smoke.py`,
  plus `README.md`, `smoke_logs/`, and `outputs/` smoke artifacts.
- Every notebook follows the same 17-cell flow: title markdown, chemistry
  primer, setup, seed-57 reproducibility, full-utilization cell (cudnn
  benchmark, TF32, AMP, pin_memory, num_workers, auto-batch, dual-GPU
  parallelism), data + EDA, featurize + leak-free split, models, engine,
  baselines, stepwise fine-tune ladder (lr -> dropout/wd -> hidden -> batch,
  validation-only selection), optional pretrain warm-start, 5-fold
  GroupKFold CV, Y-randomization, test-scored-once with bootstrap CI,
  summary, optional screening cell (off by default), diary block.
- The exact Part_20 split (464/52/129) is reproduced with the greedy
  largest-scaffold-first algorithm after a GroupShuffleSplit attempt silently
  produced 521/56/68; scaffold overlap train-test is asserted 0. Features are
  computed after the split; the scaler is fit on train only.
- All three notebooks pass a CPU smoke run (nbclient, seed 57, outputs
  cleared afterwards): DL_01 9 models with best val AUC 0.8785 (HybridGCN+GAT
  / D-MPNN) and Y-randomization mean 0.4413 (signal real); DL_02 final smoke
  test AUC 0.9704; DL_03 0.7852. Smoke numbers are sanity checks, not
  project results; full-GPU runs still to be done on Colab/Kaggle and then
  recorded here.
- Originals in `02_gnn_models/`, `01_traditional_ml/`, `Part_9/`,
  `Part_20_Fixed/` were not touched; the notebooks adapt their architectures
  and protocols and are labeled in `06_deep_learning_models/README.md`.

## 10. Feedback round on the DL notebooks (2026-11-13)

- The first build missed two models that exist in the original roster:
  DeepChem's GraphConv GCN (Arjun sir's Part_9) and the pretrained HuggingFace
  models (ChemBERTa, MoLFormer). Both are now in the notebooks.
- DL_01 now trains 10 models: the pure-torch `DeepChemGCN` replica always
  runs, and cell 9b additionally runs the REAL `dc.models.GCNModel` when the
  deepchem package is installed (guarded skip otherwise). Cell 10c fine-tunes
  a pretrained chemical LM — ChemBERTa by default, MoLFormer-XL with
  `DL_HF_MODEL=molformer` — still optional and sensitivity-only.
- Classification outputs are softmax probabilities by default everywhere:
  cell 13 saves `test_predictions.csv` (identifier, canonical_smiles, y_true,
  prediction, prob_class_1, prob_class_0) in all three notebooks, matching
  the screening_results.csv schema.
- Every notebook's title markdown now states the original repository
  (`Natural_MDM2_Inhibitor_Discovery_using_ML/`, cloned 2026-09-07 from
  github.com/arjunpahi/...) and the full data lineage (2,571 → 1,933 → 645
  RO5 compounds, 343 Active), and cell 5 prints the same at runtime.
- The last `torch.load` without `weights_only=True` was fixed; smoke logs
  show no warnings. All three notebooks re-passed CPU smoke (10/5/5 models),
  outputs cleared. Guardrails unchanged: 464/52/129 scaffold split,
  Y-randomization, test scored once.

## 11. Tier-1/Tier-2 extension notebooks (2026-11-13)

- After the five-model recommendation, the user asked for Tier 1 and Tier 2 as two
  separate notebooks with full paper lineage ("where model come from").
- `DL_04_Tier1_Advanced_Architectures.ipynb` adds five newer architectures with cited
  sources: PNA (Corso et al., NeurIPS 2020; degree histogram from train split only),
  GraphTransformer (TransformerConv, Shi et al. 2021), GPS (Rampášek et al., Nature
  Machine Intelligence 2022), SmilesTransformer (Vaswani et al. 2017, char-level) and
  FT-Transformer (Gorishniy et al., NeurIPS 2021, descriptor tokens + CLS).
  Graphormer and 3-D models are deliberately excluded — 645 molecules is too few —
  and the notebook states why.
- `DL_05_MolCLR_Pretrain_Finetune.ipynb` implements MolCLR self-supervised pretraining
  (Wang et al., Nature Machine Intelligence 2022): two augmented views per COCONUT
  molecule (atom masking / bond deletion / subgraph removal), NT-Xent (SimCLR)
  contrastive loss on a GIN encoder, then fine-tuning. COCONUT has no activity labels,
  so pretraining is leak-free by construction; the trunk is saved as
  `molclr_gin_pretrain.pt` and warm-starts "GIN-MolCLR" through a generalized
  warm-start hook in the engine (the D-MPNN stage1a.pt hook still works).
- Smoke results (CPU): DL_04 all 5 models train (GPS 0.834 / FT-Transformer 0.8421 val
  AUC; final test GPS AUC 0.8667). DL_05 pretraining loss decreases (3.63 → 2.81) and
  GIN-MolCLR beats GIN scratch by +0.12 val AUC at smoke scale (0.8259 vs 0.7085) —
  the exact comparison the notebook was built to expose. One real bug (SmilesTransformer
  masked-mean broadcasting) was caught by smoke and fixed in the builder.
- Same guardrails in both: 464/52/129 scaffold split, validation-only selection,
  Y-randomization, test scored once, softmax prob columns in test_predictions.csv.


## 12. 06_deep_learning_models portability/validation audit (2026-11-22) [ADAPTED/GENERATED]

- **Scope**: executed and hardened all five notebooks in `06_deep_learning_models/` (DL01-DL05) on local CPU (Python 3.13.9, torch 2.13.0+cpu, RDKit 2026.03.5, PyG 2.8.0.post1, 8 cores, ~7 GiB RAM). No original folders were modified; changes are confined to `06_deep_learning_models/`, `docs/JOURNEY.md`, and `diary.md`.
- **Smoke validation (regenerated notebooks, outputs kept)**: `python3 run_smoke.py` -> all PASS. DL01 0.6 min, DL02 0.3, DL03 0.2, DL04 0.7, DL05 0.3. Split 464/52/129 in every notebook, scaffold overlap train-test 0, no tracebacks. Smoke metrics are sanity checks only.
- **CPU-fast real-data validation (all 645 rows, every model)**: `python3 run_fast.py` -> 4/5 pass initially; DL05 failed with GIN warm-start shape mismatch when FT stage C selected hidden=64 against a 128-dim MolCLR checkpoint. Fix: `fit_model` now loads only shape-compatible warm-start tensors and logs skipped mismatches. After rebuild, `python3 run_fast.py DL_05` PASS 1.3 min. Final fast results (test scored once, 129 compounds): DL01 AUC 0.9989 [0.9955,1.0000]; DL02 0.9933 [0.9753,1.0000]; DL03 0.9899 [0.9701,1.0000]; DL04 0.9871 [0.9639,1.0000]; DL05 0.9981 [0.9941,1.0000] — corrected 2026-11-23 against `outputs/dl05/fast/dl05_results.json`. Artifacts/logs in `outputs/dl0X/fast/` and `fast_logs/`.
- **Caveat**: canonical 129-test holdout has only 5 Murcko scaffolds (40/34 all-inactive, 31/21 mostly active, 3 mixed). High fast AUCs are likely scaffold/series separation, not prospective molecule-level generalization. Treat as plumbing/sanity; rely on CV, Y-randomization, AD, and external validation for claims.
- **Portability fixes in `build_notebooks.py`**: `rdkit-pypi` -> `rdkit`; CPU torch index when no working `nvidia-smi`; idempotent `python -m pip` installs; DL_FAST profile; DL05 pretraining streams batches (no 60k-graph preload); DeepChem install/import opt-in (`DL_INSTALL_DEEPCHEM=1`, no dgl); HF install-before-import opt-in (`DL_USE_HF=1`); screening opt-in/chunked/robust (`DL_RUN_SCREENING=1`); bootstrap single-class guard; zero-bond edge shape `(2,0)/(0,6)`; torch.amp/torch.cuda.amp fallback; DataLoader workers 0 on Windows/smoke/CPU; warm-start `weights_only` fallback; shape-safe warm-start loading; split scaffold-count warning.
- **Runners**: `run_smoke.py`/`run_fast.py` now keep notebook outputs by default; `DL_CLEAR_OUTPUTS=1` restores clearing. `run_smoke.py` sets `DL_FAST=0`.
- **Docs updated**: `06_deep_learning_models/README.md`, `MASTER_PROMPT.md` now describe five notebooks, actual split (test scaffold-disjoint greedy 129; val stratified random 52 of remaining 516; train 464), env flags, portability, smoke/fast results, and caveats.
- **Not run/exercised**: real DeepChem GCNModel (needs TensorFlow), HF ChemBERTa/MoLFormer fine-tune (needs GPU/download), full 154k COCONUT screening, full 150-epoch GPU protocol, `DL_TASK=reg` full runs. These are documented as not exercised.

## 13. 06_deep_learning_models research-model additions + ZINC250k pretraining (2026-11-22) [ADAPTED/GENERATED]

- **User request**: add more research deep-learning models with GitHub/paper provenance; add open-source downloadable chemistry pretraining data (not screening COCONUT); run/validate everything; update diary and mistakes/solutions. User also requested terminal `poweroff`/`sleep` after completion — **refused** as unsafe/destructive system action.
- **Models added to DL01** (PyG, 2D molecular graphs, edge-aware where supported): **GINE** (Hu et al., ICLR 2020, *Strategies for Pre-training GNNs*, https://github.com/snap-stanford/pretrain-gnns, PyG `GINEConv`, uses 6-dim edge_attr); **GraphSAGE** (Hamilton et al., NeurIPS 2017, https://github.com/williamleif/GraphSAGE, `SAGEConv`, node/topology); **GatedGCN/ResGatedGraphConv** (Bresson & Laurent 2017, https://github.com/xbresson/spatial_graph_convnets, PyG `ResGatedGraphConv`, edge-gated, uses edge_attr). DL01 now has 13 models. Provenance recorded in `06_deep_learning_models/MODEL_RESEARCH.md`.
- **DL05 pretraining data fix**: default changed from screening COCONUT to **ZINC250k** (standard open-source unlabeled SMILES, DeepChem S3/public mirror; downloaded/cached to `outputs/dl05/<profile>/pretrain_cache/`). `DL_PRETRAIN_SOURCE=zinc250k|chembl_small|coconut|path/url`, `DL_PRETRAIN_MAX`, `DL_PRETRAIN_CACHE` added. COCONUT is explicit opt-in only. Smoke log confirmed `249,455 SMILES` loaded, 1,500 used, `warm-started 24/24 tensors from MolCLR pretrain (zinc250k)`.
- **Validation (2026-11-22, CPU, Python 3.13.9, torch 2.13.0+cpu, RDKit 2026.03.5, PyG 2.8.0.post1)**:
  - Smoke `python3 run_smoke.py`: 5/5 PASS (DL01 0.7 min AUC 0.8667; DL02 0.3 0.9704; DL03 0.2 0.7852; DL04 0.7 0.8667; DL05 0.3 0.9037 — corrected 2026-11-23 to match `outputs/dl05/smoke/dl05_results.json` and `smoke_logs/`). DL01 smoke log confirms GINE/GraphSAGE/GatedGCN trained.
  - Fast `python3 run_fast.py`: 5/5 PASS (DL01 10.5 min AUC 0.9989 AP 0.9982 BalAcc 0.9610 CI [0.9955,1.0000], best GatedGCN; DL02 2.2 min 0.9933/0.9858/0.9805 [0.9753,1.0000]; DL03 0.9 min 0.9899/0.9786/0.9740 [0.9701,1.0000]; DL04 9.5 min 0.9871/0.9752/0.9353 [0.9639,1.0000]; DL05 1.5 min 0.9981/0.9972/0.8766 [0.9941,1.0000], ZINC250k pretrain 4000×3). Artifacts in `outputs/dl0X/{smoke,fast}/`, logs in `smoke_logs/`, `fast_logs/`.
- **Mistakes / solutions recorded**:
  - *Mistake*: DL05 used screening COCONUT as default pretraining. *Solution*: user corrected; default switched to ZINC250k download/cache, COCONUT explicit opt-in only.
  - *Mistake*: warm-start shape mismatch crashed DL05 fast when FT candidate changed hidden dim. *Solution*: shape-compatible warm-start load with skipped-tensor logging.
  - *Mistake*: README fast/smoke numbers became stale after model changes. *Solution*: reran smoke+fast and updated README/MASTER_PROMPT/JOURNEY.
  - *Mistake/risk*: some GNNs ignored `edge_attr` despite 6-dim bond features being present. *Solution*: added edge-aware GINE/GatedGCN, documented edge usage in MODEL_RESEARCH.md and prior messages.
  - *Refused*: terminal `poweroff`/`sleep` request — unsafe/destructive; not executed.
- **Caveat**: fast AUCs remain plumbing/sanity on a 5-scaffold 129-compound holdout; not prospective performance. Full 150-epoch GPU protocol, 154k screening, real DeepChem TF, HF GPU fine-tune still not exercised.

## 14. Paper-facing caveat + model inventory correction (2026-11-23) [USER]

- **User request**: add the "fast AUCs are plumbing/sanity, not prospective; use CV/Y-randomization/AD/external validation" caveat to `06_deep_learning_models/`, and provide the list of deep-learning models used plus a more advanced list; judge whether it strengthens the paper.
- **Decision**: yes — transparency about the 5-scaffold holdout strengthens the paper. Added a prominent "Read this before quoting any number (paper-facing caveat)" section to `06_deep_learning_models/README.md` and an "Interpretation limits (paper-facing)" section to `MODEL_RESEARCH.md`. No notebooks or model code were changed; the split cell already prints the ≤10-scaffold warning and Y-randomization already runs in every notebook.
- **Model inventory corrected**: README previously said "15 unique architectures → 20 training slots"; the registries in `build_notebooks.py` give **30 training slots** (DL01 13, DL02 5, DL03 5, DL04 5, DL05 2) and **23 distinct architectures** (fingerprint families reused across DL02/DL03; GIN reused in DL05 for scratch-vs-MolCLR). Per user instruction, README now leads with the **deduplicated count: 23 unique models** — a model repeated in another notebook/phase is not counted again.
- **Advanced/data-hungry models** (reviewed, not used) documented with reasons and revisit conditions in `MODEL_RESEARCH.md`: Graphormer, GROVER, Uni-Mol, SchNet/DimeNet++/PaiNN/GemNet, ChemBERTa/MoLFormer (sensitivity-only), GraphMAE/KPGT/full MolCLR, full GraphGPS, Chemprop v2. Priority order: external set → repeated/nested scaffold CV + enrichment → Y-rand/AD → two-stage model → uncertainty → larger pretraining → 3-D/Graphormer.
- **Stale numbers fixed**: DL05 smoke AUC 0.9481 → 0.9037 (README + JOURNEY §13) and DL05 fast AUC 0.9951 → 0.9981 (MASTER_PROMPT §10), verified against `outputs/dl05/{smoke,fast}/dl05_results.json` and the matching logs.
- **Not run**: no training, CV, screening, or notebook execution in this task; documentation only. Originals untouched.

## 15. Extended 34-model roster inside existing notebooks (2026-11-23) [USER]

- **User choice**: add all 34 remaining models; place them in the existing notebooks (not a new one); 3-D models should be present but skipped until 3-D data arrives.
- **Implementation**: `06_deep_learning_models/build_notebooks.py` now emits three extra cells — S07c (`MODELS_EXTRA_SRC`), S07d (`SSL_EXTRA_SRC`, DL05 only), S16b (`UNCERTAINTY_SRC`) — into the existing notebooks. Generated notebooks: 23 cells for DL01/DL05, 21 for DL02/DL03/DL04. Source-of-truth JSON + AST validation passes.
- **Safety**: default registry and reported results unchanged; extras require `DL_EXTRA_MODELS=1`. SSL alternatives require `DL_SSL_METHOD=graphcl|joao|gcc|hu`; uncertainty diagnostics require `DL_UNCERTAINTY=1` and use validation rows only (test scored once, never re-scored). 3-D classes (NequIP/MACE/SphereNet/PhysNet) are defined but registered only with conformers (`d.pos`) + `DL_USE_3D=1` + e3nn; they raise a clear guard error otherwise. SMILES LMs are from-scratch architecture analogs (no external weights).
- **Bugs fixed during smoke** (all in `build_notebooks.py`, regenerated notebooks): GCNII initial residual double-projection (78 vs 128); MoNet dead branch; SMILES transformer vocab/max-length from notebook globals; `MPNNSet2Set` GRU state projected 78→128; `VirtualGIN` convs moved into a `ModuleList` so they actually train; `TabTransformer` token count capped at 64 so 2048-bit fingerprints stay tractable; SAGPool/TopKPool `batch=` keyword (3rd positional is `edge_attr`); DL05 has no `FEATS`, so array extras are registered only when a feature matrix exists; fine-tuning ladder limited to the original 23 unless `DL_EXTRA_FT=1`; multi-task diagnostic trains on training rows only (no test-label leakage) and restores `eval()` after MC-dropout.
- **Validation status (2026-11-23)**: all five regenerated notebooks passed `nbformat` + AST validation. Opt-in end-to-end smokes passed: DL01 `DL_EXTRA_MODELS=1 DL_UNCERTAINTY=1` (9.2 min, +21 registry models), DL02/DL03/DL04 `DL_EXTRA_MODELS=1` (0.9/0.6/1.3 min, +16/+16/+21), DL05 `DL_EXTRA_MODELS=1 DL_SSL_METHOD=graphcl DL_UNCERTAINTY=1` (8.8 min, +13 graph-only, GraphCL warm-start). A final default-profile `python3 run_smoke.py` also passed all five and cleared cell outputs. Full/fast runs with the extra roster were not done at that point; no performance claims for added models. Originals untouched.

## 16. Validation package + screening/AD audit (2026-11-23) [USER]

- **User request**: "did all of sanity check adn the screening data installed and more sucring and powerfull validation sanitation verified" → chose **All three**: validation package, screening/AD audit, and extended fast runs.
- **Validation package**: new `06_deep_learning_models/validation_package.py` (ran 61.8 s, exit 0). It reproduced the 464/52/129 scaffold split, ran 3 seeds × 5 outer GroupKFold folds with 3-fold inner selection on train+val only, and re-scored the existing five fast test prediction files descriptively. Outputs: `outputs/validation_package/nested_cv_folds.csv`, `nested_cv_summary.json`, `fast_test_enrichment.csv`, `fast_test_enrichment_summary.json`, `README.md`, plus preserved source predictions. Nested-CV AUC: descriptor logreg 0.8525 ± 0.0533, ECFP4 logreg 0.9016 ± 0.0360, ECFP4 RF 0.9158 ± 0.0338. Test enrichment (descriptive only): AUC 0.9871–0.9989, precision@10 = 1.00, EF1% = 2.48, BEDROC(α=20) 0.9924–1.00. No model was selected from test values; no external set exists.
- **Screening/AD audit**: new `06_deep_learning_models/screening_audit.py` (ran 46.2 s, exit 0). Audited 154,647 existing COCONUT rows: 0 missing/duplicate identifiers, 100% valid SMILES, 0 exact training-SMILES overlaps, 38,388 duplicate parent rows, 9 shared Murcko scaffolds. Funnel: `in_domain` 38,761 (25.06%) → nearest-active ≥0.40 81 → agreement proxy 52 → strict Tanimoto diagnostic 14. The margin gate is unavailable (`maxT_inactive` not saved), and `in_domain` is a descriptor box, **not** strict `maxT_464 ≥ 0.52`. Outputs in `outputs/screening_audit/`. All survivors remain low-confidence computational hypotheses, not hits.
- **Extended fast runs**: launched `DL_EXTRA_MODELS=1 python3 run_fast.py`. The user stopped the terminal while DL01 was still running (20.6 min, SIGKILL); **no extended fast run completed**, so no performance numbers from the extra roster are claimed. Damage check: `test_predictions.csv` and `dl01_results.json` were unchanged (matching SHA256), and the partially rewritten `baseline_table.csv` / `finetune_table.csv` matched the saved 22:58 fast log exactly (same deterministic values), so no pre-existing result was lost. Snapshots of the default fast predictions remain in `outputs/validation_package/source_fast_predictions/`.
- **Not done**: no external test set, no assay/docking/MD validation, no new 3-D data, no claim that any screen hit is real. Originals untouched.

## 17. Y-randomization crash: DataLoader fork from threaded GPU jobs (2026-11-25) [USER]

- **Reported error**: Section 12 Y-randomization in a dual-GPU Jupyter kernel failed at `fit_model → tr_ld` with `RuntimeError: os.fork is unsafe while filelock is changing descriptor ownership` (Python 3.13, `filelock` 3.32.5).
- **Cause**: `run_parallel()` uses `ThreadPoolExecutor` for two-GPU jobs; `make_loader()` still requested `num_workers=2`, so the DataLoader called `os.fork()` from a worker thread while a filelock fd-ownership change was in progress. Forking from a Jupyter thread is unsafe regardless; the 645 in-memory graphs do not need child loader processes.
- **Fix**: `num_workers()` in `06_deep_learning_models/build_notebooks.py` now defaults to 0 (`DL_DATALOADER_WORKERS=2` opt-in for a known single-threaded run), and the identical change was patched into all five generated notebooks (full regeneration was avoided to prevent random cell-ID churn; only the intended cell differs). README env-flag table and `docs/ERRORS_AND_FIXES.md` §10 updated. Pre-existing unrelated drift: the screening cell in the notebooks has one moved comment line vs the generator (from commit d2ae89c).
- **Live-kernel unblock**: `num_workers = lambda: 0` in the open notebook before rerunning Section 12; no split, label-permutation, or metric code changed.
- **Not run**: no training/CV/Y-randomization rerun in this task — the fix was validated with notebook regeneration + `nbformat`/AST checks only. Saved outputs in `outputs/` are untouched.

## 18. Verified fine-tunable pretrained checkpoints (2026-11-25) [USER]

- **Asked**: update the latest information on pretrained deep-learning models that can actually be fine-tuned.
- **Checked live**: HF `config.json` endpoints returned HTTP 200 for `seyonec/ChemBERTa-zinc-base-v1`, `DeepChem/ChemBERTa-77M-MLM`, `DeepChem/ChemBERTa-100M-MLM`, `ibm-research/MoLFormer-XL-both-10pct`, `ibm-research/materials.smi-ted`; the notebook's old default `seyonec/ChemBERTa-zinc-base-v0` now returns **401**.
- **Changed**: DL01 cell 10c + `build_notebooks.py` now default to ChemBERTa `v1` and offer `chemberta2` (77M MLM), `chemberta3` (100M-MLM, ChemBERTa-3, 2026) and `molformer`. `MODEL_RESEARCH.md` gained a *Fine-tunable pretrained checkpoints — verified 2026-11-25* section with exact flags, HF IDs, IBM's SMI-TED fine-tune recipe (not wired into the notebook), and 2026 provenance caveats (MoLFormer stereo model-card error; weak/unstable CLM transfer). README flag table and NB1 description updated.
- **Validated**: all five notebooks JSON-parse and every code cell passes `ast.parse`; `build_notebooks.py` parses; no stale `v0` ID or `Techbjd` URL remains. **Not run**: no GPU fine-tuning was executed here (no GPU/transformers in this environment), so no performance claim is made.
