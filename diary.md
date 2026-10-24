# Diary: Natural MDM2 Inhibitor Discovery Using ML

**Cloned from:** https://github.com/arjunpahi/Natural_MDM2_Inhibitor_Discovery_using_ML.git
**Date:** 2026-09-07

---

## Project Overview

A Jupyter notebook pipeline (10 parts) for discovering natural product inhibitors of the MDM2 protein (a cancer drug target) using machine learning. It starts from raw ChEMBL bioactivity data, builds and evaluates ML models, and screens ~154k natural products from the COCONUT database to identify candidate inhibitors.

**Parts 1–8:** Traditional ML (fingerprints + 20 classifiers including RF, SVM, KNN)
**Part 9:** Graph Neural Network from scratch (DeepChem GraphConvMol)
**Part 10:** Pretrained GNN fine-tuning (MolCLR GIN on MDM2 data)

---

## Repository Structure

```
Natural_MDM2_Inhibitor_Discovery_using_ML/
├── Data_Extraction_1.ipynb              # Part 1: ChEMBL data extraction
├── Datase_Filteration_2.ipynb           # Part 2: Lipinski RO5 filtering + classification
├── Exploratory-Data-Analysis-MDM2.ipynb # Part 3: Statistical tests & visualizations
├── MACCS_rdkit_part1.ipynb              # Part 4a: MACCS fingerprint generation
├── Morgan_&_ECFP_part2.ipynb            # Part 4b: Morgan/ECFP fingerprint generation
├── Data_Exploration_With_Fingerprints_5.ipynb # Part 5: PCA, t-SNE, similarity analysis
├── Maccs_Machines_Models.ipynb          # Part 6a: 20 ML classifiers on MACCS features
├── Morgan_Machines_Models.ipynb         # Part 6b: 20 ML classifiers on Morgan features
├── Coconut_data_extraction.ipynb        # Part 7: COCONUT natural product extraction
├── Final_screening_of_coconut.ipynb     # Part 8: Virtual screening (154k → 116 candidates)
├── Part_9_GNN_DeepChem.ipynb            # Part 9: GraphConvMol (GNN from scratch)
├── Part_10_Pretrained_GNN.ipynb         # Part 10: Pretrained GIN fine-tuning (MolCLR)
├── requirements.txt                     # Root dependencies (GPU/CUDA + CPU)
├── random_forest_model.pkl              # Serialized trained Random Forest model
├── Part_1/bioactivity_data_2.csv        # 3,568 compounds with pIC50
├── Part_2/ro5_properties_filtered.csv   # 645 compounds passing Lipinski RO5
├── Part_6/performance_morgan.csv        # Morgan fingerprint model metrics
├── Part_6/performance_maccs.csv         # MACCS fingerprint model metrics
├── Part_6/merged_metrics_by_model.csv   # Side-by-side MACCS vs Morgan comparison
├── Part_8/screening_results.csv         # 154,648 COCONUT compounds screened
├── Part_8/filtered_compounds.csv        # 116 predicted MDM2 inhibitors
├── Part_8/mdm2_smiles.txt              # 116 confirmed SMILES for downstream use
├── Part_4/requirements1.txt             # Part 4 dependencies
├── Part_4/requirements2.txt             # Part 4 dependencies
├── Part_6/requirements1.txt             # Part 6 dependencies
├── Part_6/requirements2.txt             # Part 6 dependencies
└── zip_file/                            # PDB structure archives (4HG7, etc.)
```

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab
- (Optional) NVIDIA GPU + CUDA for Part 1 & 4 GPU-accelerated code

### Step 1: Create a virtual environment

```bash
cd Natural_MDM2_Inhibitor_Discovery_using_ML
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The root `requirements.txt` includes CUDA/GPU libraries (cudf-cu12, cuml-cu12, cupy-cuda12x). If running on CPU only, install the lighter per-part requirements instead:
> ```bash
> pip install -r Part_6/requirements1.txt   # CPU-only alternative for Parts 5-8
> ```

### Step 3: Launch Jupyter

```bash
jupyter notebook
```

---

## How to Run the Pipeline (Step-by-Step)

Run the notebooks in order. Each part produces data files consumed by later parts.

### Part 1: Data Extraction (`Data_Extraction_1.ipynb`)
- **What it does:** Queries ChEMBL for MDM2 bioactivity data (IC50 values, compound structures).
- **Input:** ChEMBL API (requires internet).
- **Output:** `Part_1/bioactivity_data_raw.csv` → `Part_1/bioactivity_data_2.csv` (3,568 compounds with SMILES, pIC50, bioactivity class).
- **Notes:** Skip the GitHub push/pull cells if you don't have a PAT token.

### Part 2: Data Filtering (`Datase_Filteration_2.ipynb`)
- **What it does:** Applies Lipinski Rule of 5 (MW ≤ 500, LogP ≤ 5, HBA ≤ 10, HBD ≤ 5). Classifies compounds as active (pIC50 > 6), inactive (< 5), or intermediate.
- **Input:** `Part_1/bioactivity_data_2.csv`
- **Output:** `Part_2/ro5_properties_filtered.csv` (645 compounds).
- **Output:** `Part_2/processed_data.csv`

### Part 3: Exploratory Data Analysis (`Exploratory-Data-Analysis-MDM2.ipynb`)
- **What it does:** Mann-Whitney U test, Chi-square test, box plots, violin plots, swarm plots, correlation heatmaps.
- **Input:** `Part_2/ro5_properties_filtered.csv`
- **Output:** Various `.png` visualization files.

### Part 4: Fingerprint Generation
- **Part 4a (`MACCS_rdkit_part1.ipynb`):** Generates 166-bit MACCS keys from SMILES using RDKit.
- **Part 4b (`Morgan_&_ECFP_part2.ipynb`):** Generates Morgan/ECFP circular fingerprints (radius 2, 2048-bit).
- **Input:** `Part_2/ro5_properties_filtered.csv`
- **Output:** Feature matrices (numpy arrays) saved as `.npy` files for Part 5-6.

### Part 5: Fingerprint Exploration (`Data_Exploration_With_Fingerprints_5.ipynb`)
- **What it does:** PCA, t-SNE dimensionality reduction. Tanimoto similarity analysis between active/inactive compounds.
- **Input:** Fingerprint feature matrices from Part 4.
- **Output:** PCA/t-SNE scatter plots, similarity distribution plots.

### Part 6: Model Training & Evaluation
- **Part 6a (`Maccs_Machines_Models.ipynb`):** Trains 20 ML classifiers on MACCS features with 5-fold stratified cross-validation.
- **Part 6b (`Morgan_Machines_Models.ipynb`):** Same 20 classifiers on Morgan features.
- **Input:** Fingerprint feature matrices + activity labels from Part 2.
- **Output:**
  - `Part_6/performance_morgan.csv` — metrics for all 20 models on Morgan features
  - `Part_6/performance_maccs.csv` — metrics for all 20 models on MACCS features
  - `Part_6/merged_metrics_by_model.csv` — side-by-side comparison
  - `random_forest_model.pkl` — serialized best model (Random Forest on Morgan)

**20 classifiers tested:**
1. LogisticRegression
2. SVC
3. NuSVC
4. DecisionTreeClassifier
5. ExtraTreeClassifier
6. RandomForestClassifier
7. ExtraTreesClassifier
8. AdaBoostClassifier
9. GradientBoostingClassifier
10. BaggingClassifier
11. KNeighborsClassifier
12. MLPClassifier
13. RidgeClassifier
14. SGDClassifier
15. Perceptron
16. PassiveAggressiveClassifier
17. LinearDiscriminantAnalysis
18. QuadraticDiscriminantAnalysis
19. GaussianNB
20. HistGradientBoostingClassifier

**Top performers (Morgan fingerprints):**

| Model                | ROC-AUC | F1    | MCC   | Accuracy |
|----------------------|---------|-------|-------|----------|
| RandomForest         | 0.942   | 0.906 | 0.813 | 0.913    |
| HistGradientBoosting | 0.942   | 0.903 | 0.807 | 0.910    |
| SVC                  | 0.941   | 0.908 | 0.817 | 0.914    |
| LogisticRegression   | 0.940   | 0.896 | 0.793 | 0.903    |

### Part 7: COCONUT Data Extraction (`Coconut_data_extraction.ipynb`)
- **What it does:** Downloads and preprocesses ~154k natural product structures from the COCONUT database.
- **Input:** COCONUT database (requires internet).
- **Output:** `Part_7/coconut_processed.csv`

### Part 8: Virtual Screening (`Final_screening_of_coconut.ipynb`)
- **What it does:** Applies the trained Random Forest model to predict MDM2 inhibitory activity for all 154k COCONUT compounds.
- **Input:** `Part_7/coconut_processed.csv` + `random_forest_model.pkl`
- **Output:**
  - `Part_8/screening_results.csv` — all 154,648 compounds with predictions and probabilities
  - `Part_8/filtered_compounds.csv` — **116 predicted MDM2 inhibitors**
  - `Part_8/mdm2_smiles.txt` — 116 SMILES strings for downstream docking/validation

### Part 9: GCN From Scratch (`Part_9_GNN_DeepChem.ipynb`)
- **What it does:** Trains a 3-layer Graph Convolutional Network directly on molecular graphs (atoms=nodes, bonds=edges). No pretrained weights — learns from scratch on 645 MDM2 compounds.
- **Input:** `ro5_properties_filtered.csv` (same data as Parts 2–8)
- **Architecture:** GCNConv(78→128) → GCNConv(128→128) → GCNConv(128→128) → mean+max pool(256) → Linear(256→64) → ReLU → Dropout → Linear(64→2)
- **Training:** 100 epochs, Adam lr=1e-3, batch_size=64, 5-fold stratified CV
- **Colab output — Test set (80/20 split):**

  | Metric | Value |
  |--------|-------|
  | Accuracy | 0.915 |
  | Precision | 0.892 |
  | F1-score | 0.923 |
  | Sensitivity | 0.957 |
  | ROC-AUC | 0.952 |
  | MCC | 0.830 |

- **Colab output — 5-fold CV mean:**

  | Metric | Mean ± Std |
  |--------|------------|
  | Accuracy | 0.896 ± 0.035 |
  | F1-score | 0.906 ± 0.030 |
  | ROC-AUC | 0.945 ± 0.019 |
  | MCC | 0.796 ± 0.065 |

- **Y-randomization:** Shuffled AUC range 0.276–0.593 (all ~random 0.5), confirming model learns real signal
- **Output files:** `performance_gcn_test.csv`, `performance_gcn_cv.csv`, `performance_gcn_shuffled.csv`, `gcn_scratch_model.pth`, comparison plots
- **Install:** `!pip install torch-geometric rdkit` (in Colab)

### Part 10: Pretrained GIN Fine-Tuning (`Part_10_Pretrained_GNN.ipynb`)
- **What it does:** Loads a GIN encoder (MolCLR architecture), adds a classification head, and trains on 645 MDM2 compounds. Note: MolCLR pretrained weights had architecture mismatch (28/63 layers matched), so model trained from scratch with MolCLR architecture.
- **Input:** `ro5_properties_filtered.csv`
- **Architecture:** 5-layer GIN (hidden=300) → mean+max pool(600) → Linear(600→128) → ReLU → Dropout(0.3) → Linear(128→2). Total: 465,091 params.
- **Training:** 100 epochs, Adam with differential LR (encoder: 1e-5, classifier: 1e-3), batch_size=64
- **Colab output — Test set (80/20 split):**

  | Metric | Value |
  |--------|-------|
  | Accuracy | 0.899 |
  | Precision | 0.850 |
  | F1-score | 0.913 |
  | Sensitivity | 0.986 |
  | ROC-AUC | 0.947 |
  | MCC | 0.807 |

- **Colab output — 5-fold CV mean:**

  | Metric | Mean ± Std |
  |--------|------------|
  | Accuracy | 0.876 ± 0.036 |
  | F1-score | 0.891 ± 0.029 |
  | ROC-AUC | 0.937 ± 0.017 |
  | MCC | 0.757 ± 0.069 |

- **Y-randomization:** Shuffled AUC range 0.244–0.822 (mostly ~0.3–0.5), confirming model learns real signal
- **Output files:** `performance_pretrained_gnn_test.csv`, `performance_pretrained_gnn_cv.csv`, `performance_pretrained_gnn_shuffled.csv`, `pretrained_gin_mdm2.pth`, comparison plots
- **Note:** Training from scratch with MolCLR architecture still produces competitive results (0.947 AUC vs RF's 0.960). The GIN architecture itself (5 layers, 300 hidden, mean+max pooling) is stronger than the simpler 3-layer GCN in Part 9.

---

## Final Results (Actual Colab Run)

### Traditional ML (Parts 1–8) vs GNN (Parts 9–10)

| Model | Type | ROC-AUC | F1 | Accuracy | MCC | Sensitivity |
|-------|------|---------|-----|----------|-----|-------------|
| RF (Morgan) — Part 6 | Fingerprint + RF | **0.960** | **0.943** | **0.938** | **0.876** | 0.957 |
| Soft Voting — Part 6 | FP + RF/SVM/KNN | **0.960** | 0.928 | 0.922 | 0.844 | 0.928 |
| GCN (scratch) — Part 9 | 3-layer GCN | 0.952 | 0.923 | 0.915 | 0.830 | 0.957 |
| GIN (MolCLR arch) — Part 10 | 5-layer GIN | 0.947 | 0.913 | 0.899 | 0.807 | **0.986** |

### 5-Fold CV Results

| Model | ROC-AUC (mean ± std) | F1 (mean ± std) |
|-------|---------------------|-----------------|
| GCN (Part 9) | 0.945 ± 0.019 | 0.906 ± 0.030 |
| GIN (Part 10) | 0.937 ± 0.017 | 0.891 ± 0.029 |

### Y-Randomization Validation
- **GCN (Part 9):** Shuffled AUC range 0.28–0.59 (all ~random) — model learns real signal
- **GIN (Part 10):** Shuffled AUC range 0.24–0.82 (mostly 0.3–0.5) — model learns real signal

### Key Finding
Both GNN models are **competitive with RF** (0.952/0.947 vs 0.960 AUC). The GCN from scratch nearly matches RF, and the GIN achieves the **highest sensitivity (0.986)** of all models — it catches 98.6% of actual actives, making it best for screening where missing actives is costly.

---

## Downstream Use of Results

The 116 candidate SMILES in `Part_8/mdm2_smiles.txt` can be used for:

1. **Molecular Docking** — Dock into MDM2 crystal structure (PDB: 4HG7) using AutoDock Vina, Glide, or GOLD.
2. **ADMET Prediction** — Assess drug-likeness, toxicity, bioavailability.
3. **Molecular Dynamics** — Run MD simulations to assess binding stability.
4. **Experimental Validation** — Select top candidates for in vitro MDM2 binding assays.

PDB structures for docking are available in `zip_file/rcsb_4hg7.zip` and `zip_file/rcsb_4hg72.zip`.

---

## Common Pitfalls

- **GitHub token errors:** Skip the push/pull helper cells in notebooks if you don't have a valid PAT.
- **Version clashes:** Use the part-specific `requirements.txt` (e.g., `Part_6/requirements1.txt`) instead of the root file if you encounter GPU/CUDA dependency issues.
- **RDKit install:** If `pip install rdkit-pypi` fails, try `conda install -c conda-forge rdkit`.
- **ChEMBL API:** Parts 1 and 7 require internet access to query ChEMBL/COCONUT APIs.

---

## Useful Commands

```bash
# Run a specific notebook from terminal
jupyter nbconvert --to script Data_Extraction_1.ipynb
python Data_Extraction_1.py

# Convert notebook to HTML
jupyter nbconvert --to html Data_Extraction_1.ipynb

# Install missing packages on the fly
pip install rdkit-pypi chembl-webresource-client pandas numpy scikit-learn matplotlib seaborn
```

---

## References

- MDM2 protein: Negative regulator of p53 tumor suppressor
- ChEMBL: https://www.ebi.ac.uk/chembl/
- COCONUT database: https://coconut.naturalproducts.net/
- PDB 4HG7: Crystal structure of MDM2
- **Yasir et al., 2025** — \"Integration of Deep Learning with Molecular Docking and MD Simulation for Novel TACE Inhibitors\" — *Future Pharmacology* 5(4):55. DOI: 10.3390/futurepharmacol5040055 (GraphConvMol pipeline reference)
- **Wang et al., 2020** — \"Self-Supervised Graph Transformer on Large-Scale Molecular Data\" (MolCLR) — *NeurIPS 2020*. Repo: https://github.com/yuyangw/MolCLR (Pretrained GIN reference)
- **DeepChem** — https://github.com/deepchem/deepchem (GraphConvModel, ConvMolFeaturizer)
- **PyTorch Geometric** — https://pyg.org/ (Graph neural network framework)

---

## Changelog

| Date | Change |
|------|--------|
| 2026-09-07 | Initial clone — Parts 1–8 (traditional ML pipeline) |
| 2026-09-07 | Added Part 9: GCN from scratch (PyTorch Geometric, 3-layer GCN) |
| 2026-09-07 | Added Part 10: GIN fine-tuning (MolCLR architecture, 5-layer GIN) |
| 2026-09-07 | Both notebooks verified on Colab — zero errors, actual results recorded |
| 2026-09-10 | Part 12 partial PC run (main 100 epochs + test eval only, CPU) — metrics below; Y-rand/CV moved to Colab |
| 2026-09-10 | Part 14: appended paper-COCONUT download cell (idx 14) — only addition, no existing cell touched |

---

## 2026-09-10 — Parts 12/13/14: PC run stopped, Colab runbook

PC (8-core CPU) was saturating (load ~7.4) on the AttentiveFP full protocol
(100 main + 10x100 Y-randomization + 5x100 CV epochs), so local execution was
stopped. Full training + screening moves to Colab GPU. No existing notebook
cell was modified for this — only one cell was appended to Part 14.

### Part 12 partial result (real, from unmodified Part_12_AttentiveFP_Local code)

- Ran as-is (script form, repo root as cwd, CPU torch 2.13, pyg 2.8.0) until stopped.
- Completed: 100-epoch main training + test-set evaluation (80/20 stratified, seed 57).
- NOT completed: 10x Y-randomization, 5-fold CV, final weight save (killed first).
- Test metrics (`Part_12/performance_attentivefp_test.csv`):

| Model | Accuracy | Precision | F1 | Sensitivity | ROC-AUC | MCC |
|-------|----------|-----------|----|-------------|---------|-----|
| AttentiveFP | 0.853 | 0.821 | 0.871 | 0.928 | 0.928 | 0.708 |

- Plots saved alongside: `training_curves_attentivefp.png`, `roc_curve_attentivefp.png`,
  `confusion_matrix_attentivefp.png`.
- Context: RF 0.960 / GCN 0.952 / GIN 0.947 ROC-AUC (diary 2026-09-07). AttentiveFP
  0.928 trails them on this split — re-check after full Colab run (100 epochs,
  Y-rand, 5-fold CV) before concluding.
- Local helper copies removed after the stop (`Part_2/`, `Part_6/`, `Part_8/` at repo
  root were duplicates of `Natural_MDM2_.../Part_*` used only for the PC run).

### Part 14 change (append-only)

- Added one code cell (idx 14, right after the OPTION B raw-COCONUT cell, before
  `## 4. Featurize`): downloads `coconut_csv-03-2025.csv` via gdown
  (`FILE_ID = 1-DFc6lMf6maNAWPwZFA8uY6951SokoNY`, same public Drive file as
  Parts 7/11), skips if already present. Default screen still uses the Part 8
  COCONUT table (154647 compounds); OPTION B raw screen stays opt-in.
- Notebook validates (nbformat, 30 cells). All model/screening/consensus/medchem/SDF
  cells untouched.

### Colab completions found locally (2026-09-10, user runs, 0 errors, no re-run needed)

- `Part_12_AttentiveFP_Colab.ipynb` (32 cells, exec 1–21, Device: cuda): full protocol
  done — 100 epochs (final Test AUC 0.924), 10x Y-randomization (AUC 0.18–0.70, no
  signal on shuffled labels — good), 5-fold CV AUC 0.948/0.858/0.928/0.964/0.887,
  weight saved (`attentivefp_mdm2.pth`, 350,661 params).
- `Part_13_HybridGCNGAT_Colab.ipynb` (32 cells, exec 1–20, Device: cuda): full protocol
  done — 100 epochs (final Test AUC 0.956), 10x Y-randomization (AUC 0.28–0.67),
  5-fold CV AUC 0.956/0.898/0.941/0.959/0.940, weight saved
  (`hybrid_gcn_gat_mdm2.pth`, 60,482 params).
- Test-set comparison (all recorded, same 80/20 split family, seed 57):

| Model | Accuracy | Precision | F1 | Sensitivity | ROC-AUC | MCC |
|-------|----------|-----------|----|-------------|---------|-----|
| RF (Morgan, Part 6) | 0.938 | 0.930 | 0.943 | 0.957 | 0.960 | 0.876 |
| GCN scratch (Part 9) | 0.915 | 0.892 | 0.923 | 0.957 | 0.952 | 0.830 |
| GIN (Part 10) | 0.899 | 0.850 | 0.913 | 0.986 | 0.947 | 0.807 |
| AttentiveFP (Part 12, Colab GPU) | 0.829 | 0.776 | 0.857 | 0.957 | 0.924 | 0.673 |
| Hybrid GCN+GAT (Part 13, Colab GPU) | 0.891 | 0.857 | 0.904 | 0.957 | 0.956 | 0.786 |
| AttentiveFP (Part 12, PC CPU partial) | 0.853 | 0.821 | 0.871 | 0.928 | 0.928 | 0.708 |

- Hybrid GCN+GAT is the best GNN (0.956, ~RF level). AttentiveFP trails (0.924 GPU /
  0.928 CPU partial) — same ranking on both machines, so it is architecture/data,
  not a training accident.
- Still to fetch from Colab into the repo: `Part_12/attentivefp_mdm2.pth` +
  `performance_attentivefp_{test,shuffled,cv}.csv`, `Part_13/hybrid_gcn_gat_mdm2.pth` +
  `performance_hybrid_{test,shuffled,cv}.csv`. Then run Part 14 (all 5 voters).

## 2026-09-10 — Part 15 created (small-data deep learning, GitHub-standard)

New folder `Part_15/` mirrors the Part 12/13 structure (README + requirements +
Local/Colab notebooks + `ro5_properties_filtered.csv` copy, MD5 `eae88fc7...`):

- `Part_15_SmallData_DL_Local.ipynb` (41 cells, 25 code) + `Part_15_SmallData_DL_Colab.ipynb`
  (42 cells: clone + `pip install torch-geometric rdkit transformers` prefix, rest identical core).
  Both validate (nbformat) and all code cells compile (Colab `!`/`%` cells excluded, same as Part 12).
- Three paper-backed, small-data-safe models (<50k trainable params each), same protocol as
  Parts 9/10/12/13 (80/20 split seed 57, 100 epochs Adam 1e-3, batch 64, 10x Y-rand, 5-fold CV):
  1. **D-MPNN-small** (~23k params) after Yang et al. 2019, `chemprop/chemprop` — directed
     edge messages with reverse-edge exclusion + skip, hidden=64, depth=2.
  2. **GINE-small** (~40k params) after Hu et al. 2020, `snap-stanford/pretrain-gnns` —
     hand-rolled GINE layers (atom project + bond-MLP messages + eps update), 3x hidden=64.
  3. **Frozen ChemBERTa + MLP** (~49k trainable) after Chithrananda et al. 2020 —
     `seyonec/ChemBERTa-zinc-base-v1` frozen mean-pooled embeddings (cached to
     `chemberta_embeddings.pt`), only the 768->64->2 head trains; gracefully skipped if the
     hub download/`transformers` is unavailable.
- Small-data tactics: small hidden dims, dropout 0.2–0.3, **train-only SMILES-enumeration
  augmentation (x4)** via `MolToSmiles(doRandom=True)`; Y-rand/CV stay non-augmented (canonical)
  so honesty checks remain comparable. Outputs: `performance_part15_{test,shuffled,cv}.csv`,
  `comparison_part15_vs_all.csv`, `training_curves_{dmpnn,gine,chemberta}.png`,
  `roc_curve_part15.png`, `confusion_matrix_part15.png`, weights `dmpnn_mdm2.pth` /
  `gine_mdm2.pth` / `chemberta_mlp_mdm2.pth` (saved by YOUR run, never committed untrained).

### Part 14 change (append-only, cells 0–30 proven byte-identical in-script)

- Appended 3 cells (now 34 cells, nbformat-valid, new code compiles): md `## 10. Part 15 voters`
  + loader/inference cell (verbatim Part 15 class defs — verified identical to the Part 15
  notebook — WANT15 download `Part_15/*.pth` with skip-if-missing, D-MPNN/GINE scored on the
  same `screen_loader` with edge_attr forward, ChemBERTa voter guarded on `transformers` +
  hub download, chunked so no 475 MB embedding matrix is stored) + 8-voter recompute cell
  (same consensus/soft-vote rules as idx22, saves `ensemble_screening_results_8model.csv` /
  `ensemble_consensus_hits_8model.csv`, original 5-voter files kept).
- To run on Colab: train Part 15 (T4, Run all) → download the 3 `.pth` into `Part_15/` →
  Run-all Part 14 (5 voters, then the 3 appended cells for 8 voters).

## 2026-09-25 — Full-flow port to all DL notebooks + rigor closers (Arjun parity)

All 7 training notebooks now run the fast-notebook flow (append-only edits, old outputs kept,
JSON-valid, new code compiles): TASK flag (regression|classification|both) with Delta-ML
(Ridge alpha=1.0 + StandardScaler, train-only fit, regression) and derived 0/1 @ pIC50>=7.0 +
ROC-AUC for the regression path; graph-first fingerprints (USE_MACCS/USE_MORGAN=False default);
WEIGHT_DECAY=1e-4; early stopping (PATIENCE=15 on val R2/AUC); AUGMENT_N=2; FILTER_DUP_SMILES=True;
PRED_CAP=10.0 clip; STRICT_MEDCHEM/AD wiring; ALREADY_FILTERED COCONUT warning; RF-consensus
merge; Part_11 gained docking tiers (`docking_tiers.csv` + T1/T2/T3 SMILES). Final cell counts:
`testing/fix_of_chemphore/colabtestregressiom_fast.ipynb` 60, Part_9_GNN_DeepChem 44,
Part_9_GCN_HFooladi 54, Part_10_Pretrained_GNN 50 (head-only wd, encoder lr 1e-5 untouched),
Part_10_GIN_HFooladi 48, Part_12_AttentiveFP 45, Part_13_HybridGCNGAT 44, Part_11_Screening 44.

Rigor closers (closes the last 2 Arjun gaps: GridSearchCV, RepeatedStratifiedKFold-125): each of
the 7 training notebooks gained 3 appended cells — mini hyperparameter grid
(lr{1e-3,3e-4} x wd{1e-4,1e-3} x dropout = 8 configs; 4 configs where the constructor lacks a
dropout arg, i.e. Part_10-Pretrained head-only grid and Part_9-HFooladi BasicGCN; early-stopped,
ranked table, BEST_CFG + CPU state_dict) + repeated CV (3 repeats x 5 folds = 15 evals, per-fold
Ridge+scaler refit on train-fold only, seeds RANDOM_SEED/SEED+100*r+fold, BEST_CFG-aware with
flag-default fallback, per-repeat and grand mean±std vs original single-5-fold). Verified:
expected cell counts 60/45/44/44/50/54/48 all match, py_compile clean, diffs additive-only.

Still to run (Colab T4, ~45–90 min/notebook): the new grid + repeated-CV cells in each notebook;
rescue `coconut_part15_hits*.csv` / `docking_tiers` outputs into the repo; docking remains the
critical path before paper claims.

## 2026-09-25 — Single-model ranking (RANK_MODE) + PC-linked vs Colab path split

Research call: ensemble (all 3) stays the primary ranker — it gives `std_pred_pic50` /
`high_variance` AD annotation, dilutes GINE's weak fold, and produced the 3x RF enrichment;
single model is a sensitivity check (appendix). No retraining needed for either (screening-only).

Fast notebook (60→61 cells, default `RANK_MODE='ensemble'` so all numbers unchanged):
`RANK_MODE = ensemble|dmpnn|gine|chemberta` + `rank_col` drives `coconut15-ensemble` sweep (now
prints ensemble sweep AND per-model sweep table at 6.0…8.5), `coconut15-save` threshold/rank,
`coconut15-plot` labels, `coconut15-export-rf` hit selection; std/`high_variance` always from all
3 models. New `coconut15-agreement` cell: Spearman ensemble-vs-each-model + top-155 overlap →
`coconut_part15_model_agreement.csv`. Recommended lone screener if ever needed: ChemBERTa
(most stable CV 0.720±0.022); avoid GINE alone (0.643±0.030).

All 7 training notebooks: `IN_COLAB = os.path.isdir('/content')` path split — PC resolves only
local linked candidates (`SOURCE=PC-linked …`, clear FileNotFoundError naming expected paths,
no downloads); Colab keeps clone/gdown fallbacks (`SOURCE=Colab …`). RF pkl chains split the
same way (`RF_SOURCE=…`). Verified: counts 61/45/44/44/50/54/48, py_compile clean, outputs kept.

## 2026-09-25 — Four locked decisions patched into fast notebook (structure-only, no run)

In-place upgrade to `testing/fix_of_chemphore/colabtestregressiom_fast.ipynb` (still 61 cells,
JSON-valid, all code cells `ast.parse` clean, outputs kept, notebook NOT executed):

1. GIN 64-dim/3-layer ~33k LOCKED (not 128/60k): `GIN_HIDDEN/LAYERS/DROPOUT=64/3/0.4`,
docstring + `assert params<50k` guard in `cb01e8a0`; fresh-model dropouts aligned to flags
(0.3/0.4/0.4). Why: <50k envelope for 516 train; Xu/Hu 300x5/465k is pretrain-scale, overfits.
2. Baseline Ridge-6desc kept (alpha=1.0, train-only): `USE_STRONG_BASELINE=False`; Delta-R2
= R2_delta − R2_ridge (~+0.29) is the DL-value metric. XGB-200d rejected (overfits 516, hides gain).
3. Split scaffold PRIMARY, random appendix: `SPLIT_MODE='scaffold'` → Murcko GroupShuffleSplit
80/20 seed 57 with qcut-random fallback; header cell documents random as optimistic appendix.
4. AD-tiered T1/T2/T3: `APPLICABILITY_DOMAIN_FILTER=True`, `AD_TANIMOTO_CUT=0.35`,
`TIER_MODE=True`; ensemble cell writes `docking_tier` (T1=GNN≥7.5&RF>0.6&in-AD dock first;
T2=GNN-only novel appendix; OOD never T1). Fixes 0/1600 RF-agree without losing novelty.
Y-rand honesty fix (`d9530b79`): shuffle RAW pIC50 then refit Ridge+scaler per shuffle
(graph + ChemBERTa branches); old code shuffled scaled delta after fit → fake R2 0.29–0.49.
Single-run contract: `RUN_ENV`/`SOURCE`, deterministic cudnn, seed 57 — same Run-All on
Colab-T4 and PC-GPU/CPU.

## 2026-09-25 — Colab run sheet (fresh run, T4 GPU)

Upload/run: open Colab → T4 GPU → upload `colabtestregressiom_fast.ipynb` → Runtime → Run all
(~45–90 min: trains D-MPNN/GINE/ChemBERTa-MLP 100ep early-stop + 10x Y-rand + 5-fold CV +
154647 COCONUT screen + RF-consensus on ≥7.5 hits). No flags to change (`SPLIT_MODE=scaffold`,
`AD_FILTER=True`, `RANK_MODE=ensemble` are the claim defaults).
Expect (scaffold lowers vs old random-appendix 0.746/0.646/0.714): D-MPNN R2~0.60–0.70,
ChemBERTa most stable, GINE lowest; Y-rand R2~0/neg (honest now); screen sweep
≥7.5 ~1–2k → tiers T1 (small, dock first) / T2 (novelty appendix); RF>0.6 on T1 only.
Rescue back to repo: `performance_part15_{test,shuffled,cv}.csv`,
`performance_comparison_part15.csv`, `coconut_part15_screening_full.csv`,
`coconut_part15_hits{,_top01pct}.csv`, `coconut_part15_hits_pIC50_7.5{,_rf_screened}.csv`,
`coconut_part15_model_agreement.csv`, `dmpnn/gine/chemberta_mdm2.pth`,
`training_curves/parity/residuals/screening.png`. Docking (4HG7) stays critical path after.

## 2026-09-26 — Pipeline fixes + regression-AUC applied to fast notebook (code-only, not executed)

Why: the 2026-09-25 Colab run (`Untitled1.ipynb`, T4) exposed two `KeyError` crashes
(`medchem_pass` skipped-cell ordering; `pred_chemberta` from the emptied ChemBERTa cell)
plus three silent degradations found in the 10-researcher review (dropped `prob_class_1`
kills T1/T2/T3 tiering; duplicate featurize wastes ~7 min; per-row RF loop). Also adds the
requested regression-AUC so GNN regressors can be compared on the RF's home turf.
All edits in `testing/fix_of_chemphore/colabtestregressiom_fast.ipynb` (61→60 cells):

A. medchem asserts (rank/save cells): explicit "run PAINS/Brenk/Ro5 cell first" instead of
   raw pandas `KeyError` — reason: crash came from skipped/out-of-order cell, message must say so.
B. Tolerant `gnn_cols`/`show_cols` (export-rf cell): filter to present columns + warn —
   reason: `HAVE_CB=False` runs must stay green without `pred_chemberta` (validated live on Colab).
C. ChemBERTa screening cell restored (was emptied 2026-09-25) — reason: without it the
   "ensemble" is silently 2-model while tables imply 3; one forward pass, embeddings cached.
D. Deleted duplicate featurize cell (byte-identical 49/50) — reason: ~7 min wasted per run.
E. Loader keeps `prediction/prob_class_0/prob_class_1` when present — reason: old code dropped
   them, so DECISION-4 tiering degraded to T1-ready/T2-OOD and true T1 (GNN+RF) was unreachable.
F. Vectorized RF `predict/predict_proba` (verbatim loop kept as comment) — reason: ~100x faster,
   numerically identical, Arjun parity preserved in comment.
G. `WEIGHT_DECAY` 1e-4→1e-3 — reason: rigor-grid best cfg (val R2 0.748 vs 0.737); numbers refresh on retrain.
H. Regression-AUC block (eval cell): per-model ROC-AUC + PR-AUC binarizing TRUE pIC50 at
   6.0/6.5/7.0, scored by PREDICTED pIC50 → `performance_part15_test_auc.csv` — reason: requested
   "AUC with regression"; single-class test slices guarded to NaN with a printed skip note.

Validation: `nbformat.validate` clean; `ast.parse` clean on all edited cells (only pre-existing
failures are cells 0–1 IPython magics `!git clone/%cd/!pip`, untouched); 11/11 structural checks
pass (WD, imports, AUC block+csv, RF-cols, 2 asserts, tolerant cols, vectorized loop, chemberta
producer, single featurize, single-class guard). Stored outputs KEPT (now stale for edited cells —
marked here, refresh via Colab Run-all). Out of scope (deferred): val-split early stopping,
scaffold-singleton fix, repeated-CV for all models, assay-type audit.
Rescue add: `performance_part15_test_auc.csv` joins the file list above.

## 2026-10-13 — MACCS late-fusion ON + softmax-saturation guard (fast notebook, code-only, not executed)

File: `testing/fix_of_chemphore/colabtestregressiom_fast.ipynb` (61→63 cells, JSON-valid, outputs kept stale).

What changed and why:

1. `USE_MACCS = False → True` (cell 6). MACCS 167 bits are projected `167→32` and concatenated with the pooled GNN output `g[64-d] → 96-d` head. Not fed into message-passing, so no remodel of the 78+6 graph convs and no topology domination. `USE_MORGAN` stays OFF (1024 bits = +33k params, too big for n=645). Effect: stable substructure keys help COCONUT screening generalization alongside learned topology.
2. Fusion upgrade `DirectedMPNN` + `GINESmall` (cells 9–10): `Linear→ReLU→Dropout(0.1)` proj, `LayerNorm + Dropout` after `cat`, instance `maccs_proj_dim` (removes global `MACCS_PROJ_DIM` fallback bug). Params: D-MPNN 22,977→30,593, GINE ~40,068 — still small-data-safe. Heads stay linear `out_dim=1`.
3. Fingerprint-carry fix (cells 39/41/59): Y-rand, 5-fold CV, and 3×5 repeated-CV helpers carried only `maccs`; now carry both `maccs` + `morgan` when present. Reason: CV folds silently dropped Morgan fusion before.
4. Softmax-bias solution (new cells 61–62, `## 17`): old classification collapsed (`avg_prob>0.5 → 62k hits, top10=1.0`, all probs near 0.5–1.0). Fix enforced = regression-only `TASK=regression` linear heads (no `softmax/sigmoid/CrossEntropy`), Delta-ML (`delta = pIC50 − Ridge baseline` + `StandardScaler`), `PRED_CAP=10.0` clip, ranking by continuous `avg_pred_pic50 + std_pred_pic50` top-0.1% instead of `p>0.5`. New check cell asserts no softmax string in any `forward` and reports spread/clip/variance (`std>0.15`, `>100` unique preds), failing loudly if saturation returns.

Error solved: classification saturation bias (near-identical softmax outputs) + missing MACCS signal at screen time.

Validation: `ast.parse` clean on all edited cells (only pre-existing `!/%` magics in cells 0–2 fail, untouched); dummy-batch forward pass CPU-verified for both models with/without MACCS (`[B]` continuous outputs, missing-MACCS zero fallback works); backup at `/tmp/opencode/backup_colabtestregressiom_fast.ipynb`. Not executed — refresh via Colab T4 Run-all, then rescue `performance_part15_*.csv`, `coconut_part15_*.csv`, `*.pth`, `*.png` as before.

Solution — what we do (applies the fix end-to-end):

1. Train graph-first with MACCS assist: RDKit graph (78+6) → D-MPNN/GINE → pool `g` → `cat(g, MACCS-32)` → `LayerNorm/Dropout` → linear pIC50/delta. ChemBERTa branch stays frozen-embedding + MLP. Delta-ML on (`USE_DELTA=True`, Ridge-6desc + scaler, train-only fit).
2. Prove no bias: train 100ep early-stop on val R2 → test R2/RMSE/MAE + parity/residual plots → 10× Y-rand (expect R2~0/neg) → 5-fold + 3×5 repeated CV → regression-AUC at pIC50 6.0/6.5/7.0 → run Cell 17 check (no-softmax assert + spread/clip/variance report).
3. Screen COCONUT: featurize once (graph + MACCS together) → predict scaled delta → inverse + baseline → clip 10.0 → `avg_pred_pic50 + std_pred_pic50` → sweep 6.0–8.5 → top-0.1% + medchem + AD-variance tiers (T1 dock-first, T2 novelty) → RF-consensus on ≥7.5 hits → rescue hits/CSVs/weights for docking (4HG7).

## 2026-10-19 — Both local papers digested + Notebook upgrade (bias-free, all-models × all-compounds) + Pretraining track (NPASS/PubChem/ZINC/ChEMBL — all free downloads)

### 1. Papers read in full (text extracted from the PDFs in Desktop/already read paper/)
- **Budha et al. 2025** (DOI 10.1002/adts.202501502 — the base paper this repo replicates). Verified constants: ChEMBL 2,571 → dedupe → RO5 (≤1 violation) → **645** (343 actives / 302 inactives; Active = IC50 ≤ 100 nM); 40 classifiers (MACCS vs Morgan r3/512, 5-fold CV × 25 reps, seed 57); **RF deployed: test AUROC 0.960 / F1 0.943 / MCC 0.876** (SVC 0.963); COCONUT funnel 695,119 → PAINS 640,409 → Brenk 214,920 → Lipinski **154,627** (repo reproduces 154,647 — 20-row canonicalization drift) → RF 3,713 predicted active → **116 at prob > 0.6** (repo reproduces exactly); docking 4GH7 (nutlin-3a redock −8.6 kcal/mol, RMSD 0.360 Å) → top hits −10.0/−9.6 → MD 300 ns → MM/GBSA −71.26/−70.75 → **2 leads: CNP0492204.2, CNP0385629.2 — novel quinazolindione peptidomimetic** (in-vitro only proposed). GitHub: `bishalbudha/Natural_MDM2_Inhibitor_Discovery_using_ML`; COCONUT snapshot `coconut_csv-03-2025.csv`.
- **Hou et al. 2025, VEGFR2 GNN** (DOI 10.1080/14756366.2025.2518192). 5,564 labels; tuned **GCN beat tuned RF/XGBoost by only ≈ +0.02 AUC/MCC** (0.8937/0.6554 vs 0.8720/0.6228). In vitro: 3/6 µM actives — and the **best-docked & best-MM-PBSA compound (β-carotene, −11.21 / −40.23) failed (−6.3%)** while lower-docked cynaroside (IC50 2,698 nM) worked. Lesson adopted: **selection needs ≥2 independent evidence axes + uncertainty**; no single score (ML prob, GNN score, docking) ranks hits alone.

### 2. Notebook upgrade — bias fixes + all-models coverage
- **DL notebook** `testing/fix_of_chemphore/colabtestregressiom_fast.ipynb` (91 → 92 cells):
  1. **Featurizer truncation fixed** — old line silently cut atom features to `[:78]` of 124 (dropping degree/charge/numHs/hybridization/aromatic/ring for every atom). Now `assert len(f) == 124` + models use `num_node_features=124` (8 cells). Old `.pth` checkpoints are invalid → full retrain is required and expected.
  2. **Test-129 no longer selects anything** — new 3-way split **464 / 52 / 129** (seed 57); early stopping (D-MPNN, GINE, ChemBERTa, rigor grid) runs on the 52 only; test-129 is evaluated **once** in §9.
  3. **AUGMENT_N 2 → 1** — SMILES enumeration of a graph duplicates isomorphic molecules; no more fake data growth.
  4. **Exact AD block** at screening: ECFP4-analog (r2/2048) max-Tanimoto to **all 645** plus the 343 actives, nearest-neighbor ids, `in_AD` at 0.40; `std_pred` stays a separate disagreement axis, never called AD.
  5. **EXPORT_THR 7.5 → 7.0** (label-consistent: pIC50 ≥ 7.0 ⇔ IC50 ≤ 100 nM Active) with 7.5 paper-parity counts printed; `schema_version=2` stamped in the full screening CSV.
  6. Acceptance cell: split sizes, model coverage counts, featurizer dim.
- **Root notebook** `MDM2_Fixed_Pipeline_Colab.ipynb` (18 → 19 cells): defaults to AD_CUT 0.40 / GNN_HIT 7.0; **cuts.json loader** (single source of truth, written at repo root); screening AD replaced from the 200-sampled-actives approximation to **exact** chunked Tanimoto to all 645 + 343 actives (model-consistent r3/512, NN ids, correlation vs the old approximation printed); `FULL_LIB_SCREEN` flag scores the whole library with the Part_18 GCN (writes `screening_full_gnn_consensus.csv`); acceptance cell.
- Validation: `nbformat.validate` OK both; `ast.parse` clean; discipline audit — training cells use `val_loader`; `test_loader` remains only in comments, the final §9 evaluation, and the honest fixed-epoch y-rand/CV protocols.
- Note: two layouts coexist honestly — Part_18 GCN/GIN use the intentional **78-dim** (`Part_18_GNN_Triage/common.py`), DL models now use the full **124-dim**; never swap checkpoints between them.

### 3. Pretraining track (Stage 1-A) — molecules from NPASS / PubChem / ZINC / ChEMBL / COCONUT, all free
Free sources (verified, download pages):
- **NPASS v3.0** (natural products; 204,023 NPs, 8,764 targets, 1,048,756 activity records; released 15/6/2025) — structure file ~65 MB (SMILES/InChI/InChIKey), activity ~105 MB, general info ~48 MB; https://bidd.group/NPASS/downloadnpass.html
- **PubChem** — https://ftp.ncbi.nlm.nih.gov/pubchem/ (Compound/CURRENT-Full/SDF bulk; PUG-REST for sampling; 124.6 M records; COCONUT merged into PubChem on 2026-10-01)
- **ZINC-22** — CartBlanche22 (≤100k molecules/request) https://cartblanche22.zinc.docking.org; tranches https://files.docking.org/zinc22 (`*.smi.gz`); 2D subsets https://files.docking.org/2D; rsync e.g. `rsync -L -a rsync://files.docking.org/ZINC22-3D/H06 .`; license: free to use, no bulk redistribution
- **ChEMBL family targets** — MDM2 CHEMBL1907611, MDM4 CHEMBL1255126, p53–MDM4 CHEMBL2221344 via EBI REST (pchembl_value, IC50/Ki only); local `chembl_37.fps` (1.5 GB) + `chembl_37.h5` in ~/Downloads
- **COCONUT** — existing snapshot (gdown FILE_ID `1-DFc6lMf6maNAWPwZFA8uY6951SokoNY`)

Recipe + rules: InChIKey dedupe → **L1/L6 leak filter: Tanimoto ≥ 0.85 OR same Murcko scaffold OR InChIKey match vs all 645 are REMOVED from the pool, with nearest-neighbor proof printed** → cap ≤ 120k rows → structure/descriptor pretraining on the shared model body (label-free w.r.t. bioactivity) → 1–2 epochs, 4 h wall-clock cap → `stage1a.pt` → warm-start via `PRETRAIN_W` (hooks already in the DL notebook cell 9) → gate: repeated-CV R2 ≥ 0.60, else documented negative. Notebook: `pretrain_stage1a_npass.ipynb` (created 2026-10-19; downloads free data, filters with leak proof, builds `pretrain_data/pretrain_pool.parquet`, pretrains the D-MPNN body). Reference-only asset: Budha clone's `pretrained_gin_mdm2.pth`.

## 2026-10-19 — Kaggle speed patch (CPU-100% / GPU-idle fix)
- Root nb: parallel RDKit fingerprint precompute (multiprocessing.Pool over all vCPUs) -> LIB_FP reused by the RF screen loop AND the exact-AD block (no double featurization); FULL_LIB_SCREEN graph build also pooled; batch 128->512.
- DL nb: COCONUT graph featurization loop pooled (imap, chunksize 256); GNN_WORKERS raised from 2 to min(4, cpu_count); SCREEN_BATCH=768 (was 256). DataLoader num_workers/persistent_workers and the dual-GPU screening path (D-MPNN cuda:0 + GINE cuda:1 via ThreadPoolExecutor) already existed in cells 5/72 — left as is.
- Fallback: every pooled block falls back to the original serial loop on any exception (so nothing breaks if multiprocessing is unavailable).
- Validation: nbformat + ast.parse clean on both notebooks; Pool+RDFKit smoke test passed locally; flags USE_MP/N_WORKERS printed at startup.

## 2026-10-24 — Part_20_Fixed rebuilt: dual-GPU regression + resume + train↔screen similarity
- **File rebuilt**: `Part_20_Fixed/MDM2_Fixed_Pipeline_Colab.ipynb` — now 19 cells (1 markdown + 18 code), rewritten from the old RF-classification 15-cell version. Backup: `/tmp/opencode/backups/Part_20_Fixed__MDM2_Fixed_Pipeline_Colab.PRE_DUAL_REG.bak.ipynb`.
- **Regression only (no RF comparison)**: delta-ML = Ridge(6 RDKit descriptors: MolWt, MolLogP, TPSA, HBD, HBA, RingCount) fitted on the 464 training compounds + GCN-reg / GIN-reg learning the *scaled residual*; `StandardScaler` fitted on the training deltas only. Final prediction = baseline + residual, clipped to 0–12 pIC50.
- **Dual-GPU fully used**: `ThreadPoolExecutor` trains GCN on `cuda:0` and GIN on `cuda:1` simultaneously (same parallel pattern for screening the library); automatic single-GPU/CPU fallback. Models: 78-dim `common.py` graphs, 3 message-passing layers, mean+max pooling, 17,793 (GCN) / 26,113 (GIN) params, Adam + `ReduceLROnPlateau`, SmoothL1 on scaled delta, grad-clip 5.0, batch 64, max 120 epochs, patience 15.
- **Honest accuracy discipline**: scaffold split **464 / 52 / 129** (204 unique Murcko scaffolds, 0 INVALID, canonical-overlap 0, scaffold overlap 0); early stopping only on the 52; **test-129 evaluated once**; no test-based model selection. Baseline sanity on real data: test R2 = 0.410, RMSE = 1.412.
- **Split bug caught during smoke test**: `GroupShuffleSplit(test_size=0.20)` splits *groups*, not molecules — it silently produced 391/44/210 (test = 33%). Replaced with a deterministic cumulative scaffold split (scaffold groups sorted by size, greedily assigned until 20% of molecules = 129) → verified 464/52/129.
- **Idempotent re-runs**: every stage cached under `cache/` and skipped with `[resume]` — GitHub downloads, Morgan fingerprint matrices, train/val/test graphs, library graphs, trained models + metrics + val/test predictions, library descriptors, exact-AD arrays. Library-keyed caches use `LIB_HASH` (row count + first/last SMILES md5).
- **GitHub-first data**: `Part_2/ro5_properties_filtered.csv` (train), `Part_8/screening_results.csv` (filtered 154,647-row library), `Part_18_GNN_Triage/common.py` (78-dim featurizer) all verified HTTP 200 on `raw.githubusercontent.com/Techbjd/new_method/main`; HEAD availability table printed at runtime. `cuts.json` is 404 on GitHub → notebook falls back to defaults `ad_cut 0.40`, `gnn_hit_cut 7.0`. Raw COCONUT gdown (`1-DFc6lMf6maNAWPwZFA8uY6951SokoNY`) kept only as a last-resort fallback.
- **Pretrained GCN/GIN `.pth` deliberately not used**: their temperature was calibrated on the held-out 129 test set, so warm-starting from them would leak test labels.
- **Train↔screen similarity properly handled**: ONE Morgan r3/512 fingerprint for training and screening; exact chunked Tanimoto to **all 645** training compounds and **all 343 actives**, with nearest-neighbour ChEMBL IDs, `maxT_645`, `maxT_343act`, `in_AD`; domain-shift table (train leave-one-out vs test↔train vs screen↔train: n, mean, median, p10, p90, fraction ≥ AD_CUT) written to `cache/domain_shift.json`. Tiers: T1 = avg pIC50 ≥ 7.0 AND in_AD AND std ≤ median; T2 = ≥ 7.0 or (≥ 6.5 AND in_AD); else T3.
- **Kaggle copy-paste artifact**: `Part_20_Fixed/MDM2_Fixed_Pipeline_KAGGLE_ALL_CODE.py` — the complete notebook code head-to-toe (1,016 lines, 44 KB), markdown sections as comments; paste into one Kaggle cell, or import the `.ipynb` directly.
- **Validation**: `nbformat.validate` VALID; `ast.parse` clean on all 18 code cells and on the flattened script; smoke tests passed — GCN/GIN forward+backward, 78-dim graph featurizer, exact Tanimoto self-similarity = 1, scaffold split cell executed on real 645 → 464/52/129, delta-ML baseline cell executed → test R2 0.410.
- **Outputs**: `screening_full_regression.csv` (schema_version 2, preds + std + conservative + AD + NN ids + tier), `hits_T1/T2/T3.csv` (top 100 by avg pIC50), `screening_regression_QC.png` (6 panels incl. test parity and validation-R2 curves), `run_manifest.json`.

## 2026-10-24 (run) — Full end-to-end execution + train↔screen similarity bug fix
- **All packages handled**: setup cell now verifies/installs every required package (numpy, pandas, scikit-learn>=1.3, rdkit, matplotlib, joblib, gdown, torch, torch_geometric); local run printed "all required packages already installed"; Kaggle installs only what is missing and never replaces the CUDA torch build.
- **Memory-safe screening refactor**: cell 13 caches only the Morgan fingerprint matrix; cell 14 streams the 154,587-molecule library in 2,000-SMILES chunks (build graphs → predict GCN+GIN → discard, `gc.collect()`), caching `screen_preds_<LIB_HASH>_<MODEL_SIG>.npz`. Avoids the 3–5 GB all-graphs RAM spike (this machine has 7.1 GB total).
- **Full run on CPU**: 5.96 min end-to-end (GCN/GIN training 59 s; screening 154,587 molecules in 176 s). Notebook executed with nbclient — 18/18 code cells, zero errors, outputs saved into the `.ipynb`. Cached re-run: 0.51 min.
- **Similarity bug caught and fixed**: run 1 compared the 129 test fingerprints against all 645 training fingerprints, so every test compound matched itself → test↔train maxT = 1.000. Corrected AD reference = the **464 actual training compounds**: exact chunked Tanimoto to 464 train (`maxT_464`, `nn_464_id`, `in_AD`), with all 645 (`maxT_645`, `in_AD_all645`) and 343 actives (`maxT_343act`, `nn_343act_id`) as secondary columns. Domain-shift table recomputed.
- **Corrected regression results** (pIC50; test-129 touched once; early stopping only on val-52):
  - Ridge baseline: val R2 0.485 / RMSE 1.174; test R2 0.410 / RMSE 1.412 / MAE 1.273
  - GCN-reg: val 0.861 / 0.609; test 0.649 / 1.089 / 0.794
  - GIN-reg: val 0.866 / 0.599; test 0.671 / 1.055 / 0.773
  - Ensemble: val 0.871 / 0.587; test 0.666 / 1.063 / 0.768
- **Domain shift** (Morgan r3/512): train LOO median 0.812 (95.0% ≥0.40); test↔train median 0.633 (97.7% ≥0.40); screen↔train median **0.217** (0.57% ≥0.40) → COCONUT is ~99.4% outside the applicability domain; ranking alone is extrapolation.
- **Screening/tiers**: 154,587 unique COCONUT rows; 886 in AD (maxT_464 ≥ 0.40), 917 via all-645; T1=103, T2=24,901, T3=129,583. `hits_T1.csv` stores the top 100 of the 103 T1 hits.
- **Top T1** (avg pIC50 / std / maxT_464 / nearest train): CNP0423424.1 8.42/0.067/0.567/CHEMBL803; CNP0071615.1 8.30/0.059/0.451/CHEMBL4105199; CNP0234138.1-.2 8.29/0.240/0.405/CHEMBL4105199; CNP0141832.1 8.27/0.051/0.412/CHEMBL3819001; CNP0327098.1 8.26/0.058/0.414/CHEMBL4074577; CNP0603870.1 8.17/0.224/0.420/CHEMBL3818299; CNP0155594.1 8.06/0.013/0.427/CHEMBL3819001. Raw top predictions (10.2–10.5) are all OOD (maxT 0.16–0.24) → T2, not T1; never rank without the AD columns.
- **Duplicates**: CNP0234138.1/.2, CNP0505565.1-.4, CNP0049127.1-.3 give identical predictions (stereoisomers/duplicates) — collapse these before any wet-lab selection.
- **Kaggle**: notebook is importable and self-contained; Settings → Accelerator GPU T4 ×2, Internet ON, Persistence Files; Run All. Auto-selects cuda:0/cuda:1 for parallel GCN/GIN training and screening; CPU fallback verified locally.
- **Validation**: `nbformat.validate` valid; all code cells AST-clean; zero error outputs; `run_manifest.json` and `cache/domain_shift.json` match the printed tables.

## 2026-10-24 (audit) — Full DL input/output verification cell
- Notebook is now **20 cells (1 markdown + 19 code)**; flat Kaggle file regenerated to **1,267 lines / 57,579 bytes**, AST-clean, with every notebook code line present.
- New final cell `# 18 — FULL INPUT/OUTPUT AUDIT` prints:
  - split sizes, pIC50 ranges, graph node/edge stats, first 3 atom-feature rows and first 5 edges of a real training graph;
  - delta-scaler mean/scale, model config, devices, split policy;
  - full GCN/GIN model summaries, strict state-dict load and non-finite-weight count;
  - **every epoch** of validation R2 (training loop now prints every epoch, not every 10);
  - the **complete 645-row prediction table** (train + val + test; SMILES, true pIC50, Ridge/GCN/GIN/ensemble, residuals), with in-sample vs held-out clearly marked;
  - worst 8 absolute errors per split, signed bias by true-pIC50 bin;
  - a screening **SMILES → graph → prediction → saved CSV** spot-check.
- Saved `dl_full_predictions.csv` (645 rows), `dl_training_history.csv`, `dl_audit.json`.
- Audit result: **13/13 PASS**; test ensemble R2 0.666 / RMSE 1.063 / MAE 0.768; test bias −0.525, calibration slope 0.696. Low-pIC50 test bin (0,6] bias −1.059 (systematic over-prediction of weak compounds) — the main honest error pattern; worst test error 3.43 pIC50 (CHEMBL3703675). Screening coverage 154,587/154,587, in_AD 886, spot-check max|diff| 1e-6.
- Notebook executed end-to-end with nbclient: 19/19 code cells, zero error outputs; audit output ~127 KB.
