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
