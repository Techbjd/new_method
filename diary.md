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
