"""Atlas cell definitions, part A (title/TOC, setup, data, features, split)."""

_CELLS_A = []


def _add(kind, cid, src):
    _CELLS_A.append((kind, cid, src))


_add("markdown", "S01", """# MDM2 Atlas — All Models, 5-Fold Scaffold CV, Softmax Screening (input → output)

Single-notebook pipeline in the style of Arjun sir's teaching repo: **data information →
features → deep-learning AND traditional ML models → 5-fold CV → ensemble → screening → outputs**,
every stage in its own code cell, whole run guarded so it finishes without error
(heavy optional libraries print a clean `[skipped]` line instead of crashing).

**Contents**
1. Runtime & installs
2. Config (seed 57, cuts.json, fallback paths)
3. Data: 645-compound training table (ChEMBL CHEMBL303), 154,647-row COCONUT library
4. Data information (quantity, lineage, thresholds)
5. Features: 10 descriptors + 6 fingerprint families + 78-dim graphs
6. Scaffold split ≈80/9/10 (non-isomeric groups) + 5-fold GroupKFold
7. Traditional ML models (one per notebook cell-style section)
8. Deep learning models (GCN, GAT, GATv2, GIN, AttentiveFP, D-MPNN, SMILES-CNN)
9. Optional heavy extras (DeepChem GCN, Chemprop D-MPNN, ChemBERTa, MoLFormer)
10. 5-fold CV scan
11. Ensemble (softmax-mean of core GNNs)
12. Screening the library → all-model votes, top-50, AD-gated shortlist (not-hits discipline)
13. Robust evaluation (RMSE/MAE/R², AUC/AAP/balanced-acc, bootstrap CIs, precision@10, EF1%, BEDROC)
14. Manifest & verification

## Models used by researchers in this project (paper + code, simple list)

| # | Model | Research paper | Code (GitHub/HF) |
|---|---|---|---|
| 1 | Random Forest | Breiman, *Random Forests*, Machine Learning 2001, **DOI 10.1023/A:1010933404324**; project ref: Budha et al., Adv. Theory Simul. 2025 | **scikit-learn** github.com/scikit-learn/scikit-learn (Arjun sir Part 6/8 code) |
| 2 | XGBoost | Chen & Guestrin, KDD 2016, **DOI 10.1145/2939672.2939785** | **github.com/dmlc/xgboost** |
| 3 | LightGBM | Ke et al., NeurIPS 2017 | **github.com/microsoft/LightGBM** |
| 4 | CatBoost | Dorogush et al., arXiv:1810.11363 (2018) | **github.com/catboost/catboost** |
| 5 | Extra Trees | Geurts et al., ML 2006, **DOI 10.1007/s10994-006-6226-1** | scikit-learn |
| 6 | Hist Gradient Boosting | sklearn HistGB (LightGBM-inspired) | scikit-learn |
| 7 | AdaBoost | Freund & Schapire, JCSS 1997, **DOI 10.1006/jcss.1997.1504** | scikit-learn |
| 8 | SVM (RBF) | Cortes & Vapnik, ML 1995, **DOI 10.1007/BF00994018** | scikit-learn |
| 9 | Ridge (6 descriptors) | Hoerl & Kennard, Technometrics 1970, **DOI 10.1080/00401706.1970.10488634** | scikit-learn |
| 10 | k-NN (fingerprints) | Cover & Hart, IEEE TIT 1967 | scikit-learn |
| 11 | GCN | Kipf & Welling, ICLR 2017, arXiv:1609.02907 | **github.com/HFooladi/GNNs-For-Chemists** (replication), **github.com/deepchem/deepchem** (method source) |
| 12 | GIN | Xu et al., ICLR 2019, arXiv:1810.00826 | **github.com/HFooladi/GNNs-For-Chemists** (notebook 06); MolCLR **github.com/yuyangw/MolCLR** |
| 13 | GAT / GATv2 | Veličković et al., ICLR 2018, arXiv:1710.10903; Brody et al., ICLR 2022 | PyG github.com/pyg-team/pytorch_geometric |
| 14 | AttentiveFP | Xiong et al., J. Med. Chem. 2020, 63(16):8749, **DOI 10.1021/acs.jmedchem.9b00959** | **github.com/OpenDrugAI/AttentiveFP**, PyG port |
| 15 | D-MPNN (Chemprop-style, built-in compact) | Yang et al., JCIM 2019, 59:3370, **DOI 10.1021/acs.jcim.9b00237**; Heid et al. 2023 | method source **github.com/chemprop/chemprop**; this notebook uses a compact in-repo implementation |
| 16 | DeepChem GCN (Arjun sir Part 9) | Ramsundar et al., *Molecular Machine Learning with DeepChem* | **github.com/deepchem/deepchem** |
| 17 | SMILES char-CNN | Gómez-Bombarelli et al., ACS Cent. Sci. 2018, **DOI 10.1021/acscentsci.7b00572** | in-repo (part21) |
| 18 | ChemBERTa (optional) | Chithrananda et al., arXiv:2010.09891 | **huggingface.co/seyonec/ChemBERTa-zinc-base-v1**, **github.com/seyonechithrananda/bert-loves-chemistry** |
| 19 | MoLFormer-XL (optional) | Ross et al., Nature MI 2022 | **huggingface.co/ibm-research/MoLFormer-XL-both-10pct** |
| 20 | Ensemble | — (uncalibrated simple mean of core GNNs; Part 20 calibration was separate) | in-repo (Part 14 / Part 22 S17) |
| 21 | Pharmacophore search (SOURCES) | clubbed with ML screening | **github.com/MarciaC18/Identification-of-MDM2-P53-Inhibitors-Using-Machine-Learning-Guided-Screening**, **github.com/vktrannguyen/MLSF-protocol** |

> **Scope of the default run.** The six graph models (GCN, GAT, GATv2, GIN,
> AttentiveFP, D-MPNN) and the SMILES char-CNN are trained by default. DeepChem
> GCN, the official Chemprop package, ChemBERTa and MoLFormer are optional and
> print `[skipped]` unless their flags are enabled; they are not part of the
> default evidence. The built-in D-MPNN follows the Chemprop directed-message
> idea but is a compact reimplementation, not the Chemprop package.

> Everything here follows project guardrails: seed 57, Murcko-scaffold split,
> validation-only model selection, test scored once, strict AD gate (0.52),
> dedup by parent + Murcko scaffold, and *shortlist = low-confidence hypotheses,
**not validated hits**.
""")


_add("code", "S02", """
import os, sys, time, json, subprocess
import numpy as np

RUN_SEED = 57
np.random.seed(RUN_SEED)
T_START = time.time()
print("[S02] runtime:", sys.version.split()[0])

def _pip(pkg, pip_name=None):
    try:
        return __import__(pkg)
    except Exception:
        print(f"[S02] installing missing package: {pip_name or pkg}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        (pip_name or pkg).split("==")[0]], check=False)
    return __import__(pkg)

import torch
torch_geometric = _pip("torch_geometric")
sklearn = _pip("sklearn", "scikit-learn")
rdkit = _pip("rdkit")
print("[S02] torch", torch.__version__, "| pyg", torch_geometric.__version__,
      "| sklearn", sklearn.__version__, "| rdkit", rdkit.__version__,
      "| cuda", torch.cuda.is_available())
""")


_add("code", "S03", """
# --- config: fallback paths, flags, thresholds (cuts.json with defaults) ---
import os, json

SMOKE = bool(int(os.environ.get("SMOKE", "1")))
SCAN_FOLDS = int(os.environ.get("SCAN_FOLDS", "5" if not SMOKE else "2"))
DL_EPOCHS = int(os.environ.get("DL_EPOCHS", "60" if not SMOKE else "3"))
CUTS = {"ad_cut": 0.4, "rf_t1_cut": 0.6, "rf_t2_cut": 0.4, "gnn_hit_cut": 7.0, "ad_cut_strict": 0.52}
_INPUT = os.environ.get("MDM2_ATLAS_INPUT_DIR", "")
for cand in [_INPUT, os.path.join(os.getcwd()), ".."]:
    cuts_path = os.path.join(cand.strip(), "00_data", "config", "cuts.json") if cand else "00_data/config/cuts.json"
    if os.path.exists(cuts_path):
        try:
            CUTS.update(json.load(open(cuts_path)))
        except Exception as e:
            print("[S03] cuts.json parse failed, defaults kept:", e)
        break

USE_CHEMBERTA = bool(int(os.environ.get("USE_CHEMBERTA", "0")))
USE_DEEPCHEM = bool(int(os.environ.get("USE_DEEPCHEM", "0")))
USE_CHEMPROP = bool(int(os.environ.get("USE_CHEMPROP", "0")))
OUT_DIR = os.environ.get("MDM2_ATLAS_OUTPUT_DIR", os.getcwd())
os.makedirs(OUT_DIR, exist_ok=True)
SAMPLE_N = 96 if SMOKE else None      # smoke slices the data; full run uses 645
LIB_SLICE = 2000 if SMOKE else None   # screening subsample in smoke
print("[S03] SMOKE=", SMOKE, "| SCAN_FOLDS=", SCAN_FOLDS, "| DL_EPOCHS=", DL_EPOCHS, "| CUTS=", CUTS)
""")


_add("code", "S04", """
# --- load training table (fallbacks: env dir -> cwd -> ../ -> GitHub raw -> synthetic smoke) ---
import pandas as pd

TRAIN_CANDIDATES = [
    os.path.join(os.environ.get("MDM2_ATLAS_INPUT_DIR", "") or ".", "00_data/training/ro5_properties_filtered.csv"),
    "00_data/training/ro5_properties_filtered.csv",
    "../00_data/training/ro5_properties_filtered.csv",
    "https://raw.githubusercontent.com/Techbjd/new_method/main/00_data/training/ro5_properties_filtered.csv",
]
SYNTHETIC_FRAGMENTS = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "CCN", "c1ccncc1", "CSc1ccccc1",
                       "CC(C)O", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "O=C1CCCCC1", "CC1=CC=C(C=C1)C=C",
                       "NC1=NC2=C(N1)C(=O)N(C)C(=O)NC2=O", "c1ccc2c(c1)[nH]c1ccccc12", "CCCC",
                       "NS(=O)(=O)c1ccc(cc1)N", "O=C(N)c1ccccc1N", "COc1ccc(cc1)C=C"]

def _load_training():
    for path in TRAIN_CANDIDATES:
        try:
            df = pd.read_csv(path)
            return df, path, "real"
        except Exception:
            continue
    rng = np.random.default_rng(RUN_SEED)
    smiles_syn, labels_syn, pic50_syn = [], [], []
    for i in range(128):
        frags = list(rng.choice(SYNTHETIC_FRAGMENTS, size=int(rng.integers(1, 3)), replace=False))
        smiles_syn.extend([".".join(frags)])
        labels_syn.append(int(rng.random() < 0.53))
        pic50_syn.append(float(4 + 6 * (labels_syn[-1] * 0.4 + rng.random() * 0.6)))
    df = pd.DataFrame({"molecule_chembl_id": [f"SYN{i:03d}" for i in range(len(smiles_syn))],
                       "bioactivity_class": ["Active" if y else "Inactive" for y in labels_syn],
                       "pIC50": pic50_syn, "canonical_smiles": smiles_syn})
    return df, None, "synthetic-smoke"

TRAIN_DF, TRAIN_SRC, TRAIN_MODE = _load_training()
print("[S04] training table:", TRAIN_DF.shape, "| source:", TRAIN_SRC or "synthetic", "| mode:", TRAIN_MODE)
assert {"canonical_smiles", "bioactivity_class", "pIC50"} <= set(TRAIN_DF.columns)
""")


_add("code", "S05", """
# --- load the 154k screening library (structures-only: prediction cols dropped) ---
LIB_CANDIDATES = [
    os.path.join(os.environ.get("MDM2_ATLAS_INPUT_DIR", "") or ".", "00_data/screening/screening_results.csv"),
    "00_data/screening/screening_results.csv",
    "../00_data/screening/screening_results.csv",
    "https://raw.githubusercontent.com/Techbjd/new_method/main/00_data/screening/screening_results.csv",
]

def _load_library():
    for path in LIB_CANDIDATES:
        try:
            df = pd.read_csv(path)
            keep = [c for c in ("identifier", "canonical_smiles") if c in df.columns]
            if keep:
                return df[keep].copy(), path, "real"
        except Exception:
            continue
    rng = np.random.default_rng(RUN_SEED + 1)
    rows = []
    for i in range(500):
        frags = list(rng.choice(SYNTHETIC_FRAGMENTS, size=int(rng.integers(1, 3)), replace=False))
        rows.append((f"LIB{i:06d}", ".".join(frags)))
    return pd.DataFrame(rows, columns=["identifier", "canonical_smiles"]), None, "synthetic-smoke"

LIB_DF, LIB_SRC, LIB_MODE = _load_library()
print("[S05] library:", LIB_DF.shape, "| source:", LIB_SRC or "synthetic", "| mode:", LIB_MODE)
if LIB_SLICE:
    LIB_DF = LIB_DF.head(LIB_SLICE).reset_index(drop=True)
    print("[S05] smoke slice ->", LIB_DF.shape)
""")

CELLS_A = _CELLS_A
