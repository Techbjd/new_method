#!/usr/bin/env python3
# =============================================================================
# build_notebooks.py — generates the three DL notebooks in 06_deep_learning_models/
# Provenance: [USER]+[GENERATED] builder; cell code [ADAPTED] from
#   02_gnn_models/models/*, 01_traditional_ml/featurize.py,
#   03_screening_pipeline/part22_atlas (split/models/fit),
#   part20_fixed (dual-GPU util), Part_9 (Y-randomization protocol).
# Run:  python3 build_notebooks.py   → writes 3 validated .ipynb files
# =============================================================================
import ast
import json
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Shared code-cell templates (%%TOKEN%% placeholders replaced per notebook)
# ---------------------------------------------------------------------------

SETUP_SRC = r'''
# ## 2. Environment Setup & Installs (idempotent — Colab / Kaggle / local)
import os, sys, platform, shutil, subprocess, importlib

# Thread/env limits set BEFORE torch is imported anywhere (speeds up CPU math,
# avoids oversubscription when the GPU DataLoader workers spawn later).
# Cap at 8: PyTorch gains little from more threads on these small graphs and
# oversubscription on 32/64-core servers is slower, not faster.
_NCPU = max(1, min(os.cpu_count() or 4, 8))
os.environ.setdefault("OMP_NUM_THREADS", str(_NCPU))
os.environ.setdefault("MKL_NUM_THREADS", str(_NCPU))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(_NCPU))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def _pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "--disable-pip-version-check", *args], check=False)

def _have(mod):
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False

# --- torch: only install when missing; pick CPU wheels for CPU-only machines ---
if not _have("torch"):
    _has_nvidia = bool(shutil.which("nvidia-smi")) and subprocess.run(
        ["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if platform.system() == "Darwin" or _has_nvidia:
        _pip("torch")                                   # PyPI wheel (CUDA on Linux/Win GPU)
    else:
        _pip("torch", "--index-url", "https://download.pytorch.org/whl/cpu")
if not _have("torch_geometric"):
    _pip("torch_geometric")
for _mod, _pkg in [("rdkit", "rdkit"), ("sklearn", "scikit-learn"),
                   ("matplotlib", "matplotlib"), ("pandas", "pandas")]:
    if not _have(_mod):
        _pip(_pkg)
try:
    import torch as _t
    _accel = "CUDA" if _t.cuda.is_available() else "CPU"
    print(f"[setup] environment ready — python {sys.version.split()[0]} | torch {_t.__version__} "
          f"| {_accel} | {_NCPU} cpu threads")
except Exception as _e:
    print("[setup] environment ready, but torch import failed:", type(_e).__name__, _e)
'''

REPRO_SRC = r'''
# ## 3. Reproducibility — RANDOM_SEED = 57 everywhere (project convention)
import os, json, random, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

RANDOM_SEED = 57

def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# SMOKE  : env DL_SMOKE=1  -> tiny fast run (used for local validation)
# FAST   : env DL_FAST=1   -> reduced real run (CPU-friendly; all models, 2 CV folds)
# TASK   : env DL_TASK=cls (default) or reg (pIC50 regression)
SMOKE = os.environ.get("DL_SMOKE", "0") == "1"
FAST = os.environ.get("DL_FAST", "0") == "1"
TASK = os.environ.get("DL_TASK", "cls")
PRIMARY_METRIC = "AUC" if TASK == "cls" else "RMSE"
PROFILE = "smoke" if SMOKE else ("fast" if FAST else "full")
OUT_TAG = PROFILE

from pathlib import Path
OUT_DIR = Path("outputs") / "%%NB_ID%%" / OUT_TAG
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"[seed] RANDOM_SEED={RANDOM_SEED} | profile={PROFILE} | TASK={TASK} "
      f"| primary metric={PRIMARY_METRIC} | outputs -> {OUT_DIR}")
'''

UTIL_SRC = r'''
# ## 4. Full GPU / RAM / CPU Utilization
import gc, platform
from concurrent.futures import ThreadPoolExecutor

N_GPUS = torch.cuda.device_count()
DEVICE = torch.device("cuda:0" if N_GPUS > 0 else "cpu")
DEVICES = [torch.device(f"cuda:{i}") for i in range(N_GPUS)] or [torch.device("cpu")]

# --- speed switches ---------------------------------------------------------
torch.backends.cudnn.benchmark = True              # best conv algorithms for fixed shapes
torch.backends.cuda.matmul.allow_tf32 = True       # TF32 on Ampere+ GPUs
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
if DEVICE.type == "cpu":
    torch.set_num_threads(_NCPU)                   # use every CPU core locally

USE_AMP = DEVICE.type == "cuda"                    # mixed precision on GPU only

def autocast_ctx(device=None):
    device = device or DEVICE
    enabled = device.type == "cuda" and USE_AMP
    try:
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast(device_type=device.type, enabled=enabled)
    except TypeError:
        pass
    return torch.cuda.amp.autocast(enabled=enabled)          # torch < 2.3 fallback

def scaler_ctx(device=None):
    device = device or DEVICE
    enabled = device.type == "cuda" and USE_AMP
    try:
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            return torch.amp.GradScaler(device.type, enabled=enabled)
    except TypeError:
        pass
    return torch.cuda.amp.GradScaler(enabled=enabled)        # torch < 2.3 fallback

# --- monitoring / cleanup ---------------------------------------------------
def _gpu_util(i):
    try:
        return torch.cuda.utilization(i)
    except Exception:
        return -1

def gpu_report(tag="[gpu]"):
    if N_GPUS == 0:
        print(f"{tag} no CUDA — CPU mode ({_NCPU} threads, python {platform.python_version()})")
    for i in range(N_GPUS):
        free, total = torch.cuda.mem_get_info(i)
        print(f"{tag} cuda:{i} {torch.cuda.get_device_name(i)} | "
              f"{total / 2**30:.1f} GB total, {free / 2**30:.1f} GB free | util {_gpu_util(i)}%")

def free_mem():
    gc.collect()
    if N_GPUS:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

# --- DataLoader factory (pin_memory, workers, prefetch) ---------------------
from torch.utils.data import DataLoader, TensorDataset, Subset

def num_workers():
    # DataLoader workers use fork() on Linux. run_parallel() runs training jobs in
    # ThreadPoolExecutor threads (dual-GPU mode); Python 3.13 + filelock >= 3.3x
    # reject os.fork() while a lock is held ("changing descriptor ownership"), and
    # forking from a Jupyter thread is unsafe in general. PyG datasets also do not
    # pickle reliably under Jupyter spawn. The 645 in-memory graphs load in-process
    # at negligible cost, so keep workers OFF by default. Opt in only for a run you
    # know is single-threaded: DL_DATALOADER_WORKERS=2.
    return max(0, int(os.environ.get("DL_DATALOADER_WORKERS", "0")))

def make_loader(ds, batch_size, shuffle, kind="graph"):
    if kind == "graph":
        from torch_geometric.loader import DataLoader as _DL
    else:
        _DL = DataLoader
    kw = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers(),
              pin_memory=(DEVICE.type == "cuda"))
    if kw["num_workers"] > 0:
        kw.update(persistent_workers=True, prefetch_factor=4)
    return _DL(ds, **kw)

def auto_batch(base_batch):
    """Scale the batch up when lots of VRAM is idle (full-utilization mode)."""
    if SMOKE or DEVICE.type != "cuda":
        return base_batch
    free, _ = torch.cuda.mem_get_info(0)
    factor = 4 if free / 2**30 > 12 else 2 if free / 2**30 > 6 else 1
    return int(base_batch * factor)

# --- dual-GPU parallelism (ThreadPoolExecutor pattern from Part_20) ---------
def run_parallel(fn, jobs):
    """jobs: list of dicts, each with a unique 'key'; fn(job, device_id) -> dict.
    With >1 GPU, jobs are split across cuda:0 / cuda:1 and run in threads."""
    if N_GPUS <= 1 or len(jobs) <= 1:
        return [fn(j, 0) for j in jobs]

    def _run_part(part, g):
        return [fn(j, g) for j in part]

    parts = [jobs[i::N_GPUS] for i in range(N_GPUS)]
    with ThreadPoolExecutor(max_workers=N_GPUS) as ex:
        results = [f.result() for f in [ex.submit(_run_part, p, g) for g, p in enumerate(parts)]]
    out = [r for part in results for r in part]
    order = {j["key"]: i for i, j in enumerate(jobs)}
    return sorted(out, key=lambda r: order[r["key"]])

gpu_report("[S04]")
'''

LOAD_SRC = r'''
# ## 5. Load the MDM2 Dataset (645 compounds) + EDA
# Kaggle/Colab-safe: NO local repo structure required. Local copies are tried
# first (when running inside this repo); otherwise the canonical CSV is
# downloaded from Arjun sir's public GitHub repo. On Kaggle enable Internet.
GITHUB_ARJUN = ("https://raw.githubusercontent.com/arjunpahi/"
                "Natural_MDM2_Inhibitor_Discovery_using_ML/main")
DATA_CANDIDATES = [
    "../00_data/training/ro5_properties_filtered.csv",
    "00_data/training/ro5_properties_filtered.csv",
    "../Part_2/ro5_properties_filtered.csv",
    "/content/00_data/training/ro5_properties_filtered.csv",
    f"{GITHUB_ARJUN}/Part_2/ro5_properties_filtered.csv",
]
df = None
for c in DATA_CANDIDATES:
    try:
        df = pd.read_csv(c)
        print("[S05] loaded:", c)
        break
    except Exception:
        continue
assert df is not None, ("training CSV not found locally and the GitHub download failed — "
                        "on Kaggle enable Internet in the notebook settings")
assert len(df) == 645, f"expected 645 compounds, got {len(df)}"

SMILES_COL, LABEL_COL, PIC50_COL = "canonical_smiles", "bioactivity_class", "pIC50"
smiles_all = df[SMILES_COL].astype(str).tolist()
y_cls_all = (df[LABEL_COL].astype(str).str.title() == "Active").astype(int).to_numpy()
y_reg_all = df[PIC50_COL].astype(float).to_numpy()
print(f"[S05] {len(df)} compounds | Active {int(y_cls_all.sum())} / "
      f"Inactive {int((1 - y_cls_all).sum())} | pIC50 {y_reg_all.min():.2f}-{y_reg_all.max():.2f}")
print("[S05] source: ChEMBL MDM2-p53 bioactivity -> 2,571 raw records -> 1,933 unique "
      "compounds -> 645 after Lipinski-RO5 filtering (Arjun sir pipeline, Part_2)")
print("[S05] columns: molecule_chembl_id | canonical_smiles | bioactivity_class "
      "(Active/Inactive) | pIC50 = -log10(IC50 M) | MW / HBA / HBD / logP | ro5_fulfilled")
print("[S05] labels: 1 = Active, 0 = Inactive -> every classifier outputs softmax "
      "probabilities (prob_class_1 = P(Active), same schema as screening_results.csv)")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].bar(["Inactive", "Active"], [int((1 - y_cls_all).sum()), int(y_cls_all.sum())],
            color=["#8a8a8a", "#2a7f62"])
axes[0].set_title("Bioactivity classes")
axes[1].hist(y_reg_all, bins=30, color="#2a7f62", alpha=0.85)
axes[1].set_title("pIC50 distribution")
axes[2].scatter(df["molecular_weight"], df["logp"], c=y_cls_all, cmap="coolwarm", s=14, alpha=0.6)
axes[2].set_xlabel("MW"); axes[2].set_ylabel("logP"); axes[2].set_title("RO5 space (orange=Active)")
plt.tight_layout(); plt.savefig(OUT_DIR / "eda.png", dpi=120); plt.show()
'''

# --- scaffold split block (identical in all notebooks; mirrors Part_22 S10) ---
SPLIT_SRC = r'''
# --- Leak-free Murcko scaffold split -> EXACTLY 464/52/129 (Part_20 algorithm, seed 57) ---
from collections import Counter, defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import StandardScaler

# drop any unparseable SMILES first so every feature array stays row-aligned
valid = np.array([Chem.MolFromSmiles(s) is not None for s in smiles_all])
if not valid.all():
    keep = np.where(valid)[0]
    smiles_all = [smiles_all[i] for i in keep]
    y_cls_all = y_cls_all[keep]; y_reg_all = y_reg_all[keep]
    print(f"[S06] dropped {int((~valid).sum())} unparseable rows")

def murcko(smi):
    # RDKit 2026 API: keyword required (Part_20 fix); whole scaffolds never straddle splits
    try:
        m = Chem.MolFromSmiles(str(smi))
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m is not None else "INVALID"
    except Exception:
        return "INVALID"

scaffolds_all = np.array([murcko(s) for s in smiles_all])
n_uniq = len(set(scaffolds_all))
print("[S06] unique Murcko scaffolds:", n_uniq, "| INVALID:", int((scaffolds_all == "INVALID").sum()))
assert n_uniq >= 50, "[S06] scaffold collapse detected - Murcko API may have changed!"

groups = defaultdict(list)
for _i, _s in enumerate(scaffolds_all):
    groups[_s].append(_i)
TEST_SIZE, VAL_FRAC = 0.20, 0.10
n_test_target = int(round(TEST_SIZE * len(scaffolds_all)))
order = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))  # largest scaffolds first
te_list = []
for _s, _ii in order:
    if len(te_list) + len(_ii) <= n_test_target:
        te_list.extend(_ii)
test_idx = np.array(sorted(te_list), dtype=int)
tr_all = np.array(sorted(set(range(len(scaffolds_all))) - set(te_list)), dtype=int)
# Validation: stratified 10% of the non-test remainder, seed 57 (exactly Part_20).
# The 129-compound TEST set is the scaffold-disjoint holdout; val is used only
# for early stopping / tuning and is never scored as a test set.
tr_idx, va_idx = train_test_split(tr_all, test_size=VAL_FRAC, random_state=RANDOM_SEED,
                                  stratify=y_cls_all[tr_all])
tr_idx, va_idx = np.array(tr_idx), np.array(va_idx)
assert len(set(tr_idx) & set(va_idx)) == 0 and len(set(tr_idx) & set(test_idx)) == 0 \
    and len(set(va_idx) & set(test_idx)) == 0, "index overlap between splits"

def _canon(s):
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m) if m is not None else str(s)

canon_tr = {_canon(s) for s in np.asarray(smiles_all)[tr_idx]}
canon_te = {_canon(s) for s in np.asarray(smiles_all)[test_idx]}
assert not (canon_tr & canon_te), "canonical SMILES overlap between train and test!"

Y_REG_MEAN = float(np.mean(y_reg_all[tr_idx]))
Y_REG_STD = float(np.std(y_reg_all[tr_idx])) or 1.0
tr_va_idx = np.array(sorted(set(tr_idx.tolist()) | set(va_idx.tolist())))   # train+val pool for CV
_te_scaf = Counter(scaffolds_all[test_idx])
print(f"[S06] split: train {len(tr_idx)} | val {len(va_idx)} | test {len(test_idx)} "
      f"| Active tr/va/te = {y_cls_all[tr_idx].sum()}/{y_cls_all[va_idx].sum()}/{y_cls_all[test_idx].sum()}"
      f" | scaffold overlap train-test = "
      f"{len({scaffolds_all[i] for i in tr_idx} & {scaffolds_all[i] for i in test_idx})}"
      f" | test Murcko scaffolds={len(_te_scaf)} largest={_te_scaf.most_common(3)}"
      " | test scored ONCE (section 13)")
if len(_te_scaf) <= 10:
    print("[S06][WARN] test set has very few Murcko scaffolds; high holdout AUC may "
          "reflect scaffold/series separation, not prospective molecule-level performance.")
'''

# --- graph featurizer block (mirrors 02_gnn_models/graph_featurizer.py) ------
GRAPH_FEAT_SRC = r'''
# --- Molecular graph featurizer ([ADAPTED] from 02_gnn_models/graph_featurizer.py) ---
from torch_geometric.data import Data

HYB_MAP = {"SP": 0, "SP2": 1, "SP3": 2, "SP3D": 3, "SP3D2": 4}
BOND_MAP = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1,
            Chem.BondType.TRIPLE: 2, Chem.BondType.AROMATIC: 3}

def atom_feat(a):
    f = [0.0] * 78                                  # same 78-dim layout as graph_featurizer.py
    z = a.GetAtomicNum()
    if 1 <= z <= 53:
        f[z - 1] = 1.0                              # 0:53  atomic number one-hot
    if a.GetDegree() <= 5:
        f[53 + a.GetDegree()] = 1.0                 # 53:59 degree
    fc = a.GetFormalCharge()
    if -2 <= fc <= 3:
        f[59 + fc + 2] = 1.0                        # 59:65 formal charge
    hs = a.GetTotalNumHs()
    if hs <= 4:
        f[65 + hs] = 1.0                            # 65:70 num Hs
    hyb = HYB_MAP.get(str(a.GetHybridization()), -1)
    if 0 <= hyb <= 4:
        f[70 + hyb] = 1.0                           # 70:75 hybridization
    f[75] = float(a.GetIsAromatic())
    f[76] = float(a.IsInRing())
    f[77] = float(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED)
    return f

def bond_feat(b):
    f = [0.0] * 6                                   # single/double/triple/aromatic/conjugated/ring
    f[BOND_MAP.get(b.GetBondType(), 0)] = 1.0
    f[4] = float(b.GetIsConjugated())
    f[5] = float(b.IsInRing())
    return f

def mol_to_graph(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None or m.GetNumAtoms() == 0:
        return None
    x = torch.tensor(np.array([atom_feat(a) for a in m.GetAtoms()], dtype=np.float32))
    ei, ea = [], []
    for b in m.GetBonds():                          # edges emitted as adjacent i->j, j->i pairs
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = bond_feat(b)
        ei += [[i, j], [j, i]]
        ea += [bf, bf]
    if ei:
        edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.array(ea, dtype=np.float32))
    else:                                               # single-atom molecule: keep 2-D shapes
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 6), dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

graphs_base = [mol_to_graph(s) for s in smiles_all]
bad = [i for i, g in enumerate(graphs_base) if g is None]
assert not bad, f"{len(bad)} unparseable SMILES survived the validity filter"
print(f"[S06] graphs built: {len(graphs_base)} (78-dim atoms, 6-dim bonds, both directions)")
'''

FEAT_NB1_SRC = r'''
# --- NB1 features: 10 RDKit descriptors (MLP) + graphs (GNNs) + char codes (CNN) — NO fingerprints ---
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

DESC_NAMES = ["MW", "logP", "TPSA", "rotB", "HBA", "HBD", "heavy", "rings", "arom", "fracSP3"]
_mols = [Chem.MolFromSmiles(s) for s in smiles_all]
print("[S06] valid SMILES:", sum(m is not None for m in _mols), "/", len(_mols))

def _desc_row(m):
    if m is None:
        return [0.0] * 10
    return [Descriptors.MolWt(m), Crippen.MolLogP(m), rdMolDescriptors.CalcTPSA(m),
            Descriptors.NumRotatableBonds(m), rdMolDescriptors.CalcNumHBA(m),
            rdMolDescriptors.CalcNumHBD(m), float(m.GetNumHeavyAtoms()),
            float(rdMolDescriptors.CalcNumRings(m)),
            float(rdMolDescriptors.CalcNumAromaticRings(m)), Descriptors.FractionCSP3(m)]

X_desc = np.nan_to_num(np.array([_desc_row(m) for m in _mols], dtype=np.float32))
desc_scaler = StandardScaler().fit(X_desc[tr_idx])          # fit on TRAIN only (no leakage)
X_desc_z = desc_scaler.transform(X_desc).astype(np.float32)

CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789[]()#+-=@%/\\.") + \
    ["PAD", "UNK", "BOS", "EOS"]
VOCAB = {c: i + 1 for i, c in enumerate(CHARS)}
VOCAB_SIZE = max(VOCAB.values()) + 1          # PAD=0 ... max token, no off-by-one

def encode_smiles(sl, max_len=96):
    out = np.zeros((len(sl), max_len), dtype=np.int64)      # PAD token = 0 (padding_idx)
    for i, s in enumerate(sl):
        for j, c in enumerate(str(s)[:max_len]):
            out[i, j] = VOCAB.get(c, VOCAB["UNK"])
    return out

X_chars = encode_smiles(smiles_all)
FEATS = {"desc": X_desc_z, "chars": X_chars}
print(f"[S06] descriptors {X_desc_z.shape} (standardised) | char codes {X_chars.shape}"
      f" | vocab {len(VOCAB)} chars")
'''

FEAT_FP_SRC = r'''
# --- %%NB_FP_TITLE%% featurization + graphs for the hybrid model ---
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors, rdFingerprintGenerator
%%FP_IMPORT%%

%%FP_GEN%%

def _fpdense(fps, nbits):
    out = np.zeros((len(fps), nbits), dtype=np.float32)
    for i, fp in enumerate(fps):
        if fp is not None:
            DataStructs.ConvertToNumpyArray(fp, out[i])
    return out

_mols = [Chem.MolFromSmiles(s) for s in smiles_all]
print("[S06] valid SMILES:", sum(m is not None for m in _mols), "/", len(_mols))
X_fp = _fpdense([_make_fp(m) for m in _mols], %%FP_DIM%%)
FP_DIM = int(X_fp.shape[1])
bit_rate = X_fp.mean(axis=0).mean()
print(f"[S06] %%FP_NAME%% matrix {X_fp.shape} | mean bit-on rate {bit_rate:.3f} "
      f"| active-vs-inactive mean bits "
      f"{X_fp[y_cls_all == 1].mean():.2f} vs {X_fp[y_cls_all == 0].mean():.2f}")
FEATS = {"fp": X_fp}
'''

FP_GEN_MORGAN = r'''# Morgan/ECFP4: radius=2 circular fingerprints hashed to 2048 bits (Rogers & Hahn 2010)
_morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def _make_fp(m):
    return _morgan_gen.GetFingerprint(m) if m is not None else None'''

FP_GEN_MACCS = r'''from rdkit.Chem import MACCSkeys

# MACCS: 167 structural key bits (Durant et al. 2002) — substructure presence/absence
def _make_fp(m):
    return MACCSkeys.GenMACCSKeys(m) if m is not None else None'''

# --- model definition cells --------------------------------------------------

MODELS_NB1_SRC = r'''
# ## 7. Model Architectures — graphs, sequences, descriptors (no fingerprints)
# [ADAPTED] from 02_gnn_models/models/* + Part_22 S13 (validated). Every factory call is
# KEYWORD-ONLY (a positional call once shrank GCN to 411 params — do not repeat that bug).
from torch_geometric.nn import (GCNConv, GATConv, GATv2Conv, GINConv, GINEConv,
                                SAGEConv, ResGatedGraphConv, AttentiveFP,
                                global_mean_pool, global_max_pool)
from torch_geometric.utils import scatter

HIDDEN = 128

class GCNNet(nn.Module):
    """Kipf & Welling 2017 — 3x GCNConv + BN, mean+max pool."""
    def __init__(self, in_dim=78, hidden=HIDDEN, dropout=0.2, out_dim=2):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList([GCNConv(in_dim if i == 0 else hidden, hidden) for i in range(3)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(3)])
        self.head = nn.Sequential(nn.Linear(hidden * 2, 64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64, out_dim))

    def forward(self, d):
        x, ei, bat = d.x.float(), d.edge_index, d.batch
        for conv, bn in zip(self.convs, self.bns):
            x = F.dropout(torch.relu(bn(conv(x, ei))), self.dropout, self.training)
        h = torch.cat([global_mean_pool(x, bat), global_max_pool(x, bat)], dim=1)
        return self.head(h)

class GATNet(nn.Module):
    """Velickovic 2018 (GAT) / Brody 2022 (GATv2) — 2 layers, 4 heads concatenated."""
    def __init__(self, in_dim=78, hidden=HIDDEN, heads=4, dropout=0.2, out_dim=2, v2=False):
        super().__init__()
        conv = GATv2Conv if v2 else GATConv
        self.dropout = dropout
        self.l1 = conv(in_dim, hidden, heads=heads)
        self.l2 = conv(hidden * heads, hidden, heads=heads)
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden * heads) for _ in range(2)])
        self.head = nn.Sequential(nn.Linear(hidden * heads * 2, 64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64, out_dim))

    def forward(self, d):
        x, ei, bat = d.x.float(), d.edge_index, d.batch
        x = F.dropout(torch.relu(self.bns[0](self.l1(x, ei))), self.dropout, self.training)
        x = torch.relu(self.bns[1](self.l2(x, ei)))
        h = torch.cat([global_mean_pool(x, bat), global_max_pool(x, bat)], dim=1)
        return self.head(h)

class GINNet(nn.Module):
    """Xu 2019 GIN / MolCLR-style — GINConv(Linear+BN+ReLU), learnable eps."""
    def __init__(self, in_dim=78, hidden=HIDDEN, layers=3, dropout=0.3, out_dim=2):
        super().__init__()
        self.layers = nn.ModuleList(
            [GINConv(nn.Sequential(nn.Linear(in_dim if i == 0 else hidden, hidden),
                                   nn.BatchNorm1d(hidden), nn.ReLU()),
                     train_eps=True) for i in range(layers)])
        self.head = nn.Sequential(nn.Linear(hidden * 2, 128), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(128, out_dim))

    def forward(self, d):
        x, ei, bat = d.x.float(), d.edge_index, d.batch
        h = x
        for og in self.layers:
            h = og(h, ei)
        h = torch.cat([global_mean_pool(h, bat), global_max_pool(h, bat)], dim=1)
        return self.head(h)

class GINENet(nn.Module):
    """Hu et al. ICLR 2020 GINE — GIN with edge features. Edge-aware message passing
    makes bond information explicit (GCN/GIN in this notebook ignore edge_attr)."""
    def __init__(self, in_dim=78, edge_dim=6, hidden=HIDDEN, layers=3, dropout=0.3, out_dim=2):
        super().__init__()
        self.layers = nn.ModuleList([
            GINEConv(nn.Sequential(nn.Linear(in_dim if i == 0 else hidden, hidden),
                                   nn.BatchNorm1d(hidden), nn.ReLU()),
                     train_eps=True, edge_dim=edge_dim) for i in range(layers)])
        self.head = nn.Sequential(nn.Linear(hidden * 2, 128), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(128, out_dim))

    def forward(self, d):
        h, ei, ea, bat = d.x.float(), d.edge_index, d.edge_attr.float(), d.batch
        for og in self.layers:
            h = og(h, ei, ea)
        h = torch.cat([global_mean_pool(h, bat), global_max_pool(h, bat)], dim=1)
        return self.head(h)

class GraphSAGENet(nn.Module):
    """Hamilton et al. NeurIPS 2017 GraphSAGE — inductive neighborhood sampling/aggregation;
    here mean aggregation with full-graph message passing on small molecular graphs."""
    def __init__(self, in_dim=78, hidden=HIDDEN, dropout=0.2, out_dim=2):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList([SAGEConv(in_dim if i == 0 else hidden, hidden) for i in range(3)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(3)])
        self.head = nn.Sequential(nn.Linear(hidden * 2, 64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64, out_dim))

    def forward(self, d):
        x, ei, bat = d.x.float(), d.edge_index, d.batch
        for conv, bn in zip(self.convs, self.bns):
            x = F.dropout(torch.relu(bn(conv(x, ei))), self.dropout, self.training)
        h = torch.cat([global_mean_pool(x, bat), global_max_pool(x, bat)], dim=1)
        return self.head(h)

class GatedGCNNet(nn.Module):
    """Bresson & Laurent 2017 Residual Gated Graph ConvNet (ResGatedGraphConv in PyG) —
    edge-gated message passing; uses both node and bond features."""
    def __init__(self, in_dim=78, edge_dim=6, hidden=HIDDEN, dropout=0.2, out_dim=2):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList([
            ResGatedGraphConv(in_dim if i == 0 else hidden, hidden, edge_dim=edge_dim)
            for i in range(3)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(3)])
        self.head = nn.Sequential(nn.Linear(hidden * 2, 64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64, out_dim))

    def forward(self, d):
        h, ei, ea, bat = d.x.float(), d.edge_index, d.edge_attr.float(), d.batch
        for conv, bn in zip(self.convs, self.bns):
            h = F.dropout(torch.relu(bn(conv(h, ei, ea))), self.dropout, self.training)
        h = torch.cat([global_mean_pool(h, bat), global_max_pool(h, bat)], dim=1)
        return self.head(h)

class AttentiveFPNet(nn.Module):
    """Xiong 2020 AttentiveFP — graph attention with edge features (PyG implementation)."""
    def __init__(self, in_dim=78, edge_dim=6, hidden=HIDDEN, dropout=0.2, out_dim=2):
        super().__init__()
        self.net = AttentiveFP(in_channels=in_dim, hidden_channels=hidden, out_channels=out_dim,
                               edge_dim=edge_dim, num_layers=3, num_timesteps=2, dropout=dropout)

    def forward(self, d):
        return self.net(d.x.float(), d.edge_index, d.edge_attr.float(), d.batch)

class DMPNN(nn.Module):
    """Yang 2019 D-MPNN (Chemprop) — directed message passing, rev = arange ^ 1."""
    def __init__(self, in_dim=78, edge_dim=6, hidden=HIDDEN, depth=3, dropout=0.2, out_dim=2):
        super().__init__()
        self.depth, self.dropout = depth, dropout
        self.W_i = nn.Linear(in_dim + edge_dim, hidden)
        self.W_h = nn.Linear(hidden, hidden)
        self.W_o = nn.Linear(hidden, hidden)
        self.head = nn.Sequential(nn.Linear(hidden * 2, 64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64, out_dim))

    def forward(self, d):
        x, ei, ea, bat = d.x.float(), d.edge_index, d.edge_attr.float(), d.batch
        src, dst = ei[0], ei[1]
        rev = torch.arange(src.size(0), device=src.device) ^ 1   # bonds stored as i->j, j->i pairs
        e0 = self.W_i(torch.cat([x[src], ea], dim=1))
        m = e0.clone()
        for _ in range(self.depth):
            in_sum = scatter(m, dst, dim=0, dim_size=d.num_nodes, reduce="sum")
            m = F.dropout(F.relu(e0 + self.W_h(in_sum[src] - m[rev])), self.dropout, self.training)
        node_h = F.relu(self.W_o(scatter(m, dst, dim=0, dim_size=d.num_nodes, reduce="sum")))
        h = torch.cat([global_mean_pool(node_h, bat), global_max_pool(node_h, bat)], dim=1)
        return self.head(F.relu(h))

class HybridGCNGAT(nn.Module):
    """Parallel GCN + GAT branches, pooled features concatenated (02_gnn_models hybrid)."""
    def __init__(self, in_dim=78, hidden=HIDDEN, heads=4, dropout=0.2, out_dim=2):
        super().__init__()
        self.g1, self.g2 = GCNConv(in_dim, hidden), GCNConv(hidden, hidden)
        self.a1 = GATConv(in_dim, hidden, heads=heads, concat=False)
        self.a2 = GATConv(hidden, hidden, heads=heads, concat=False)
        self.bn1, self.bn2 = nn.BatchNorm1d(hidden), nn.BatchNorm1d(hidden)
        self.head = nn.Sequential(nn.Linear(hidden * 4, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, out_dim))

    def forward(self, d):
        x, ei, bat = d.x.float(), d.edge_index, d.batch
        xg = torch.relu(self.bn1(self.g1(x, ei)))
        xg = torch.relu(self.bn2(self.g2(xg, ei)))
        xa = torch.relu(self.a1(x, ei))
        xa = torch.relu(self.a2(xa, ei))
        hg = torch.cat([global_mean_pool(xg, bat), global_max_pool(xg, bat)], 1)
        ha = torch.cat([global_mean_pool(xa, bat), global_max_pool(xa, bat)], 1)
        return self.head(torch.cat([hg, ha], dim=1))

class SmilesCNN(nn.Module):
    """Char-level 1-D CNN over SMILES strings (Part_22 SmilesCNN)."""
    def __init__(self, vocab_size=None, in_dim=None, hidden=HIDDEN, dropout=0.2, out_dim=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size or VOCAB_SIZE, 96, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(96, hidden, k, padding=k // 2) for k in (3, 5, 7)])
        self.head = nn.Sequential(nn.Linear(hidden * 3, 64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64, out_dim))

    def forward(self, codes):
        z = self.emb(codes).transpose(1, 2)
        f = [F.max_pool1d(F.relu(c(z)), z.size(2)).squeeze(2) for c in self.convs]
        return self.head(torch.cat(f, dim=1))

class DescriptorMLP(nn.Module):
    """Plain MLP on the 10 standardized RDKit descriptors."""
    def __init__(self, in_dim=10, hidden=128, dropout=0.2, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim))

    def forward(self, x):
        return self.net(x)

class GraphConvLayer(nn.Module):
    """DeepChem GraphConv-style layer, pure torch (mean aggregation).
    [ADAPTED] from 02_gnn_models/deepchem_gcn_model.py (Arjun sir Part_9)."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin_neigh = nn.Linear(in_dim, out_dim)
        self.lin_self = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        x = x.float()                       # Linear weights are float32; keep math consistent
        src, dst = edge_index[0], edge_index[1]
        agg = x.new_zeros(x.size(0), x.size(1))
        agg = agg.index_add_(0, dst, x[src])                     # sum over neighbours
        deg = x.new_zeros(x.size(0))
        deg = deg.index_add_(0, dst, torch.ones_like(src, dtype=x.dtype))
        return F.relu(self.lin_neigh(agg / deg.clamp(min=1.0).unsqueeze(1))
                      + self.lin_self(x))

class DeepChemGCN(nn.Module):
    """Arjun sir's Part_9 DeepChem GCN, rebuilt in pure torch (same GraphConv math:
    neighbourhood mean + self transform -> ReLU, stacked, mean+max pool head)."""
    def __init__(self, in_dim=78, hidden=128, layers=2, dropout=0.2, out_dim=2):
        super().__init__()
        dims = [in_dim] + [hidden] * layers
        self.convs = nn.ModuleList(GraphConvLayer(dims[i], dims[i + 1])
                                   for i in range(layers))
        self.bns = nn.ModuleList(nn.BatchNorm1d(d) for d in dims[1:])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden * 2, 256), nn.ReLU(),
                                  nn.Linear(256, 64), nn.ReLU(),
                                  nn.Linear(64, out_dim))

    def forward(self, d):
        x, ei, b = d.x.float(), d.edge_index, d.batch
        for conv, bn in zip(self.convs, self.bns):
            x = self.dropout(bn(conv(x, ei)))
        return self.head(torch.cat([global_mean_pool(x, b),
                                    global_max_pool(x, b)], dim=1))

# registry: name -> (factory, loader kind, feature key)
DL_MODELS = {
    "DescriptorMLP": (DescriptorMLP, "array", "desc"),
    "SmilesCNN":     (SmilesCNN,     "array", "chars"),
    "GCN":           (GCNNet,        "graph", None),
    "GIN":           (GINNet,        "graph", None),
    "GINE":          (GINENet,       "graph", None),
    "GAT":           (GATNet,        "graph", None),
    "GATv2":         (lambda **kw: GATNet(v2=True, **kw), "graph", None),
    "GraphSAGE":     (GraphSAGENet,  "graph", None),
    "GatedGCN":      (GatedGCNNet,   "graph", None),
    "HybridGCNGAT":  (HybridGCNGAT,  "graph", None),
    "AttentiveFP":   (AttentiveFPNet, "graph", None),
    "D-MPNN":        (DMPNN,         "graph", None),
    "DeepChemGCN":   (DeepChemGCN,   "graph", None),
}
DL_PAPERS = {
    "DescriptorMLP": "classic QSAR descriptors -> MLP (this project, 01_traditional_ml lineage)",
    "SmilesCNN": "Crichton & Guo 2017 char-CNN / Part_22 SmilesCNN",
    "GCN": "Kipf & Welling ICLR 2017 | 02_gnn_models/gcn_model.py",
    "GIN": "Xu et al. ICLR 2019 (MolCLR-style) | 02_gnn_models/gin_model.py",
    "GINE": "Hu et al. ICLR 2020 (Strategies for Pre-training GNNs, GINEConv edge-aware GIN) | PyG GINEConv",
    "GAT": "Velickovic et al. ICLR 2018 | 02_gnn_models/gat_model.py",
    "GATv2": "Brody et al. ICLR 2022 (PyG GATv2Conv)",
    "GraphSAGE": "Hamilton et al. NeurIPS 2017 (inductive representation learning) | PyG SAGEConv",
    "GatedGCN": "Bresson & Laurent 2017 (Residual Gated Graph ConvNets) | PyG ResGatedGraphConv",
    "HybridGCNGAT": "Part_13 hybrid (this project)",
    "AttentiveFP": "Xiong et al. J Med Chem 2020 | 02_gnn_models/attentivefp_model.py",
    "D-MPNN": "Yang et al. JCIM 2019 (chemprop) | 02_gnn_models/mpnn_model.py",
    "DeepChemGCN": "Ramsundar et al. deepchem GraphConv — Arjun sir Part_9 | 02_gnn_models/deepchem_gcn_model.py",
}
print("[S07] NB1 models:", sorted(DL_MODELS))
try:
    import deepchem  # noqa: F401
    HAS_DEEPCHEM = True
    print("[S07] deepchem package available -> real GCNModel cell (S09b) will also run")
except Exception:
    HAS_DEEPCHEM = False
    print("[S07] deepchem not installed -> S09b real GCNModel skipped "
          "(pure-torch DeepChemGCN still trains above)")
'''

MODELS_FP_SRC = r'''
# ## 7. Model Architectures — deep nets on %%FP_NAME%% (+ one graph hybrid)
# Fingerprints enter as plain bit vectors -> MLP / residual MLP / 1-D CNN families,
# plus a GCN encoder late-fused with the %%FP_NAME%% branch. Keyword-only factories.
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

HIDDEN = 128

class FPMlpShallow(nn.Module):
    """One hidden layer on the bit vector — fast, strong baseline."""
    def __init__(self, in_dim=%%FP_DIM%%, hidden=512, dropout=0.2, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, out_dim))

    def forward(self, x):
        return self.net(x)

class FPMlpDeep(nn.Module):
    """Deeper MLP: %%FP_DIM%% -> 1024 -> 512 -> 256 -> out."""
    def __init__(self, in_dim=%%FP_DIM%%, hidden=256, dropout=0.3, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, out_dim))

    def forward(self, x):
        return self.net(x)

class _ResBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x):
        h = F.dropout(F.relu(self.bn(self.fc1(x))), self.dropout, self.training)
        return F.relu(x + self.fc2(h))

class ResidualMlp(nn.Module):
    """ResNet-style MLP with skip connections over the bit vector."""
    def __init__(self, in_dim=%%FP_DIM%%, hidden=512, blocks=3, dropout=0.2, out_dim=2):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden) if in_dim != hidden else nn.Identity()
        self.blocks = nn.ModuleList([_ResBlock(hidden, dropout) for _ in range(blocks)])
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x):
        h = self.proj(x)
        for b in self.blocks:
            h = b(h)
        return self.head(h)

class FingerPrintCNN(nn.Module):
    """Treat the bit vector as a 1-D signal: Conv1d stack + adaptive pooling."""
    def __init__(self, in_dim=%%FP_DIM%%, hidden=128, dropout=0.2, out_dim=2):
        super().__init__()
        self.c1 = nn.Conv1d(1, 64, 7, padding=3)
        self.c2 = nn.Conv1d(64, 64, 5, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(16)      # shape-agnostic -> flatten 64*16 = 1024
        self.head = nn.Sequential(nn.Linear(64 * 16, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, out_dim))

    def forward(self, x):
        z = x.unsqueeze(1)                        # (B, 1, nbits)
        z = F.max_pool1d(F.relu(self.c1(z)), 4)
        z = self.pool(F.relu(self.c2(z)))
        return self.head(z.flatten(1))

class HybridGCNFp(nn.Module):
    """GCN graph encoder late-fused with the %%FP_NAME%% branch (best of both worlds)."""
    def __init__(self, in_dim=78, fp_dim=%%FP_DIM%%, hidden=HIDDEN, dropout=0.2, out_dim=2):
        super().__init__()
        self.g1, self.g2 = GCNConv(in_dim, hidden), GCNConv(hidden, hidden)
        self.bn1, self.bn2 = nn.BatchNorm1d(hidden), nn.BatchNorm1d(hidden)
        self.fp_branch = nn.Sequential(nn.Linear(fp_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, out_dim))

    def forward(self, d):
        x, ei, bat = d.x.float(), d.edge_index, d.batch
        xg = torch.relu(self.bn1(self.g1(x, ei)))
        xg = torch.relu(self.bn2(self.g2(xg, ei)))
        hg = torch.cat([global_mean_pool(xg, bat), global_max_pool(xg, bat)], 1)
        hf = self.fp_branch(d.fp)                 # fingerprint carried on the batch
        return self.head(torch.cat([hg, hf], dim=1))

DL_MODELS = {
    "FPMlpShallow":   (FPMlpShallow,   "array", "fp"),
    "FPMlpDeep":      (FPMlpDeep,      "array", "fp"),
    "ResidualMlp":    (ResidualMlp,    "array", "fp"),
    "FingerPrintCNN": (FingerPrintCNN, "array", "fp"),
    "HybridGCNFp":    (HybridGCNFp,    "graph", "fp"),   # hybrid consumes graphs + fp branch
}
DL_PAPERS = {
    "FPMlpShallow": "MLP on circular fingerprints (standard QSAR-DL baseline)",
    "FPMlpDeep": "deep MLP (this project, Part_12 lineage)",
    "ResidualMlp": "He et al. CVPR 2016 residual idea applied to fingerprints",
    "FingerPrintCNN": "1-D CNN over bit vectors (this project, Part_14 lineage)",
    "HybridGCNFp": "%%FP_PAPER%%",
}

# sanity: graph-level 'fp' attribute must collate to [num_graphs, nbits] in a Batch
from torch_geometric.loader import DataLoader as _TDL
_probe = next(iter(_TDL([Data(x=torch.zeros(3, 78), edge_index=torch.zeros(2, 4, dtype=torch.long),
                              edge_attr=torch.zeros(4, 6), fp=torch.ones(1, %%FP_DIM%%))
                         for _ in range(2)], batch_size=2)))
assert tuple(_probe.fp.shape) == (2, %%FP_DIM%%), f"fp collated wrong: {_probe.fp.shape}"
print("[S07] hybrid fp collate check ok:", tuple(_probe.fp.shape))
print("[S07] NB%%NB_NUM%% models:", sorted(DL_MODELS))
'''

MODELS_T1_SRC = r'''
# ## 7. Tier-1 Model Architectures — attention & aggregation advances
# [USER] implementations adapted from the cited papers; protocol (split, engine,
# guardrails) is identical to DL 01-03. Every factory is KEYWORD-ONLY (see DL 01 note).
from torch_geometric.nn import (PNAConv, TransformerConv, GPSConv, GCNConv,
                                global_mean_pool, global_max_pool)

HIDDEN = 128

# --- degree histogram for PNA scalers — computed on TRAIN split only (leak-free) ---
_max_deg = 1
for g in graphs_base:
    if g.edge_index.numel():
        _max_deg = max(_max_deg, int(g.edge_index[0].max()) + 1)
_e0 = torch.cat([graphs_base[i].edge_index[0] for i in tr_idx])
_deg = torch.bincount(_e0, minlength=_max_deg).float()
print(f"[S07] PNA degree histogram: max degree {_max_deg - 1} | from {len(tr_idx)} train graphs")

class PNANet(nn.Module):
    """Corso, Cavalleri et al., NeurIPS 2020 — Principal Neighbourhood Aggregation (PNA):
    mean/min/max/std aggregators + degree-aware scalers (identity/amplification/attenuation)."""
    def __init__(self, in_dim=78, hidden=HIDDEN, edge_dim=6, dropout=0.3, out_dim=2):
        super().__init__()
        agg, sc = ["mean", "min", "max", "std"], ["identity", "amplification", "attenuation"]
        self.convs = nn.ModuleList([
            PNAConv(in_dim if i == 0 else hidden, hidden, aggregators=agg, scalers=sc,
                    deg=_deg, edge_dim=edge_dim) for i in range(3)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(3)])
        self.dropout = dropout
        self.head = nn.Sequential(nn.Linear(hidden * 2, 128), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(128, out_dim))

    def forward(self, d):
        h = d.x.float()
        for conv, bn in zip(self.convs, self.bns):
            h = bn(F.relu(conv(h, d.edge_index, d.edge_attr.float())))
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.cat([global_mean_pool(h, d.batch), global_max_pool(h, d.batch)], dim=1)
        return self.head(h)

class GraphTransformerNet(nn.Module):
    """Shi et al. 2021 (TransformerConv) — multi-head, edge-aware graph attention:
    a transformer-style update where neighbours are the tokens."""
    def __init__(self, in_dim=78, hidden=HIDDEN, edge_dim=6, heads=4, dropout=0.3, out_dim=2):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([
            TransformerConv(hidden, hidden // heads, heads=heads, concat=True,
                            edge_dim=edge_dim, dropout=dropout) for _ in range(2)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(2)])
        self.head = nn.Sequential(nn.Linear(hidden * 2, 128), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(128, out_dim))

    def forward(self, d):
        h = F.relu(self.inp(d.x.float()))
        for conv, bn in zip(self.convs, self.bns):
            h = bn(F.relu(conv(h, d.edge_index, d.edge_attr.float())))
        h = torch.cat([global_mean_pool(h, d.batch), global_max_pool(h, d.batch)], dim=1)
        return self.head(h)

class GPSNet(nn.Module):
    """Rampasek et al., Nature Machine Intelligence 2022 — GPS (General, Powerful, Scalable):
    every layer combines a LOCAL message-passing conv with GLOBAL multi-head attention.
    Mini version: GCNConv local path + 4-head attention per GPSConv layer."""
    def __init__(self, in_dim=78, hidden=HIDDEN, dropout=0.2, out_dim=2):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.PReLU())
        self.layers = nn.ModuleList([
            GPSConv(hidden, conv=GCNConv(hidden, hidden), heads=4, dropout=dropout)
            for _ in range(2)])
        self.head = nn.Sequential(nn.Linear(hidden * 2, 128), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(128, out_dim))

    def forward(self, d):
        h = self.inp(d.x.float())
        for l in self.layers:
            h = l(h, d.edge_index, batch=d.batch)
        h = torch.cat([global_mean_pool(h, d.batch), global_max_pool(h, d.batch)], dim=1)
        return self.head(h)

VOCAB_SIZE = max(VOCAB.values()) + 2

class SmilesTransformer(nn.Module):
    """Vaswani et al., NeurIPS 2017 — transformer encoder, applied char-level to SMILES.
    Small 2-layer encoder + masked-mean pooling over non-PAD characters."""
    def __init__(self, in_dim=96, hidden=96, nhead=4, layers=2, dropout=0.2, out_dim=2):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, hidden, padding_idx=0)
        self.pos = nn.Embedding(in_dim, hidden)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(hidden, nhead, dim_feedforward=4 * hidden,
                                           dropout=dropout, batch_first=True,
                                           norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64, out_dim))

    def forward(self, x):
        x = x.long()                                   # array path delivers float
        p = torch.arange(x.size(1), device=x.device)
        h = self.drop(self.emb(x) + self.pos(p)[None])
        pad = (x == 0)
        h = self.enc(h, src_key_padding_mask=pad)
        h = h.masked_fill(pad.unsqueeze(-1), 0.0).sum(1) / (~pad).sum(1).clamp(min=1).unsqueeze(-1)
        return self.head(h)

class FTTransformer(nn.Module):
    """Gorishniy, Rubachev et al., NeurIPS 2021 — FT-Transformer: every feature becomes a
    token (per-feature linear projection), learnable CLS token, transformer encoder, CLS
    feeds the head. The SOTA tabular-DL architecture."""
    def __init__(self, in_dim=10, hidden=64, nhead=4, layers=2, dropout=0.2, out_dim=2):
        super().__init__()
        self.tokens = nn.Linear(1, hidden)
        self.cls = nn.Parameter(torch.zeros(hidden))
        layer = nn.TransformerEncoderLayer(hidden, nhead, dim_feedforward=4 * hidden,
                                           dropout=dropout, batch_first=True,
                                           norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, out_dim))

    def forward(self, x):
        h = self.tokens(x.float().unsqueeze(-1))       # [B, F, d]
        cls = self.cls.unsqueeze(0).expand(x.size(0), -1).unsqueeze(1)
        h = self.enc(torch.cat([cls, h], dim=1))
        return self.head(h[:, 0])

DL_MODELS = {
    "PNA":               (PNANet, "graph", None),
    "GraphTransformer":  (GraphTransformerNet, "graph", None),
    "GPS":               (GPSNet, "graph", None),
    "SmilesTransformer": (SmilesTransformer, "array", "chars"),
    "FTTransformer":     (FTTransformer, "array", "desc"),
}
DL_PAPERS = {
    "PNA":               "Corso, Cavalleri et al., NeurIPS 2020 - Principal Neighbourhood Aggregation",
    "GraphTransformer":  "Shi et al. 2021 - TransformerConv, multi-head edge-aware graph attention",
    "GPS":               "Rampasek et al., Nature Machine Intelligence 2022 - GPS graph transformer",
    "SmilesTransformer": "Vaswani et al., NeurIPS 2017 - transformer encoder on SMILES chars",
    "FTTransformer":     "Gorishniy et al., NeurIPS 2021 - Revisiting Deep Learning for Tabular Data (FT-Transformer)",
}
print("[S07] Tier-1 registry:", ", ".join(DL_MODELS))
'''

MODELS_T2_SRC = r'''
# ## 7. Tier-2 Architectures — GIN from scratch vs GIN pretrained with MolCLR
# Same GINNet class for both models: "GIN (scratch)" starts from random weights,
# "GIN-MolCLR" is warm-started from unlabeled chemistry pretraining in cell 7b (same trunk,
# same split, same engine — the ONLY difference is the initial weights).
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool

HIDDEN = 128

class GINNet(nn.Module):
    """Xu, Hu et al., ICLR 2019 — How Powerful are Graph Neural Networks? (GIN,
    train_eps=True). return_pooled=True exposes the pooled representation for
    contrastive pretraining (used by the MolCLR encoder)."""
    def __init__(self, in_dim=78, hidden=HIDDEN, layers=3, dropout=0.3, out_dim=2):
        super().__init__()
        self.return_pooled = False
        self.layers = nn.ModuleList(
            [GINConv(nn.Sequential(nn.Linear(in_dim if i == 0 else hidden, hidden),
                                   nn.BatchNorm1d(hidden), nn.ReLU()),
                     train_eps=True) for i in range(layers)])
        self.head = nn.Sequential(nn.Linear(hidden * 2, 128), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(128, out_dim))

    def forward(self, d):
        x, ei, bat = d.x.float(), d.edge_index, d.batch
        h = x
        for og in self.layers:
            h = og(h, ei)
        pooled = torch.cat([global_mean_pool(h, bat), global_max_pool(h, bat)], dim=1)
        if self.return_pooled:
            return pooled
        return self.head(pooled)

DL_MODELS = {
    "GIN (scratch)": (GINNet, "graph", None),
    "GIN-MolCLR":    (GINNet, "graph", None),
}
DL_PAPERS = {
    "GIN (scratch)": "Xu, Hu et al., ICLR 2019 - How Powerful are Graph Neural Networks? (GIN)",
    "GIN-MolCLR":    "Wang et al., Nature Machine Intelligence 2022 - Molecular Contrastive Learning (MolCLR): GIN encoder pretrained on an unlabeled open-source chemistry corpus (default ZINC250k, downloaded/cached; not screening COCONUT), then fine-tuned",
}
print("[S07] Tier-2 registry:", ", ".join(DL_MODELS))
'''

PRETRAIN_T2_SRC = r'''
# ## 7b. MolCLR Self-Supervised Pretraining on an Unlabeled Chemistry Corpus (NO MDM2 labels)
# Wang et al., Nature Machine Intelligence 2022: two stochastic augmented views per
# molecule — atom masking 15%, bond deletion 15%, subgraph removal 25% — pulled together
# by NT-Xent (SimCLR; Chen et al. ICML 2020) at temperature 0.2.
# Default source is ZINC250k (standard open-source, downloaded/cached) — NOT the
# COCONUT screening library. MDM2 labels never touch the pretraining loss.
WARM_START_NAME, WARM_START_SD, WARM_START_TAG = None, None, None
from torch_geometric.data import Data, Batch

import urllib.request, hashlib, shutil

# Default is a standard open-source chemistry corpus, NOT the screening COCONUT library.
PRETRAIN_SOURCE = os.environ.get("DL_PRETRAIN_SOURCE", "zinc250k").strip().lower()
PRETRAIN_CACHE = Path(os.environ.get("DL_PRETRAIN_CACHE", str(OUT_DIR / "pretrain_cache")))
PRETRAIN_CACHE.mkdir(parents=True, exist_ok=True)

_PRETRAIN_SOURCES = {
    # Free/open-source, no labels. Downloaded once and cached locally.
    "coconut": {
        "desc": "COCONUT natural-product library (from screening_results.csv, no labels)",
        "candidates": [
            "../00_data/screening/screening_results.csv",
            "00_data/screening/screening_results.csv",
            "/content/00_data/screening/screening_results.csv",
            # Arjun sir's original public repo (source of the COCONUT screen)
            "https://raw.githubusercontent.com/arjunpahi/Natural_MDM2_Inhibitor_Discovery_using_ML/main/Part_8/screening_results.csv",
        ],
        "smiles_col": "canonical_smiles",
    },
    "zinc250k": {
        "desc": "ZINC250k (250k drug-like molecules, standard open-source MolCLR/JT-VAE benchmark; no labels)",
        "candidates": [
            str(PRETRAIN_CACHE / "250k_rndm_zinc_drugs_clean_3.csv"),
            "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv",
            "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/250k_rndm_zinc_drugs_clean_3.csv",
        ],
        "smiles_col": "smiles",
    },
    "chembl_small": {
        "desc": "ChEMBL small molecule subset (open-source, no activity labels used for pretraining)",
        "candidates": [
            str(PRETRAIN_CACHE / "chembl_24_1_chemreps.txt.gz"),
            "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_24_1/chembl_24_1_chemreps.txt.gz",
        ],
        "smiles_col": "canonical_smiles",
        "sep": "\t",
    },
}

def _download_if_url(src, cache_dir=PRETRAIN_CACHE):
    if not str(src).startswith("http"):
        return src
    name = Path(str(src).split("?")[0]).name or ("pretrain_" + hashlib.md5(src.encode()).hexdigest()[:12] + ".csv")
    dst = cache_dir / name
    if dst.exists() and dst.stat().st_size > 0:
        return str(dst)
    print(f"[S07b] downloading pretraining corpus: {src}")
    try:
        req = urllib.request.Request(src, headers={"User-Agent": "MDM2-DL-pretrain/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as fh:
            shutil.copyfileobj(r, fh)
        print(f"[S07b] cached -> {dst} ({dst.stat().st_size/1e6:.1f} MB)")
        return str(dst)
    except Exception as e:
        print(f"[S07b] download failed: {e}")
        try:
            if dst.exists(): dst.unlink()
        except Exception:
            pass
        return None

def load_pretrain_smiles(source=PRETRAIN_SOURCE):
    spec = _PRETRAIN_SOURCES.get(source)
    if spec is None:
        # treat as path/URL
        spec = {"desc": f"custom source {source}", "candidates": [source], "smiles_col": "canonical_smiles"}
    last_err = None
    for cand in spec["candidates"]:
        resolved = _download_if_url(cand)
        if resolved is None:
            continue
        try:
            sep = spec.get("sep", ",")
            dfp = pd.read_csv(resolved, sep=sep, low_memory=False)
            col = spec["smiles_col"]
            if col not in dfp.columns:
                # common fallbacks
                for alt in ["canonical_smiles", "smiles", "SMILES", "Smiles"]:
                    if alt in dfp.columns:
                        col = alt; break
            if col not in dfp.columns:
                continue
            pool = dfp[col].dropna().astype(str).tolist()
            pool = [s for s in pool if s and s.lower() != "nan"]
            if pool:
                print(f"[S07b] pretrain source: {source} | {spec['desc']}")
                print(f"[S07b] loaded {len(pool):,} SMILES from {resolved} (col={col})")
                return pool, source
        except Exception as e:
            last_err = e
            continue
    raise FileNotFoundError(f"No pretraining SMILES found for source={source}; last_err={last_err}")

_pool, _src_used = load_pretrain_smiles(PRETRAIN_SOURCE)

N_PRETRAIN = int(os.environ.get("DL_PRETRAIN_MAX", 1500 if SMOKE else (4000 if FAST else 30000)))
PRE_EPOCHS = 2 if SMOKE else (3 if FAST else 30)
PRE_BATCH = 64 if DEVICE.type == "cpu" else 128
_rng = np.random.RandomState(RANDOM_SEED)
_pick = _rng.choice(len(_pool), size=min(N_PRETRAIN, len(_pool)), replace=False)
print(f"[S07b] source={_src_used} rows {len(_pool):,} | pretraining on {len(_pick):,} molecules "
      f"({PRE_EPOCHS} epochs, batch {PRE_BATCH}, no labels used)")

def molclr_augment(g, p_mask=0.15, p_bond=0.15, p_sub=0.25):
    """MolCLR augmentation — returns a NEW Data; never mutates the original graph."""
    x, ei, ea = g.x.clone(), g.edge_index.clone(), g.edge_attr.clone()
    n = x.size(0)
    x[torch.rand(n) < p_mask] = 0.0                                  # 1) atom masking
    half = ei.size(1) // 2
    if half:                                                          # 2) bond deletion
        keep_half = torch.rand(half) >= p_bond
        keep = torch.repeat_interleave(keep_half, 2)                  # directed pairs
        ei, ea = ei[:, keep], ea[keep]
    if torch.rand(1).item() < p_sub and n > 4:                        # 3) subgraph removal
        node = int(torch.randint(0, n, (1,)))
        kill = {node}
        kill |= {int(v) for v in ei[0][ei[1] == node].tolist()}
        kill |= {int(v) for v in ei[1][ei[0] == node].tolist()}
        keepn = torch.ones(n, dtype=torch.bool); keepn[list(kill)] = False
        remap = torch.full((n,), -1, dtype=torch.long)
        remap[keepn] = torch.arange(int(keepn.sum()))
        m = keepn[ei[0]] & keepn[ei[1]]
        if int(keepn.sum()) >= 2:
            x, ei, ea = x[keepn], remap[ei[:, m]], ea[m]
    if x.size(0) < 1 or ei.numel() == 0:
        return g
    return Data(x=x, edge_index=ei, edge_attr=ea)

def _make_views(indices):
    """Parse + augment only the current batch: keeps RAM flat for 30k molecules."""
    va, vb = [], []
    for i in indices:
        g = mol_to_graph(_pool[int(i)])
        if g is None:
            continue
        va.append(molclr_augment(g))
        vb.append(molclr_augment(g))
    return va, vb

def nt_xent(z1, z2, T=0.2):
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
    sim = z @ z.t() / T
    n = z1.size(0)
    sim.fill_diagonal_(-1e9)
    target = torch.cat([torch.arange(n) + n, torch.arange(n)]).to(z.device)
    return F.cross_entropy(sim, target)

enc = GINNet().to(DEVICE)
enc.return_pooled = True
proj = nn.Sequential(nn.Linear(2 * HIDDEN, HIDDEN), nn.ReLU(),
                     nn.Linear(HIDDEN, HIDDEN)).to(DEVICE)
opt = torch.optim.Adam(list(enc.parameters()) + list(proj.parameters()), lr=1e-3)
_pars = list(enc.parameters()) + list(proj.parameters())
for ep in range(1, PRE_EPOCHS + 1):
    enc.train(); proj.train()
    perm = _rng.permutation(len(_pick))
    tot, nb = 0.0, 0
    for i0 in range(0, len(perm), PRE_BATCH):
        bi = perm[i0:i0 + PRE_BATCH]
        va, vb = _make_views(_pick[bi])
        if len(va) < 2:                              # NT-Xent needs at least one pair
            continue
        b1 = Batch.from_data_list(va).to(DEVICE)
        b2 = Batch.from_data_list(vb).to(DEVICE)
        loss = nt_xent(proj(enc(b1)), proj(enc(b2)))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(_pars, 5.0)
        opt.step()
        tot += float(loss); nb += 1
        if nb % 50 == 0:
            print(f"[S07b]   epoch {ep} batch {nb} | running NT-Xent {tot / nb:.4f}")
    print(f"[S07b] epoch {ep}/{PRE_EPOCHS} NT-Xent {tot / max(nb, 1):.4f} "
          f"({nb} batches)")

WARM_START_SD = {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()
                 if not k.startswith("head")}
WARM_START_NAME, WARM_START_TAG = "GIN-MolCLR", f"MolCLR pretrain ({_src_used})"
torch.save(WARM_START_SD, OUT_DIR / "molclr_gin_pretrain.pt")
print(f"[S07b] saved {len(WARM_START_SD)} trunk tensors -> molclr_gin_pretrain.pt "
      f"| 'GIN-MolCLR' warm-starts from them (head stays fresh, source={_src_used})")
'''

# --- training engine ---------------------------------------------------------

ENGINE_SRC = r'''
# ## 8. Training Engine — validation-only early stopping, AMP, grad clipping
import copy

EPOCHS = 3 if SMOKE else (8 if FAST else 150)
PATIENCE = 2 if SMOKE else (4 if FAST else 15)
BATCH_GRAPH = auto_batch(64 if DEVICE.type == "cpu" else 128)
BATCH_ARRAY = auto_batch(128 if DEVICE.type == "cpu" else 256)
GRAD_CLIP = 5.0

def task_metrics(y_true, y_pred):
    """cls: AUC (primary) + AP + balanced accuracy | reg: RMSE (primary) + MAE + R2."""
    if TASK == "cls":
        from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
        yt = np.asarray(y_true).astype(int)
        yp = np.asarray(y_pred, dtype=float)
        return {"AUC": float(roc_auc_score(yt, yp)) if len(set(yt.tolist())) > 1 else float("nan"),
                "AP": float(average_precision_score(yt, yp)),
                "BalAcc": float(balanced_accuracy_score(yt, (yp >= 0.5).astype(int)))}
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    yt, yp = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return {"RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
            "MAE": float(mean_absolute_error(yt, yp)), "R2": float(r2_score(yt, yp))}

def _y_task(y_override=None):
    if y_override is not None:
        return np.asarray(y_override, dtype=np.float32)
    return y_cls_all.astype(np.float32) if TASK == "cls" else \
        ((y_reg_all - Y_REG_MEAN) / Y_REG_STD).astype(np.float32)

def _graph_batch_y(rows, y_arr, fkey=None):
    """Attach labels (and the hybrid's fingerprint branch) to fresh graph shells —
    thread-safe: new Data objects share tensors; fp stored as [1, nbits] so PyG
    concatenates to [num_graphs, nbits] in a Batch."""
    out = []
    for i in rows:
        g = graphs_base[i]
        d = Data(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr,
                 y=torch.tensor([float(y_arr[i])], dtype=torch.float32))
        if fkey == "fp":
            d.fp = torch.tensor(FEATS["fp"][i], dtype=torch.float32).unsqueeze(0)
        out.append(d)
    return out

def predict(model, loader, kind, device=DEVICE):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            if kind == "graph":
                batch = batch.to(device)
                out = model(batch)
                yb = batch.y.view(-1).cpu().numpy()
            else:
                xb, yb = batch
                out = model(xb.to(device))
                yb = yb.numpy()
            out = out.float()
            if TASK == "cls":
                ps.append(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
            else:  # map standardized preds/labels back to pIC50 units
                ps.append(out.view(-1).cpu().numpy() * Y_REG_STD + Y_REG_MEAN)
                yb = yb * Y_REG_STD + Y_REG_MEAN
            ys.append(yb)
    return np.concatenate(ys), np.concatenate(ps)

def fit_model(name, train_rows, val_rows, y_override=None, epochs=None, lr=1e-3, wd=0.0,
              hidden=None, dropout=None, batch=None, patience=None, seed=RANDOM_SEED,
              device=None, verbose=True):
    """Train from scratch on the given rows. Early stopping uses VALIDATION only.
    Returns dict(name, model, kind, val metrics, best, history, params, sec)."""
    t0 = time.time()
    device = device or DEVICE
    factory, kind, fkey = DL_MODELS[name]
    epochs, patience = (epochs or EPOCHS), (patience or PATIENCE)
    batch = batch or (BATCH_ARRAY if kind == "array" else BATCH_GRAPH)
    if SMOKE:                                   # tiny fast validation subset
        train_rows, val_rows = np.asarray(train_rows)[:60], np.asarray(val_rows)[:32]
    y_arr = _y_task(y_override)
    set_seed(seed)

    if kind == "graph":
        tr_ds = _graph_batch_y(train_rows, y_arr, fkey)
        va_ds = _graph_batch_y(val_rows, y_arr, fkey)
    else:
        X = FEATS[fkey]
        yt_full = torch.tensor(y_arr, dtype=torch.float32)
        tr_ds = TensorDataset(torch.tensor(X[train_rows]), yt_full[torch.tensor(train_rows)])
        va_ds = TensorDataset(torch.tensor(X[val_rows]), yt_full[torch.tensor(val_rows)])
    tr_ld = make_loader(tr_ds, batch, True, kind)
    va_ld = make_loader(va_ds, batch, False, kind)

    # input dim + optional pretrained body warm-start (D-MPNN stage1a hook)
    in_dim = graphs_base[0].x.shape[1] if kind == "graph" else FEATS[fkey].shape[1]
    kw = dict(in_dim=in_dim, out_dim=1 if TASK == "reg" else 2)
    if hidden is not None:
        kw["hidden"] = int(hidden)
    if dropout is not None:
        kw["dropout"] = float(dropout)
    model = factory(**kw).to(device)
    ws_sd, ws_tag = None, None
    if name == "D-MPNN" and globals().get("PRETRAIN_W") and globals().get("PRETRAIN_SD"):
        ws_sd, ws_tag = PRETRAIN_SD, "stage1a.pt"
    elif name == globals().get("WARM_START_NAME") and globals().get("WARM_START_SD"):
        ws_sd, ws_tag = WARM_START_SD, str(globals().get("WARM_START_TAG", "pretrained"))
    if ws_sd is not None:
        tgt = model.state_dict()
        body = {k: v for k, v in ws_sd.items() if k in tgt and tuple(tgt[k].shape) == tuple(v.shape)}
        skipped = [k for k in ws_sd.keys() if k not in body]
        if body:
            model.load_state_dict(body, strict=False)
        if verbose:
            msg = f"      [{name}] warm-started {len(body)}/{len(ws_sd)} tensors from {ws_tag}"
            if skipped:
                msg += f" (skipped {len(skipped)} shape-mismatched)"
            print(msg)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min" if TASK == "reg" else "max", factor=0.5, patience=5)
    sclr = scaler_ctx(device)
    key = PRIMARY_METRIC
    best = -np.inf if TASK == "cls" else np.inf
    best_state, bad, hist = None, 0, []
    for ep in range(1, epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for batch in tr_ld:
            opt.zero_grad(set_to_none=True)
            with autocast_ctx(device):
                if kind == "graph":
                    b = batch.to(device)
                    out = model(b)
                    loss = F.cross_entropy(out, b.y.view(-1).long()) if TASK == "cls" \
                        else F.mse_loss(out.view(-1), b.y.view(-1))
                else:
                    xb, yb = batch
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    loss = F.cross_entropy(out, yb.long()) if TASK == "cls" \
                        else F.mse_loss(out.view(-1), yb)
            sclr.scale(loss).backward()
            sclr.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            sclr.step(opt)
            sclr.update()
            tot += float(loss); nb += 1
        ys, ps = predict(model, va_ld, kind, device)
        m = task_metrics(ys, ps)
        v = m[key]
        improved = (v > best + 1e-5) if TASK == "cls" else (v < best - 1e-5)
        if improved:
            best, bad = v, 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
        sched.step(v)
        hist.append({"epoch": ep, **{k: round(float(x), 4) for k, x in m.items()}})
        if bad >= patience:
            break
        if verbose and (ep == 1 or ep % 10 == 0 or ep == epochs):
            print(f"      [{name}] ep {ep:3d} loss {tot / max(nb, 1):.4f} val {key} {v:.4f}")
    if best_state:
        model.load_state_dict(best_state)
    ys, ps = predict(model, va_ld, kind, device)
    val_m = task_metrics(ys, ps)
    return {"name": name, "key": name, "model": model, "kind": kind,
            "val": val_m, "best": best, "epochs_run": len(hist), "hist": hist,
            "params": sum(p.numel() for p in model.parameters()),
            "sec": round(time.time() - t0, 1), "cfg": {"lr": lr, "wd": wd, "hidden": hidden,
                                                       "dropout": dropout, "batch": batch}}

def _fmt(d):
    return {k: round(float(v), 4) for k, v in d.items() if isinstance(v, (int, float))}

print(f"[S08] engine ready | epochs={EPOCHS} patience={PATIENCE} "
      f"| batch graph={BATCH_GRAPH} array={BATCH_ARRAY} | AMP={USE_AMP} | clip={GRAD_CLIP}")
'''

BASELINE_SRC = r'''
# ## 9. Baseline Training — every model, default config (validation-selected, test untouched)
baseline_rows, TRAINED = [], {}
for name in DL_MODELS:
    try:
        r = fit_model(name, tr_idx, va_idx)
        TRAINED[name] = r["model"]
        baseline_rows.append({"model": name,
                              **{f"val_{k}": round(float(v), 4) for k, v in r["val"].items()},
                              "params": r["params"], "epochs_run": r["epochs_run"], "sec": r["sec"]})
        print(f"[S09] {name:14s} val {_fmt(r['val'])} | {r['params']:,} params | "
              f"{r['epochs_run']} ep | {r['sec']}s")
    except Exception as e:
        print(f"[S09] {name} -> [skipped] {type(e).__name__}: {e}")
base_df = pd.DataFrame(baseline_rows).sort_values(
    f"val_{PRIMARY_METRIC}", ascending=(TASK == "reg")).reset_index(drop=True)
base_df.to_csv(OUT_DIR / "baseline_table.csv", index=False)
base_df
'''

FT_SRC = r'''
# ## 10. Stepwise Fine-Tuning — stage ladder, VALIDATION-only selection
# Stage A learning rate -> Stage B dropout/weight-decay -> Stage C capacity -> Stage D batch.
# Each stage keeps its best validation config; the next stage searches around it.
# Extended-roster models are trained once at baseline by default; the four-stage
# ladder (4 stages x N candidates) is only for the original 23 unless explicitly asked.
FT_MODELS = (list(DL_MODELS) if os.environ.get("DL_EXTRA_FT", "0") == "1" else
             [n for n in DL_MODELS if n not in EXTRA_REGISTRY])  # trim to fine-tune fewer
FT_EPOCHS, FT_PATIENCE = (3, 2) if SMOKE else ((5, 3) if FAST else (60, 8))
LR_STAGE = [1e-3] if SMOKE else ([1e-3, 5e-4] if FAST else [1e-3, 5e-4, 3e-4, 1e-4])
DO_STAGE = ([(0.2, 0.0)] if SMOKE else
            ([(0.1, 0.0), (0.3, 0.0)] if FAST else
             [(0.1, 0.0), (0.1, 1e-4), (0.3, 0.0), (0.3, 1e-4)]))
CAP_STAGE = [None] if SMOKE else ([None, 64] if FAST else [None, 64, 256])
BATCH_STAGE = ([None] if (SMOKE or FAST) else
               ([None, 64] if DEVICE.type == "cpu" else [None, 256]))

ft_rows, ft_best_cfg, ft_epochs_run = [], {}, {}
for name in FT_MODELS:
    print(f"[S10] fine-tuning {name} (stage ladder)")
    cfg = {"lr": 1e-3, "dropout": None, "hidden": None, "wd": 0.0, "batch": None}
    best_u = None
    for sname, cands in [("A lr", [{"lr": l} for l in LR_STAGE]),
                         ("B dropout/wd", [{"dropout": d, "wd": w} for d, w in DO_STAGE]),
                         ("C capacity", [{"hidden": h} for h in CAP_STAGE]),
                         ("D batch", [{"batch": b} for b in BATCH_STAGE])]:
        scored = []
        for c in cands:
            cc = {**cfg, **c}
            r = fit_model(name, tr_idx, va_idx, epochs=FT_EPOCHS, lr=cc["lr"], wd=cc["wd"],
                          hidden=cc["hidden"], dropout=cc["dropout"], batch=cc["batch"],
                          patience=FT_PATIENCE, verbose=False)
            u = r["val"][PRIMARY_METRIC] if TASK == "cls" else -r["val"][PRIMARY_METRIC]
            scored.append((u, c, r))
        scored.sort(key=lambda t: -t[0])
        u, c, r = scored[0]
        if best_u is None or u > best_u + 1e-4:
            cfg.update(c)
            best_u = u
            ft_best_cfg[name] = dict(cfg)
            ft_epochs_run[name] = r["epochs_run"]
            print(f"   stage {sname}: {c} -> val {PRIMARY_METRIC} {r['val'][PRIMARY_METRIC]:.4f}")
        else:
            print(f"   stage {sname}: no improvement, keeping {cfg}")
    r = fit_model(name, tr_idx, va_idx, epochs=EPOCHS, lr=cfg["lr"], wd=cfg["wd"],
                  hidden=cfg["hidden"], dropout=cfg["dropout"], batch=cfg["batch"], verbose=False)
    TRAINED[name] = r["model"]
    ft_rows.append({"model": name, **{f"cfg_{k}": v for k, v in cfg.items()},
                    **{f"ft_val_{k}": round(float(v), 4) for k, v in r["val"].items()}})
    print(f"[S10] {name} tuned {cfg} -> val {_fmt(r['val'])}")
ft_df = pd.DataFrame(ft_rows)
ft_df.to_csv(OUT_DIR / "finetune_table.csv", index=False)
ft_df
'''

PRETRAIN_HOOK_SRC = r'''
# ## 10b. Pretrained Warm-Start Hook (D-MPNN body from stage1a.pt) — optional, label-free weights
# stage1a.pt comes from pretrain_stage1a_npass.ipynb (NPASS corpus — it never saw the 645 labels).
PRETRAIN_W = True
_cand = [Path("stage1a.pt"), Path("../stage1a.pt"), Path("../pretrain_stage1a.pt"),
         Path("../02_gnn_models/models/stage1a.pt"), Path("02_gnn_models/models/stage1a.pt")]
PRETRAIN_SD = None
for p in _cand:
    if p.exists():
        try:
            try:
                PRETRAIN_SD = torch.load(p, map_location="cpu", weights_only=True)
            except TypeError:                       # torch < 2.0 has no weights_only
                PRETRAIN_SD = torch.load(p, map_location="cpu")
            PRETRAIN_SD = PRETRAIN_SD.get("model", PRETRAIN_SD) \
                if isinstance(PRETRAIN_SD, dict) else PRETRAIN_SD
            print("[S10b] found stage1a.pt at", p)
        except Exception as e:
            print("[S10b] could not load", p, "->", e)
        break
if PRETRAIN_SD is None:
    print("[S10b] stage1a.pt not found -> D-MPNN trains from scratch (hook is optional)")
'''

HF_FT_SRC = r'''
# ## 10c. Stepwise Fine-Tuning of a Pretrained Chemical LM (ChemBERTa / ChemBERTa-3 / MoLFormer) — optional
# Genuine stepwise FINE-TUNING: (1) frozen encoder + head-only, (2) unfreeze last 2 layers,
# (3) full unfreeze at low lr. Validation-only selection. Disabled by default (needs GPU + download).
# Pick the model with DL_HF_MODEL: chemberta (default), chemberta2, chemberta3, molformer.
# HF IDs checked live 2026-11-25: seyonec ChemBERTa "v0" is gone (401) -> "v1".
USE_HF_PRETRAINED = os.environ.get("DL_USE_HF", "0") == "1"
if USE_HF_PRETRAINED and not SMOKE and DEVICE.type == "cuda":
    try:
        if not _have("transformers"):
            _pip("transformers")          # install BEFORE the import on fresh machines
        from transformers import AutoModel, AutoTokenizer
        HF_CHOICES = {"chemberta": "seyonec/ChemBERTa-zinc-base-v1",
                      "chemberta2": "DeepChem/ChemBERTa-77M-MLM",
                      "chemberta3": "DeepChem/ChemBERTa-100M-MLM",
                      "molformer": "ibm-research/MoLFormer-XL-both-10pct"}
        HF_KEY = os.environ.get("DL_HF_MODEL", "chemberta")
        HF_NAME = HF_CHOICES.get(HF_KEY, HF_CHOICES["chemberta"])
        _trust = {"molformer": True}.get(HF_KEY, False)   # MoLFormer needs remote code
        print(f"[S10c] pretrained LM: {HF_NAME} (DL_HF_MODEL={HF_KEY})")
        tok = AutoTokenizer.from_pretrained(HF_NAME, trust_remote_code=_trust)
        enc = AutoModel.from_pretrained(HF_NAME, trust_remote_code=_trust)

        class ChemLMHead(nn.Module):
            def __init__(self, base, out_dim=2, dropout=0.2):
                super().__init__()
                self.base = base
                h = base.config.hidden_size
                self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(h, out_dim))

            def forward(self, input_ids, attention_mask):
                z = self.base(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
                return self.head(z)

        hf_model = ChemLMHead(enc, out_dim=1 if TASK == "reg" else 2).to(DEVICE)

        def hf_encode(rows):
            b = tok([smiles_all[i] for i in rows], padding=True, truncation=True,
                    max_length=128, return_tensors="pt")
            return b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE)

        y_full = _y_task()
        tr_ids, tr_mask = hf_encode(tr_idx)
        va_ids, va_mask = hf_encode(va_idx)
        tr_y = torch.tensor(y_full[tr_idx], device=DEVICE)
        va_y = torch.tensor(y_full[va_idx], device=DEVICE)
        lossf = F.cross_entropy if TASK == "cls" else F.mse_loss

        def hf_eval(m):
            m.eval(); ps = []
            with torch.no_grad():
                for i0 in range(0, len(va_ids), 64):
                    out = m(va_ids[i0:i0 + 64], va_mask[i0:i0 + 64]).float()
                    ps.append(torch.softmax(out, 1)[:, 1].cpu().numpy() if TASK == "cls"
                              else out.view(-1).cpu().numpy() * Y_REG_STD + Y_REG_MEAN)
            return task_metrics(y_full[va_idx] * (Y_REG_STD if TASK == "reg" else 1)
                                + (Y_REG_MEAN if TASK == "reg" else 0), np.concatenate(ps))

        hf_hist = []
        for stage, (unfreeze, lr, eps) in enumerate([("frozen+head", 0, 1e-3),
                                                     ("last-2-layers", 2, 2e-5),
                                                     ("full", 99, 1e-5)], 1):
            for p_ in hf_model.base.parameters():
                p_.requires_grad = False
            if unfreeze > 0:
                _enc_layers = getattr(getattr(hf_model.base, "encoder", None), "layer", None)
                if _enc_layers is None:      # architecture without a .encoder.layer stack
                    for p_ in hf_model.base.parameters():
                        p_.requires_grad = True
                else:
                    layers = list(_enc_layers)
                    for lyr in layers[len(layers) - min(unfreeze, len(layers)):]:
                        for p_ in lyr.parameters():
                            p_.requires_grad = True
            opt = torch.optim.Adam(
                [p_ for p_ in hf_model.parameters() if p_.requires_grad], lr=lr)
            for ep in range(1, 6):
                hf_model.train()
                for i0 in range(0, len(tr_ids), 32):
                    opt.zero_grad(set_to_none=True)
                    out = hf_model(tr_ids[i0:i0 + 32], tr_mask[i0:i0 + 32])
                    loss = lossf(out.view(-1) if TASK == "reg" else out, tr_y[i0:i0 + 32].long()
                                 if TASK == "cls" else tr_y[i0:i0 + 32])
                    loss.backward(); opt.step()
            m = hf_eval(hf_model)
            hf_hist.append({"stage": stage, "mode": unfreeze, "lr": lr, **_fmt(m)})
            print(f"[S10c] stage {stage} ({unfreeze} layers, lr {lr}) -> {_fmt(m)}")
        pd.DataFrame(hf_hist).to_csv(OUT_DIR / "hf_finetune_ladder.csv", index=False)
    except Exception as e:
        print("[S10c] [skipped]", type(e).__name__, e)
else:
    print("[S10c] HF pretrained fine-tune disabled (set DL_USE_HF=1 on a GPU machine; "
          "DL_HF_MODEL=chemberta|chemberta2|chemberta3|molformer). "
          "Pretrained weights stay sensitivity-only.")
'''

DEEPCHEM_SRC = r'''
# ## 9b. The REAL DeepChem GCNModel (optional, guarded — origin: Arjun sir Part_9)
# Exactly the stack from 02_gnn_models/deepchem_gcn_model.py:
# dc.feat.ConvMolFeaturizer -> dc.data.NumpyDataset -> dc.models.GCNModel.
# If the deepchem package is missing this prints [skipped]; the pure-torch
# DeepChemGCN from S07 already covers the same architecture.
import warnings
warnings.filterwarnings("ignore", message=".*GetValence.*")

# Opt-in only: deepchem is heavy, often lags new Python versions, and is NOT
# needed for the 10-model roster (DeepChemGCN is reimplemented in pure torch).
# Enable with DL_INSTALL_DEEPCHEM=1 if you specifically want the original API.
INSTALL_DEEPCHEM = os.environ.get("DL_INSTALL_DEEPCHEM", "0") == "1"
if INSTALL_DEEPCHEM and not _have("deepchem"):
    _pip("deepchem")
try:
    import deepchem as dc
    HAS_DEEPCHEM = True
except Exception as _e:
    HAS_DEEPCHEM = False
    print(f"[S09b] [skipped] deepchem not importable ({type(_e).__name__}: {_e}) "
          "— pure-torch DeepChemGCN (S07) already trained")

if not HAS_DEEPCHEM:
    print("[S09b] optional real deepchem GCNModel skipped; set DL_INSTALL_DEEPCHEM=1 to try it")
else:
    _dc_feats = dc.feat.ConvMolFeaturizer()
    _y_full = _y_task()

    def _dc_ds(rows, y):
        X = _dc_feats.featurize([smiles_all[i] for i in rows])
        return dc.data.NumpyDataset(X=X, y=np.asarray(y, dtype=np.float32), ids=None)

    tr_ds = _dc_ds(tr_idx, _y_full[tr_idx])
    va_ds = _dc_ds(va_idx, _y_full[va_idx])
    mode = "classification" if TASK == "cls" else "regression"
    dc_model = dc.models.GCNModel(mode=mode, n_tasks=1, learning_rate=1e-3,
                                  batch_size=32, graph_conv_layers=[128, 128])
    t0 = time.time()
    dc_model.fit(tr_ds, nb_epoch=3 if SMOKE else 30)
    pred_va = dc_model.predict(va_ds)
    # natural units: probability for cls, back-scaled pIC50 for reg
    p1_va = (pred_va[:, 1] if TASK == "cls"
             else pred_va.ravel() * Y_REG_STD + Y_REG_MEAN)
    y_va_nat = (_y_full[va_idx] * Y_REG_STD + Y_REG_MEAN) if TASK == "reg" else _y_full[va_idx]
    m_va = task_metrics(y_va_nat, p1_va)
    print(f"[S09b] real deepchem GCNModel ({mode}) val: {_fmt(m_va)} | {time.time() - t0:.1f}s")

    _ids = (df["molecule_chembl_id"].astype(str).values if len(df) == len(smiles_all)
            else np.array([f"mol_{i}" for i in range(len(smiles_all))]))
    _dc_rows = pd.DataFrame({
        "identifier": _ids[va_idx],
        "canonical_smiles": [smiles_all[i] for i in va_idx],
        "y_true": np.asarray(y_va_nat, dtype=float).round(6)})
    p1c = np.clip(np.asarray(p1_va, dtype=float), 0.0, 1.0)
    if TASK == "cls":
        _dc_rows["prediction"] = (p1c >= 0.5).astype(int)
        _dc_rows["prob_class_1"] = p1c.round(6)
        _dc_rows["prob_class_0"] = (1.0 - p1c).round(6)
    else:
        _dc_rows["prediction"] = np.asarray(p1_va, dtype=float).round(6)
    _csv = OUT_DIR / "deepchem_gcn_val_preds.csv"
    _dc_rows.to_csv(_csv, index=False)
    print(f"[S09b] saved {_csv} ({len(_dc_rows)} rows, prob_class_1 column)")
'''

CV_SRC = r'''
# ## 11. Five-Fold GroupKFold Cross-Validation (scaffold-grouped, tuned configs)
from sklearn.model_selection import GroupKFold

CV_FOLDS = 2 if (SMOKE or FAST) else 5
CV_EPOCHS = 3 if SMOKE else (8 if FAST else None)
CV_MODELS = list(DL_MODELS)[:3] if SMOKE else list(DL_MODELS)   # trim to save time
groups_trva = scaffolds_all[tr_va_idx]
folds = list(GroupKFold(n_splits=CV_FOLDS).split(
    np.zeros(len(groups_trva)), groups=groups_trva))
print("[S11] fold val sizes:", [len(va) for _, va in folds])

jobs = [{"key": f"{name}#{k}", "name": name, "fold": k, "rows": (tr_va_idx[trr], tr_va_idx[var])}
        for name in CV_MODELS for k, (trr, var) in enumerate(folds)]

def run_fold(job, device_id):
    cfg = ft_best_cfg.get(job["name"], {})
    r = fit_model(job["name"], job["rows"][0], job["rows"][1],
                  epochs=CV_EPOCHS, lr=cfg.get("lr", 1e-3), wd=cfg.get("wd", 0.0),
                  hidden=cfg.get("hidden"), dropout=cfg.get("dropout"), batch=cfg.get("batch"),
                  verbose=False, device=(DEVICES[device_id] if DEVICE.type == "cuda" else DEVICE))
    return {"key": job["key"], "name": job["name"], "fold": job["fold"],
            "val": r["val"], "epochs_run": r["epochs_run"]}

cv_res = run_parallel(run_fold, jobs)
cv_rows = []
for name in CV_MODELS:
    vals = [r["val"][PRIMARY_METRIC] for r in cv_res if r["name"] == name]
    if vals:
        cv_rows.append({"model": name, "folds": len(vals),
                        f"cv_{PRIMARY_METRIC}_mean": round(float(np.mean(vals)), 4),
                        f"cv_{PRIMARY_METRIC}_std": round(float(np.std(vals)), 4)})
cv_df = pd.DataFrame(cv_rows).sort_values(
    f"cv_{PRIMARY_METRIC}_mean", ascending=(TASK == "reg")).reset_index(drop=True)
cv_df.to_csv(OUT_DIR / "cv_table.csv", index=False)
print("[S11] CV mean±std (validation folds only — test still untouched)")
cv_df
'''

YRAND_SRC = r'''
# ## 12. Y-Randomization — 10 shuffled-label retrains (Part_9 protocol; sanity check ONLY)
from sklearn.utils import shuffle as sk_shuffle

YRAND_N = 2 if (SMOKE or FAST) else 10
YRAND_EPOCHS = 3 if SMOKE else (8 if FAST else 100)
YRAND_MODELS = (list(DL_MODELS)[:1] if SMOKE
                else list(cv_df["model"])[:2] if len(cv_df) else list(DL_MODELS)[:2])

yr_jobs = [{"key": f"{m}#s{si}", "name": m, "si": si}
           for m in YRAND_MODELS for si in range(YRAND_N)]

def run_yrand(job, device_id):
    y_full = _y_task()
    tr_rows = np.asarray(tr_idx)[:60] if SMOKE else np.asarray(tr_idx)
    va_rows = np.asarray(va_idx)[:32] if SMOKE else np.asarray(va_idx)
    perm = np.random.RandomState(job["si"]).permutation(len(tr_rows))
    y_over = y_full.copy()
    y_over[tr_rows] = y_full[tr_rows][perm]        # shuffle TRAIN labels only; val stays real
    cfg = ft_best_cfg.get(job["name"], {})
    r = fit_model(job["name"], tr_rows, va_rows, y_override=y_over, epochs=YRAND_EPOCHS,
                  lr=cfg.get("lr", 1e-3), wd=cfg.get("wd", 0.0), hidden=cfg.get("hidden"),
                  dropout=cfg.get("dropout"), batch=cfg.get("batch"), patience=10,
                  verbose=False, device=(DEVICES[device_id] if DEVICE.type == "cuda" else DEVICE))
    return {"key": job["key"], "name": job["name"], "shuffle": job["si"] + 1, "val": r["val"]}

yr_res = run_parallel(run_yrand, yr_jobs)
yr_rows = [{"model": r["name"], "shuffle": r["shuffle"],
            **{f"shuf_{k}": round(float(v), 4) for k, v in r["val"].items()}} for r in yr_res]
yr_df = pd.DataFrame(yr_rows)
yr_agg = yr_df.groupby("model")[f"shuf_{PRIMARY_METRIC}"].agg(["mean", "std"]).reset_index()
yr_df.to_csv(OUT_DIR / "y_randomization_table.csv", index=False)
print(f"[S12] Y-randomization: {YRAND_N} shuffles x {YRAND_EPOCHS} epochs per model "
      "(higher-than-chance real AUC = the labels carry real signal)")
yr_agg
'''

TEST_SRC = r'''
# ## 13. Final Test Evaluation — scored exactly ONCE, with bootstrap 95% CI
BOOTS = 200 if SMOKE else (300 if FAST else 2000)
assert "final_eval_done" not in globals(), "test scored twice — run this cell once"
final_eval_done = True

# CV-ranked families (never saw test); fall back to the baseline leader if CV is empty
best_name = str(cv_df.iloc[0]["model"]) if len(cv_df) else str(base_df.iloc[0]["model"])
cfg = ft_best_cfg.get(best_name, {})
print(f"[S13] final model family: {best_name} | cfg {cfg}")

final = fit_model(best_name, tr_idx, va_idx, epochs=EPOCHS, lr=cfg.get("lr", 1e-3),
                  wd=cfg.get("wd", 0.0), hidden=cfg.get("hidden"), dropout=cfg.get("dropout"),
                  batch=cfg.get("batch"), verbose=False)
final_model = final["model"]
kind = final["kind"]
te_rows = np.asarray(test_idx)[:24] if SMOKE else np.asarray(test_idx)
if kind == "graph":
    te_ds = _graph_batch_y(te_rows, _y_task(), DL_MODELS[best_name][2])
else:
    X = FEATS[DL_MODELS[best_name][2]]
    y_arr = _y_task()
    te_ds = TensorDataset(torch.tensor(X[te_rows]), torch.tensor(y_arr[te_rows], dtype=torch.float32))
te_ld = make_loader(te_ds, 128, False, kind)
yt_test, yp_test = predict(final_model, te_ld, kind)
test_metrics = task_metrics(yt_test, yp_test)

rng = np.random.RandomState(RANDOM_SEED)
boot = []
for _ in range(BOOTS):
    idx = rng.randint(0, len(yt_test), len(yt_test))
    if len(set(np.asarray(yt_test)[idx].tolist())) < 2:
        continue                                # AUC undefined on single-class resamples
    boot.append(task_metrics(yt_test[idx], yp_test[idx])[PRIMARY_METRIC])
if boot:
    ci_lo, ci_hi = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
else:
    ci_lo = ci_hi = float("nan")          # single-class test set -> CI undefined
print(f"[S13] TEST (once): {_fmt(test_metrics)} | {PRIMARY_METRIC} 95% CI "
      f"[{ci_lo:.4f}, {ci_hi:.4f}] over {len(boot)}/{BOOTS} valid bootstrap resamples")

# predictions CSV in the screening_results.csv schema (softmax probabilities by default)
_ids = (df["molecule_chembl_id"].astype(str).values if len(df) == len(smiles_all)
        else np.array([f"mol_{i}" for i in range(len(smiles_all))]))
tp = pd.DataFrame({
    "identifier": _ids[te_rows],
    "canonical_smiles": [smiles_all[i] for i in te_rows],
    "y_true": np.asarray(yt_test, dtype=float).round(6)})
if TASK == "cls":
    p1 = np.clip(np.asarray(yp_test, dtype=float), 0.0, 1.0)
    tp["prediction"] = (p1 >= 0.5).astype(int)
    tp["prob_class_1"] = p1.round(6)
    tp["prob_class_0"] = (1.0 - p1).round(6)
    print("[S13] test_predictions.csv: prediction + softmax prob_class_1/prob_class_0 "
          "(same schema as screening_results.csv)")
else:
    tp["prediction"] = np.asarray(yp_test, dtype=float).round(6)   # pIC50 units
    print("[S13] test_predictions.csv: prediction = predicted pIC50")
tp.to_csv(OUT_DIR / "test_predictions.csv", index=False)
print(f"[S13] saved {OUT_DIR / 'test_predictions.csv'} ({len(tp)} rows)")

if TASK == "cls":
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(yt_test.astype(int), yp_test)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"{best_name} AUC={test_metrics['AUC']:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.title("Test ROC (scored once)")
    plt.tight_layout(); plt.savefig(OUT_DIR / "test_roc.png", dpi=120); plt.show()
else:
    plt.figure(figsize=(5, 4))
    plt.scatter(yt_test, yp_test, s=18, alpha=0.7)
    lo, hi = min(yt_test.min(), yp_test.min()), max(yt_test.max(), yp_test.max())
    plt.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    plt.xlabel("true pIC50"); plt.ylabel("pred pIC50"); plt.title("Test predictions (once)")
    plt.tight_layout(); plt.savefig(OUT_DIR / "test_pred_true.png", dpi=120); plt.show()
'''

SUMMARY_SRC = r'''
# ## 14. Summary Tables, Plots & Saved Artifacts
RUN_RECORD = {
    "nb": "%%NB_ID%%", "task": TASK, "smoke": SMOKE, "fast": FAST,
    "profile": PROFILE, "seed": RANDOM_SEED,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "device": str(DEVICE), "n_gpus": N_GPUS,
    "split_sizes": {"train": int(len(tr_idx)), "val": int(len(va_idx)), "test": int(len(test_idx))},
    "models": sorted(DL_MODELS),
    "baseline": baseline_rows,
    "finetune_cfg": ft_best_cfg,
    "cv": cv_rows,
    "yrand_mean": {r["model"]: {"mean": round(float(r["mean"]), 4),
                                "std": round(float(r["std"]), 4)} for _, r in yr_agg.iterrows()},
    "final_model": best_name,
    "test_metrics": {k: round(float(v), 4) for k, v in test_metrics.items()},
    "test_ci": [round(ci_lo, 4), round(ci_hi, 4)],
}
with open(OUT_DIR / "%%NB_ID%%_results.json", "w") as f:
    json.dump(RUN_RECORD, f, indent=2, default=str)

# leaderboard plot: baseline vs fine-tuned vs CV mean (validation metrics)
plot_df = base_df.set_index("model")[f"val_{PRIMARY_METRIC}"].to_frame("baseline")
if len(ft_df):
    plot_df = plot_df.join(ft_df.set_index("model")[f"ft_val_{PRIMARY_METRIC}"].to_frame("finetuned"),
                           how="left")
if len(cv_df):
    plot_df = plot_df.join(cv_df.set_index("model")[f"cv_{PRIMARY_METRIC}_mean"].to_frame("cv_mean"),
                           how="left")
plot_df.plot(kind="barh", figsize=(8, 0.5 * len(plot_df) + 2))
plt.xlabel(PRIMARY_METRIC + " (validation)")
plt.title("%%NB_TITLE_SHORT%% — validation leaderboard")
plt.tight_layout(); plt.savefig(OUT_DIR / "leaderboard.png", dpi=120); plt.show()

files = sorted(str(p.name) for p in OUT_DIR.iterdir())
print("[S14] artifacts written:", files)
'''

SCREEN_SRC = r'''
# ## 15. Screen the 154k COCONUT Library (ON by default; DL_RUN_SCREENING=0 to skip)
# Chunked prediction with the final model + Tanimoto applicability domain (Part_20 convention:
# Morgan r=3/512 vs train set; median screen<->train sim was 0.217 -> ~99.4% out-of-domain).
RUN_SCREENING = os.environ.get("DL_RUN_SCREENING", "1") == "1"   # ON by default; set 0 to skip
AD_CUT, AD_CUT_STRICT = 0.40, 0.52            # from 00_data/config/cuts.json
try:
    _cut_paths = [Path("../00_data/config/cuts.json"), Path("00_data/config/cuts.json"),
                  Path("/content/00_data/config/cuts.json")]
    _cuts = json.load(open(next(p for p in _cut_paths if p.exists())))
    AD_CUT, AD_CUT_STRICT = float(_cuts.get("ad_cut", AD_CUT)), float(_cuts.get("ad_cut_strict", AD_CUT_STRICT))
except Exception:
    pass

if RUN_SCREENING and not SMOKE and not FAST:
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from IPython.display import display

    _sc = None
    for c in ["../00_data/screening/screening_results.csv",
              "00_data/screening/screening_results.csv",
              "/content/00_data/screening/screening_results.csv",
              # Arjun sir's original public repo (source of the COCONUT screen)
              "https://raw.githubusercontent.com/arjunpahi/Natural_MDM2_Inhibitor_Discovery_using_ML/main/Part_8/screening_results.csv"]:
        try:
            _sc = pd.read_csv(c); print("[S15] loaded:", c); break
        except Exception:
            continue
    assert _sc is not None, "screening CSV not found (local or GitHub fallback)"
    sc_smiles = _sc["canonical_smiles"].astype(str).tolist()
    print(f"[S15] screening rows: {len(sc_smiles):,}")

    # --- Tanimoto AD (Morgan r=3/512, independent of the model's own features) ---
    _gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=512)
    train_fps = []
    for s in [smiles_all[i] for i in tr_idx]:
        _m = Chem.MolFromSmiles(s)
        if _m is not None:
            _fp = _gen.GetFingerprint(_m)
            if _fp is not None:
                train_fps.append(_fp)
    assert train_fps, "no valid training fingerprints for applicability domain"

    # --- model features for screening rows (match the final model's featurizer) ---
    kind = DL_MODELS[best_name][1]
    fkey = DL_MODELS[best_name][2]
    sc_probs, sc_ad = [], []
    CH = 5000
    for i0 in range(0, len(sc_smiles), CH):
        chunk_idx = list(range(i0, min(i0 + CH, len(sc_smiles))))
        chunk_smiles = [sc_smiles[i] for i in chunk_idx]
        chunk_mols = [Chem.MolFromSmiles(s) for s in chunk_smiles]
        if kind == "graph":
            use, keep_pos = [], []
            for j, _m in enumerate(chunk_mols):
                g = mol_to_graph(chunk_smiles[j])
                if g is None:
                    continue
                if fkey == "fp":
                    _fp = _make_fp(_m) if _m is not None else None
                    if _fp is None:
                        continue
                    g.fp = torch.tensor(_fpdense([_fp], FP_DIM)[0]).unsqueeze(0)
                use.append(g); keep_pos.append(j)
        else:
            if fkey == "fp":
                feats = np.stack([_fpdense([_make_fp(_m)], FP_DIM)[0] for _m in chunk_mols])
            elif fkey == "desc":
                feats = desc_scaler.transform(
                    np.nan_to_num(np.array([_desc_row(_m) for _m in chunk_mols],
                                           dtype=np.float32)))
            else:  # chars
                feats = encode_smiles(chunk_smiles)
            use = TensorDataset(torch.tensor(np.asarray(feats)),
                                torch.zeros(len(chunk_idx)))
            keep_pos = list(range(len(chunk_idx)))

        probs = np.zeros(len(chunk_idx))
        if len(use):
            ld = make_loader(use, 512, False, kind)
            _, p = predict(final_model, ld, kind)
            probs[np.asarray(keep_pos, dtype=int)] = p
        sc_probs.append(probs)

        ad_chunk = []
        for _m in chunk_mols:
            _fp = _gen.GetFingerprint(_m) if _m is not None else None
            if _fp is None:
                ad_chunk.append(0.0)
            else:
                _sims = DataStructs.BulkTanimotoSimilarity(_fp, train_fps)
                ad_chunk.append(max(_sims) if _sims else 0.0)
        sc_ad.append(np.array(ad_chunk))
        print(f"   chunk {i0 // CH + 1}/{(len(sc_smiles) + CH - 1) // CH} done")

    sc_prob = np.concatenate(sc_probs); sc_ad = np.concatenate(sc_ad)
    out = _sc.copy()
    out["dl_pred"] = np.round(sc_prob, 4)
    out["ad_score"] = np.round(sc_ad, 4)
    out["AD"] = sc_ad >= AD_CUT
    out["AD_strict"] = sc_ad >= AD_CUT_STRICT
    out.to_csv(OUT_DIR / "screening_dl_predictions.csv", index=False)
    print(f"[S15] AD coverage: {out['AD'].mean():.1%} (strict {out['AD_strict'].mean():.1%}) "
          "— most of COCONUT is extrapolation; predictions there are hypotheses, not hits")
    display(out.sort_values("dl_pred", ascending=False).head(10))
else:
    print(f"[S15] skipped (RUN_SCREENING={RUN_SCREENING}, SMOKE={SMOKE}, FAST={FAST}; "
          "set DL_RUN_SCREENING=1 for the full 154k screen, 0 to silence this)")
'''

DIARY_SRC = r'''
# ## 16. Run Diary — copy this block into docs/JOURNEY.md
lines = [
    "### %%NB_ID%% run record",
    f"- timestamp: {RUN_RECORD['timestamp']} | task={TASK} | profile={PROFILE} | seed={RANDOM_SEED}",
    f"- device: {RUN_RECORD['device']} (GPUs={N_GPUS}) | split "
    f"{RUN_RECORD['split_sizes']['train']}/{RUN_RECORD['split_sizes']['val']}/{RUN_RECORD['split_sizes']['test']}",
    "- baseline (val " + PRIMARY_METRIC + "):",
]
for r in baseline_rows:
    lines.append(f"    - {r['model']:14s} {r['val_' + PRIMARY_METRIC]} ({r['params']:,} params, {r['sec']}s)")
lines.append(f"- fine-tuned configs: {RUN_RECORD['finetune_cfg']}")
if len(cv_df):
    lines.append("- 5-fold CV (mean±std):")
    for _, r in cv_df.iterrows():
        lines.append(f"    - {r['model']:14s} {r[f'cv_{PRIMARY_METRIC}_mean']} ± "
                     f"{r[f'cv_{PRIMARY_METRIC}_std']}")
lines.append(f"- Y-randomization: {yr_agg.to_dict('records')}")
lines.append(f"- FINAL test (scored once, {best_name}): {RUN_RECORD['test_metrics']} "
             f"| 95% CI {RUN_RECORD['test_ci']}")
lines.append(f"- artifacts: {sorted(str(p.name) for p in OUT_DIR.iterdir())}")
report = "\n".join(lines)
print(report)
with open(OUT_DIR / "%%NB_ID%%_diary_block.txt", "w") as f:
    f.write(report + "\n")
'''

# ---------------------------------------------------------------------------
# Markdown templates
# ---------------------------------------------------------------------------

def title_md(nb):
    return md_cell(nb["title_md"])


PRIMER_NB1 = r'''# DL 01 — Chemistry Primer ("preload chemistry")

Read this before the code — every notebook section maps back to one of these ideas.

**1. Molecules as data.** A molecule is a set of **atoms** joined by **bonds** (single/double/triple/aromatic).
Bond order and conjugation determine shape and reactivity — that is why we encode them.

**2. SMILES.** A text grammar for molecules, e.g. `CC(=O)Oc1ccccc1C(=O)O` (aspirin). One molecule can
have several valid SMILES; we use RDKit's canonical form so every compound has exactly one string.

**3. Molecular graphs.** Atoms = nodes (element, degree, charge, H-count, hybridization, aromaticity,
ring, chirality → our **78-dim atom feature**), bonds = edges (type, conjugation, ring → **6-dim**).
Graph neural nets (GCN/GIN/GAT/AttentiveFP/D-MPNN) pass messages along bonds — no fingerprint needed.

**4. Descriptors.** Classic numeric summaries (MW, logP, TPSA, HBA/HBD, rotatable bonds…) — a
**standardized 10-dim vector** here, consumed by a plain MLP. They power Lipinski's RO5, which
already filtered this dataset.

**5. Sequences.** A SMILES string is also a sentence: a **char-level CNN** learns motifs (rings,
branches) directly from characters.

**6. pIC50 & activity.** pIC50 = −log₁₀(IC50 in molar); higher = more potent. `bioactivity_class`
labels Active/Inactive; we train **classification (ROC-AUC)** by default and regression via the
`DL_TASK=reg` flag.

**7. Splitting & leakage.** Analogues share a **Murcko scaffold**; if one scaffold appears in both
train and test, the model memorizes the family. We split by **whole non-isomeric scaffolds**
(≈464 train / 52 val / 129 test, seed 57) and assert zero scaffold overlap.

**8. Validation vs test.** Early stopping, hyper-parameter and model choices use **validation only**.
The 129-compound test set is scored **exactly once**, at the end.

**9. Y-randomization.** Retrain on shuffled labels ~10×: a real model beats the shuffled AUC
distribution; otherwise the "signal" was memorization or leakage.

**10. Applicability domain (AD).** A model is trustworthy only near its training chemistry. The
screening cell reports **Tanimoto similarity to the training set** — most of the 154k COCONUT
library is extrapolation.'''

PRIMER_FP_TEMPLATE = r'''# DL 0%%NB_NUM%% — Chemistry Primer ("preload chemistry")

Read this before the code — every notebook section maps back to one of these ideas.

**1. Molecules as data.** Atoms joined by bonds; bond order and aromaticity drive shape and
reactivity, so both are encoded.

**2. SMILES.** Text grammar for molecules (RDKit canonical form ⇒ one string per molecule).

**3. %%FP_PRIMER%%.**

**4. Molecular graphs.** Atoms → 78-dim features, bonds → 6-dim features; the **hybrid model**
passes messages along bonds (GCN) and **late-fuses** the fingerprint branch — fingerprints see
global substructure, graphs see connectivity.

**5. Deep nets on bit vectors.** An MLP learns interactions between bits; a **residual MLP** adds
skip connections (easier optimization of deeper stacks); a **1-D CNN** slides kernels across the
bit string to find local substructure motifs.

**6. pIC50 & activity.** pIC50 = −log₁₀(IC50 in M); higher = more potent. Default task is
classification (ROC-AUC); `DL_TASK=reg` switches to pIC50 regression.

**7. Splitting & leakage.** Split by **whole non-isomeric Murcko scaffolds** (≈464/52/129, seed 57)
so analogues never straddle train/test; assert zero overlap.

**8. Validation vs test.** Early stopping, stage-wise tuning and model selection use **validation
only**; the test set is scored **exactly once**.

**9. Y-randomization.** ~10 shuffled-label retrains — real AUC must beat the shuffled distribution.

**10. Applicability domain (AD).** Screening predictions are reported with **Tanimoto similarity to
the training set** (Morgan r=3/512, cut 0.40) — most of COCONUT is extrapolation.'''

SECTION_MD = [  # (num, title, [bullet lines]) — Arjun-style numbered headers
    (1, "Load the MDM2 Dataset + EDA", None),           # headers already in code cells; used for toc?
]

PRIMER_T1 = r'''# DL 04 — Chemistry Primer ("preload chemistry")

Read this before running anything. Same guardrails as DL 01-03; here the focus is
**newer architectures** and exactly where each one comes from.

1. **Attention in one line.** Instead of one fixed aggregation rule, every node asks its
   neighbours "how relevant are you?" (softmax weights) and mixes accordingly —
   Vaswani et al., NeurIPS 2017, ported to graphs by Shi et al. 2021 (TransformerConv:
   multi-head, edge-aware attention where neighbours are the tokens).
2. **PNA (Corso et al., NeurIPS 2020).** Molecules have very different local structures
   (terminal carbon with 1 neighbour vs. crowded aromatic ring). PNA runs mean/min/max/std
   aggregators in parallel and rescales each by the node's degree (identity /
   amplification / attenuation). The degree histogram is computed from the **TRAIN split
   only** — no leakage.
3. **GPS (Rampášek et al., Nature Machine Intelligence 2022).** Every layer = local
   message passing + global self-attention, then norm/activation. Short-range chemistry
   comes from the local conv, long-range context from attention. Our mini version uses a
   GCNConv local path + 4-head attention.
4. **SMILES Transformer.** A SMILES string is a token sequence: char embedding + position
   embedding → small 2-layer transformer encoder → masked-mean pooling over non-PAD
   characters. Same family as DL 01's CharCNN, attention instead of convolution.
5. **FT-Transformer (Gorishniy et al., NeurIPS 2021).** Tabular deep learning done right:
   every feature becomes a token (per-feature linear projection), a learnable CLS token is
   prepended, the encoder reads the tokens, and CLS feeds the head. Here each of the 10
   RDKit descriptors is one token.
6. **Why NOT Graphormer / 3-D nets?** Graphormer needs shortest-path + centrality
   encodings; 3-D models (SchNet/DimeNet/EGNN) need conformers. With 645 molecules they
   over-complicate; Tier-1 keeps capacity matched to data size.
7. **Small-data discipline.** d_model/heads kept small (64-96, 4 heads); the stepwise
   fine-tuning ladder (lr → dropout/wd → capacity → batch) finds the right capacity —
   selection on **VALIDATION only**.
8. **Same split as DL 01-03:** leak-free Murcko scaffold split 464/52/129 (seed 57),
   canonical-SMILES overlap asserted 0, features computed AFTER the split.
9. **Same metrics:** classification ROC-AUC primary (softmax `prob_class_1` = P(Active),
   `prediction / prob_class_0 / prob_class_1` schema like `screening_results.csv`);
   `DL_TASK=reg` for pIC50.
10. **Same honesty:** 5-fold GroupKFold CV, Y-randomization (10×, train labels only),
    test scored exactly ONCE with bootstrap 95% CI, `test_predictions.csv` saved.
'''

PRIMER_T2 = r'''# DL 05 — Chemistry Primer ("preload chemistry")

Read this before running anything. This notebook adds **self-supervised pretraining**.

1. **The idea.** Pretrain the GNN on molecules WITHOUT activity labels, then fine-tune on
   the 645 labelled MDM2 compounds. The encoder learns "what molecules look like" from
   data, not labels — then transfers that knowledge to the task.
2. **Why it is leak-free here.** Pretraining uses a **standard open-source unlabeled
   chemistry corpus** downloaded and cached locally — **not** the COCONUT screening library.
   Default: `DL_PRETRAIN_SOURCE=zinc250k` (250k drug-like SMILES, DeepChem/public mirror).
   Options: `zinc250k`, `chembl_small`, or a custom path/URL; `coconut` is available only
   if explicitly requested. MDM2 labels never enter the pretraining loss.
3. **MolCLR augmentations (Wang et al., Nature Machine Intelligence 2022).** Two random
   views per molecule: atom masking (15%), bond deletion (15%), subgraph removal (25%).
   Same molecule, different views → the encoder must map both to similar representations.
4. **NT-Xent (Chen et al., ICML 2020 — SimCLR).** For a batch of 2N views, every view's
   positive is its partner view; all other views are negatives; softmax cross-entropy over
   cosine similarities / temperature (0.2) pulls true pairs together.
5. **Projection head.** The contrastive loss reads a small MLP on top of the pooled
   representation; its weights are discarded — only the GIN trunk is kept for the
   warm start.
6. **Warm start.** The pretrained trunk loads into "GIN-MolCLR" with `strict=False`
   (exactly like the stage1a.pt hook in DL 01); the task head starts fresh.
7. **What the comparison shows.** "GIN (scratch)" vs "GIN-MolCLR": same architecture, same
   split, same engine — the ONLY difference is the initial weights, so the val/CV/test
   deltas isolate the value of pretraining.
8. **Compute reality.** Full pretraining (30,000 molecules × 30 epochs) is a GPU job;
   smoke mode (1,500 × 2) runs on CPU in minutes just to prove the wiring.
9. **Same guardrails as always:** leak-free Murcko scaffold split 464/52/129, validation-
   only early stopping, 5-fold GroupKFold CV, Y-randomization (train labels only), test
   scored exactly ONCE with bootstrap CI.
10. **Softmax probabilities by default:** `prob_class_1` = P(Active), same
    `prediction / prob_class_0 / prob_class_1` schema as `screening_results.csv`;
    `test_predictions.csv` saved at the end.
'''

NB_TITLES = {
    "dl01": {
        "num": "01",
        "title": "# DL 01: Deep Learning on MDM2 — Graph & Sequence Models (No Fingerprints)\n",
        "subtitle": ("**Reference:** GCN — Kipf & Welling ICLR 2017 · GIN — Xu et al. ICLR 2019 (MolCLR) · "
                     "GAT — Veličković et al. ICLR 2018 · GATv2 — Brody et al. 2022 · AttentiveFP — "
                     "Xiong et al. *J Med Chem* 2020 · D-MPNN — Yang et al. *JCIM* 2019 (chemprop) · "
                     "CharCNN — Crichton & Guo 2017 · **DeepChemGCN** — deepchem GraphConv (Arjun sir Part_9)\n\n"
                     "**Pipeline:** SMILES → RDKit mol graphs + 10 descriptors + char codes → leak-free "
                     "Murcko scaffold split (≈464/52/129, seed 57) → baseline training (validation-only "
                     "early stopping) → **stepwise fine-tuning** (lr → dropout/wd → capacity → batch) → "
                     "5-fold GroupKFold CV → Y-randomization (10×) → **test scored once** + bootstrap CI "
                     "→ optional 154k screening with Tanimoto AD\n\n"
                     "**Data:** 645 MDM2-p53 inhibitors (`00_data/training/ro5_properties_filtered.csv`, "
                     "343 Active) · **Task:** classification ROC-AUC by default (`DL_TASK=reg` for pIC50)\n\n"
                     "**Original repository:** `Natural_MDM2_Inhibitor_Discovery_using_ML/` (in this "
                     "repo) — Arjun sir's project, cloned 2026-09-07 from "
                     "`github.com/arjunpahi/Natural_MDM2_Inhibitor_Discovery_using_ML`; original "
                     "`Part_9_GNN_DeepChem.ipynb` is the DeepChem GCN origin.\n\n"
                     "**Data lineage:** ChEMBL MDM2-p53 bioactivity → 2,571 raw records → 1,933 unique "
                     "compounds → **645 after Lipinski-RO5 filtering (343 Active / 302 Inactive)**. "
                     "Columns: `molecule_chembl_id` · `canonical_smiles` · `bioactivity_class` "
                     "(Active/Inactive) · `pIC50 = −log10(IC50 M)` · MW/HBA/HBD/logP · `ro5_fulfilled`.\n\n"
                     "**No fingerprints in this notebook** — molecular graphs, descriptors and raw SMILES "
                     "only. Fingerprints live in DL 02 (Morgan) and DL 03 (MACCS).\n\n"
                     "*Style: Arjun-sir stepwise format. Models [ADAPTED] from `02_gnn_models/` and "
                     "Part_22; protocol mirrors the Part_9/20/22 guardrails.*"),
    },
    "dl02": {
        "num": "02",
        "title": "# DL 02: Deep Learning on MDM2 — Morgan Fingerprints (ECFP4)\n",
        "subtitle": ("**Reference:** Morgan/ECFP — Rogers & Hahn *JCIM* 2010 · residual MLP — He et al. "
                     "CVPR 2016 · 1-D CNN over fingerprints (this project, Part_14) · hybrid GCN+fingerprint "
                     "late fusion (this project)\n\n"
                     "**Pipeline:** SMILES → Morgan ECFP4 (r=2, 2048 bits) + mol graphs → leak-free Murcko "
                     "scaffold split (≈464/52/129, seed 57) → baseline training (validation-only early "
                     "stopping) → **stepwise fine-tuning** (lr → dropout/wd → capacity → batch) → 5-fold "
                     "GroupKFold CV → Y-randomization (10×) → **test scored once** + bootstrap CI → "
                     "optional 154k screening with Tanimoto AD\n\n"
                     "**Data:** 645 MDM2-p53 inhibitors (`00_data/training/ro5_properties_filtered.csv`, "
                     "343 Active) · **Task:** classification ROC-AUC by default (`DL_TASK=reg` for pIC50)\n\n"
                     "**Original repository:** `Natural_MDM2_Inhibitor_Discovery_using_ML/` (in this "
                     "repo) — Arjun sir's project, cloned 2026-09-07 from "
                     "`github.com/arjunpahi/Natural_MDM2_Inhibitor_Discovery_using_ML`.\n\n"
                     "**Data lineage:** ChEMBL MDM2-p53 bioactivity → 2,571 raw records → 1,933 unique "
                     "compounds → **645 after Lipinski-RO5 filtering (343 Active / 302 Inactive)**. "
                     "Columns: `molecule_chembl_id` · `canonical_smiles` · `bioactivity_class` "
                     "(Active/Inactive) · `pIC50 = −log10(IC50 M)` · MW/HBA/HBD/logP · `ro5_fulfilled`.\n\n"
                     "*Style: Arjun-sir stepwise format. Protocol mirrors Part_9/20/22 guardrails.*"),
    },
    "dl03": {
        "num": "03",
        "title": "# DL 03: Deep Learning on MDM2 — MACCS Keys (167 structural bits)\n",
        "subtitle": ("**Reference:** MACCS keys — Durant et al. *J Chem Inf Comput Sci* 2002 · residual MLP — "
                     "He et al. CVPR 2016 · 1-D CNN over fingerprints (this project, Part_14) · hybrid "
                     "GCN+MACCS late fusion (this project)\n\n"
                     "**Pipeline:** SMILES → MACCS 167-bit keys + mol graphs → leak-free Murcko scaffold "
                     "split (≈464/52/129, seed 57) → baseline training (validation-only early stopping) → "
                     "**stepwise fine-tuning** (lr → dropout/wd → capacity → batch) → 5-fold GroupKFold CV "
                     "→ Y-randomization (10×) → **test scored once** + bootstrap CI → optional 154k "
                     "screening with Tanimoto AD\n\n"
                     "**Data:** 645 MDM2-p53 inhibitors (`00_data/training/ro5_properties_filtered.csv`, "
                     "343 Active) · **Task:** classification ROC-AUC by default (`DL_TASK=reg` for pIC50)\n\n"
                     "**Original repository:** `Natural_MDM2_Inhibitor_Discovery_using_ML/` (in this "
                     "repo) — Arjun sir's project, cloned 2026-09-07 from "
                     "`github.com/arjunpahi/Natural_MDM2_Inhibitor_Discovery_using_ML`.\n\n"
                     "**Data lineage:** ChEMBL MDM2-p53 bioactivity → 2,571 raw records → 1,933 unique "
                     "compounds → **645 after Lipinski-RO5 filtering (343 Active / 302 Inactive)**. "
                     "Columns: `molecule_chembl_id` · `canonical_smiles` · `bioactivity_class` "
                     "(Active/Inactive) · `pIC50 = −log10(IC50 M)` · MW/HBA/HBD/logP · `ro5_fulfilled`.\n\n"
                     "*Style: Arjun-sir stepwise format. Protocol mirrors Part_9/20/22 guardrails.*"),
    },
    "dl04": {
        "num": "04",
        "title": "# DL 04: Deep Learning on MDM2 — Tier 1: Advanced Architectures (PNA · Graph Transformer · GPS · SMILES Transformer · FT-Transformer)\n",
        "subtitle": ("**Reference:** PNA — Corso et al. NeurIPS 2020 (Principal Neighbourhood "
                     "Aggregation) · Graph Transformer (TransformerConv) — Shi et al. 2021 · "
                     "GPS — Rampášek et al. *Nature Machine Intelligence* 2022 · SMILES "
                     "Transformer — Vaswani et al. NeurIPS 2017 applied char-level · "
                     "FT-Transformer — Gorishniy et al. NeurIPS 2021 (tabular DL)\n\n"
                     "**Pipeline:** SMILES → mol graphs + 10 descriptors + char codes → leak-free "
                     "Murcko scaffold split (≈464/52/129, seed 57) → baseline training (validation-only "
                     "early stopping) → **stepwise fine-tuning** (lr → dropout/wd → capacity → batch) → "
                     "5-fold GroupKFold CV → Y-randomization (10×) → **test scored once** + bootstrap CI "
                     "→ optional 154k screening with Tanimoto AD\n\n"
                     "**Why Tier 1:** the strongest recent architectures that still fit a "
                     "645-compound dataset — degree-aware aggregation (PNA), edge-aware attention "
                     "(TransformerConv), local+global hybrids (GPS), char-level attention (SMILES "
                     "Transformer) and tabular attention (FT-Transformer). Graphormer and 3-D "
                     "models are deliberately excluded: too few molecules.\n\n"
                     "**Data:** 645 MDM2-p53 inhibitors (`00_data/training/ro5_properties_filtered.csv`, "
                     "343 Active) · **Task:** classification ROC-AUC by default (`DL_TASK=reg` for pIC50)\n\n"
                     "**Original repository:** `Natural_MDM2_Inhibitor_Discovery_using_ML/` (in this "
                     "repo) — Arjun sir's project, cloned 2026-09-07 from "
                     "`github.com/arjunpahi/Natural_MDM2_Inhibitor_Discovery_using_ML`.\n\n"
                     "**Data lineage:** ChEMBL MDM2-p53 bioactivity → 2,571 raw records → 1,933 unique "
                     "compounds → **645 after Lipinski-RO5 filtering (343 Active / 302 Inactive)**. "
                     "Columns: `molecule_chembl_id` · `canonical_smiles` · `bioactivity_class` "
                     "(Active/Inactive) · `pIC50 = −log10(IC50 M)` · MW/HBA/HBD/logP · `ro5_fulfilled`.\n\n"
                     "*Style: Arjun-sir stepwise format. Architectures [USER]-adapted from the cited "
                     "papers; protocol mirrors Part_9/20/22 and DL 01-03 guardrails.*"),
    },
    "dl05": {
        "num": "05",
        "title": "# DL 05: Deep Learning on MDM2 — Tier 2: MolCLR Self-Supervised Pretraining (Open-Source Chemistry → fine-tune GIN)\n",
        "subtitle": ("**Reference:** GIN — Xu et al. ICLR 2019 · MolCLR — Wang et al. *Nature Machine "
                     "Intelligence* 2022 (Molecular Contrastive Learning: atom masking / bond "
                     "deletion / subgraph removal) · NT-Xent — Chen et al. ICML 2020 (SimCLR)\n\n"
                     "**Pipeline:** standard open-source unlabeled chemistry SMILES (**default ZINC250k, "
                     "downloaded/cached; NOT the COCONUT screening library**) → two augmented views per "
                     "molecule → GIN encoder + NT-Xent pretraining → fine-tune on 645 MDM2 (leak-free "
                     "Murcko scaffold split 464/52/129, seed 57) → **GIN (scratch) vs GIN-MolCLR** "
                     "head-to-head → stepwise fine-tuning → 5-fold GroupKFold CV → Y-randomization (10×) → "
                     "**test scored once** + bootstrap CI\n\n"
                     "**Why Tier 2:** gives the GNN chemistry knowledge from a model-free, label-free "
                     "open-source corpus before MDM2 fine-tuning; no pretrained weights ever see the 645 "
                     "labels (project rule), so it is honest by construction.\n\n"
                     "**Data:** 645 MDM2-p53 inhibitors (`00_data/training/ro5_properties_filtered.csv`, "
                     "343 Active) + unlabeled pretraining corpus (`DL_PRETRAIN_SOURCE=zinc250k` default, cached) · "
                     "**Task:** classification ROC-AUC by default (`DL_TASK=reg` for pIC50)\n\n"
                     "**Original repository:** `Natural_MDM2_Inhibitor_Discovery_using_ML/` (in this "
                     "repo) — Arjun sir's project, cloned 2026-09-07 from "
                     "`github.com/arjunpahi/Natural_MDM2_Inhibitor_Discovery_using_ML`.\n\n"
                     "**Data lineage:** ChEMBL MDM2-p53 bioactivity → 2,571 raw records → 1,933 unique "
                     "compounds → **645 after Lipinski-RO5 filtering (343 Active / 302 Inactive)**. "
                     "Columns: `molecule_chembl_id` · `canonical_smiles` · `bioactivity_class` "
                     "(Active/Inactive) · `pIC50 = −log10(IC50 M)` · MW/HBA/HBD/logP · `ro5_fulfilled`.\n\n"
                     "*Style: Arjun-sir stepwise format. Architectures [USER]-adapted from the cited "
                     "papers; protocol mirrors Part_9/20/22 and DL 01-03 guardrails.*"),
    },
}

FP_PRIMER_SENTENCES = {
    "dl02": ("**Morgan fingerprints (ECFP4).** For every atom, describe its circular neighbourhood "
             "up to radius 2 bonds (element, bonds, charge…), hash each substructure into one of "
             "2048 bits. Result: a fixed-length vector where \"on\" bits mean *this substructure is "
             "present* — similarity between vectors ≈ chemical similarity (Tanimoto). Deep nets "
             "then learn which substructure co-occurrences predict activity."),
    "dl03": ("**MACCS keys.** A fixed panel of 167 expert-defined substructure questions "
             "(\"contains a carbonyl?\", \"≥2 aromatic rings?\") answered 0/1 per molecule. "
             "Interpretable by construction — each bit is a named feature — but coarser than "
             "hashed circular fingerprints like ECFP4."),
}

FP_META = {
    "dl02": dict(fp_name="Morgan ECFP4 (r=2, 2048 bits)", fp_dim=2048, fp_title="Morgan/ECFP4",
                 fp_paper="GCN trunk Kipf & Welling 2017 + Morgan branch (Rogers & Hahn 2010) late fusion",
                 fp_import="", fp_gen=FP_GEN_MORGAN),
    "dl03": dict(fp_name="MACCS keys (167 bits)", fp_dim=167, fp_title="MACCS",
                 fp_paper="GCN trunk Kipf & Welling 2017 + MACCS branch (Durant 2002) late fusion",
                 fp_import="", fp_gen=FP_GEN_MACCS),
}

CLOSING_MD = {
    "dl01": ("---\n*End of DL 01. Results tables, plots, JSON run-record and the diary block are under "
             "`outputs/dl01/<smoke|full>/`. For fingerprints-based deep models continue with "
             "**DL 02 (Morgan ECFP4)** and **DL 03 (MACCS keys)**.*"),
    "dl02": ("---\n*End of DL 02. Results tables, plots, JSON run-record and the diary block are under "
             "`outputs/dl02/<smoke|full>/`. Compare against **DL 01** (graphs, no fingerprints) and "
             "**DL 03** (MACCS keys).*"),
    "dl03": ("---\n*End of DL 03. Results tables, plots, JSON run-record and the diary block are under "
             "`outputs/dl03/<smoke|full>/`. Compare against **DL 01** (graphs, no fingerprints) and "
             "**DL 02** (Morgan ECFP4).*"),
    "dl04": ("---\n*End of DL 04. Results tables, plots, JSON run-record and the diary block are under "
             "`outputs/dl04/<smoke|full>/`. Compare against **DL 01-03**; Tier-2 (MolCLR "
             "self-supervised pretraining) continues in **DL 05**.*"),
    "dl05": ("---\n*End of DL 05. Results tables, plots, JSON run-record and the diary block are under "
             "`outputs/dl05/<smoke|full>/`. The pretrained trunk is saved as "
             "`molclr_gin_pretrain.pt`; compare 'GIN (scratch)' vs 'GIN-MolCLR' in "
             "`baseline_table.csv`. Tier-1 architectures are in **DL 04**.*"),
}

# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def md_cell(text):
    return new_markdown_cell(text.strip() + "\n")


def sub(src, mapping):
    out = src
    for k, v in mapping.items():
        out = out.replace("%%" + k + "%%", v)
    return out


MODELS_EXTRA_SRC = r'''
# ## 7c. Extended model roster — the 34 reviewed models (opt-in: DL_EXTRA_MODELS=1)
# [USER] 2026-11-23. Every factory is KEYWORD-ONLY and compatible with fit_model():
# (in_dim, out_dim, hidden, dropout). Default runs keep the original 23-model roster;
# set DL_EXTRA_MODELS=1 to train the extended roster. 3-D models are DEFINED but only
# registered once conformer coordinates (d.pos) exist and DL_USE_3D=1 is set.
from torch_geometric.nn import (ChebConv, GMMConv, GatedGraphConv, JumpingKnowledge,
                                APPNP, SGConv, Set2Set, NNConv, SAGPooling,
                                TopKPooling, GCNConv, GINConv,
                                global_add_pool, global_mean_pool, global_max_pool)
from torch_geometric.utils import scatter, to_dense_batch

EXTRA_ON = os.environ.get("DL_EXTRA_MODELS", "0") == "1"
HAS_3D_COORDS = bool(len(graphs_base)) and all(
    getattr(g, "pos", None) is not None for g in graphs_base)
USE_3D = os.environ.get("DL_USE_3D", "0") == "1" and HAS_3D_COORDS

def _xhead(h, out_dim, dropout=0.2):
    return nn.Sequential(nn.Linear(h, 64), nn.ReLU(),
                         nn.Dropout(dropout), nn.Linear(64, out_dim))

def _pool2(d, x):
    return torch.cat([global_mean_pool(x, d.batch), global_max_pool(x, d.batch)], dim=1)

# ---------- 1) Classic / local GNNs (7) ----------
class ChebNet(nn.Module):
    """ChebNet — Defferrard et al. NIPS 2016, Chebyshev spectral conv."""
    def __init__(self, in_dim=78, hidden=HIDDEN, dropout=0.2, out_dim=2):
        super().__init__(); self.dropout = dropout
        self.c1, self.c2 = ChebConv(in_dim, hidden, K=3), ChebConv(hidden, hidden, K=3)
        self.bn = nn.BatchNorm1d(hidden)
        self.head = _xhead(hidden * 2, out_dim, dropout)
    def forward(self, d):
        x = F.relu(self.bn(self.c1(d.x.float(), d.edge_index)))
        x = F.dropout(torch.relu(self.c2(x, d.edge_index)), self.dropout, self.training)
        return self.head(_pool2(d, x))

class MoNet(nn.Module):
    """MoNet / GMMConv — Monti et al. CVPR 2017, Gaussian-mixture pseudo-coordinates."""
    def __init__(self, in_dim=78, hidden=HIDDEN, dropout=0.2, out_dim=2):
        super().__init__(); self.dropout = dropout
        self.m1 = GMMConv(in_dim, hidden, dim=6, kernel_size=4)
        self.m2 = GMMConv(hidden, hidden, dim=6, kernel_size=4)
        self.bn = nn.BatchNorm1d(hidden)
        self.head = _xhead(hidden * 2, out_dim, dropout)
    def forward(self, d):
        pseudo = d.edge_attr.float()
        x = F.relu(self.bn(self.m1(d.x.float(), d.edge_index, pseudo)))
        x = F.dropout(torch.relu(self.m2(x, d.edge_index, pseudo)), self.dropout, self.training)
        return self.head(_pool2(d, x))

class GGNN(nn.Module):
    """GGNN — Li et al. ICLR 2016, gated graph sequence network."""
    def __init__(self, in_dim=78, hidden=HIDDEN, dropout=0.2, out_dim=2):
        super().__init__(); self.dropout = dropout
        self.lin0 = nn.Linear(in_dim, hidden)
        self.gru_conv = GatedGraphConv(hidden, num_layers=3)
        self.head = _xhead(hidden * 2, out_dim, dropout)
    def forward(self, d):
        x = F.dropout(self.gru_conv(self.lin0(d.x.float()), d.edge_index),
                      self.dropout, self.training)
        return self.head(_pool2(d, x))

class GCNIILayer(nn.Module):
    """Initial residual + identity mapping (Chen et al. ICML 2020), PyG 2.8 has no GCNIIConv."""
    def __init__(self, hidden, alpha=0.1, beta=0.1):
        super().__init__(); self.alpha, self.beta = alpha, beta
        self.prop = GCNConv(hidden, hidden, bias=False)
        self.lin = nn.Linear(hidden, hidden, bias=False)
    def forward(self, x, edge_index, x0):
        h0 = x0                                    # x0 is already in hidden space
        s = (1 - self.alpha) * self.prop(x, edge_index) + self.alpha * h0
        return F.relu((1 - self.beta) * s + self.beta * self.lin(s))

class GCNII(nn.Module):
    """GCNII — Chen et al. ICML 2020, deep GCN without over-smoothing."""
    def __init__(self, in_dim=78, hidden=HIDDEN, dropout=0.2, out_dim=2):
        super().__init__(); self.dropout = dropout
        self.to_hidden = nn.Identity() if in_dim == hidden else nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList([GCNIILayer(hidden) for _ in range(3)])
        self.head = _xhead(hidden * 2, out_dim, dropout)
    def forward(self, d):
        x = self.to_hidden(d.x.float()); x0 = x
        for layer in self.layers:
            x = F.dropout(layer(x, d.edge_index, x0), self.dropout, self.training)
        return self.head(_pool2(d, x))

class JKNet(nn.Module):
    """JKNet — Xu et al. ICML 2018, jumping-knowledge aggregation across depths."""
    def __init__(self, in_dim=78, hidden=HIDDEN, dropout=0.2, out_dim=2):
        super().__init__(); self.dropout = dropout
        self.convs = nn.ModuleList([GCNConv(in_dim if i == 0 else hidden, hidden) for i in range(3)])
        self.jk = JumpingKnowledge(mode="cat")
        self.head = _xhead(hidden * 3 * 2, out_dim, dropout)
    def forward(self, d):
        xs = []
        x = d.x.float()
        for conv in self.convs:
            x = F.dropout(torch.relu(conv(x, d.edge_index)), self.dropout, self.training)
            xs.append(x)
        h = self.jk(xs)
        return self.head(torch.cat([global_mean_pool(h, d.batch),
                                    global_max_pool(h, d.batch)], dim=1))

class APPNPNet(nn.Module):
    """APPNP — Gasteiger et al. ICLR 2019, personalized PageRank propagation."""
    def __init__(self, in_dim=78, hidden=HIDDEN, dropout=0.2, out_dim=2):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.prop = APPNP(K=10, alpha=0.1)
        self.head = nn.Linear(hidden, out_dim)
    def forward(self, d):
        h = self.prop(self.mlp(d.x.float()), d.edge_index)
        return self.head(global_mean_pool(h, d.batch))

class SGConvNet(nn.Module):
    """SGC — Wu et al. ICML 2019, simplified graph convolution (K-step smoothing)."""
    def __init__(self, in_dim=78, hidden=HIDDEN, dropout=0.2, out_dim=2):
        super().__init__()
        self.sgc = SGConv(in_dim, hidden, K=2)
        self.head = nn.Linear(hidden, out_dim)
    def forward(self, d):
        return self.head(global_add_pool(torch.relu(self.sgc(d.x.float(), d.edge_index)), d.batch))

# ---------- 2) MPNN frameworks / readouts (6) ----------
class MPNNSet2Set(nn.Module):
    """MPNN + Set2Set — Gilmer et al. ICML 2017; Vinyals et al. ICLR 2016 readout."""
    def __init__(self, in_dim=78, hidden=HIDDEN, dropout=0.2, out_dim=2):
        super().__init__(); self.dropout = dropout; self.hidden = hidden
        self.msg = nn.Sequential(nn.Linear(2 * in_dim + 6, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden))
        self.lin0 = nn.Linear(in_dim, hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.s2s = Set2Set(hidden, processing_steps=3)
        self.head = nn.Linear(2 * hidden, out_dim)
    def forward(self, d):
        x, ei, ea = d.x.float(), d.edge_index, d.edge_attr.float()
        src, dst = ei[0], ei[1]
        m = self.msg(torch.cat([x[src], x[dst], ea], dim=1))
        agg = scatter(m, dst, dim=0, dim_size=x.size(0), reduce="sum")
        h = F.dropout(self.gru(agg, self.lin0(x)), self.dropout, self.training)
        return self.head(self.s2s(h, d.batch))

class GraphNets(nn.Module):
    """Graph Nets — Battaglia et al. 2018, NNConv edge-conditioned message passing."""
    def __init__(self, in_dim=78, hidden=HIDDEN, dropout=0.2, out_dim=2):
        super().__init__(); self.dropout = dropout
        edge_net = nn.Sequential(nn.Linear(6, 32), nn.ReLU(),
                                 nn.Linear(32, in_dim * hidden))
        self.conv1 = NNConv(in_dim, hidden, edge_net)
        self.bn = nn.BatchNorm1d(hidden)
        self.conv2 = NNConv(hidden, hidden, nn.Sequential(
            nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, hidden * hidden)))
        self.head = _xhead(hidden * 2, out_dim, dropout)
    def forward(self, d):
        x = F.relu(self.bn(self.conv1(d.x.float(), d.edge_index, d.edge_attr.float())))
        x = F.dropout(torch.relu(self.conv2(x, d.edge_index, d.edge_attr.float())),
                      self.dropout, self.training)
        return self.head(_pool2(d, x))

class VirtualGIN(nn.Module):
    """Virtual-node GIN — learned global node connected to every graph (Gilmer et al. 2017)."""
    def __init__(self, in_dim=78, hidden=HIDDEN, dropout=0.2, out_dim=2):
        super().__init__(); self.dropout = dropout
        self.lin0 = nn.Linear(in_dim, hidden)
        self.vnode = nn.Parameter(torch.zeros(1, hidden))
        self.convs = nn.ModuleList([GINConv(nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))) for _ in range(3)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(3)])
        self.head = _xhead(hidden * 2, out_dim, dropout)
    def forward(self, d):
        x, ei, bat = self.lin0(d.x.float()), d.edge_index, d.batch
        n, g = x.size(0), int(bat.max().item()) + 1
        vn = n + bat
        ei2 = torch.cat([ei, torch.stack([vn, torch.arange(n, device=x.device)]),
                         torch.stack([torch.arange(n, device=x.device), vn])], dim=1)
        x2 = torch.cat([x, self.vnode.expand(g, -1)], dim=0)
        b2 = torch.cat([bat, torch.arange(g, device=x.device)], dim=0)
        for conv, bn in zip(self.convs, self.bns):
            x2 = F.dropout(torch.relu(bn(conv(x2, ei2))), self.dropout, self.training)
        return self.head(_pool2(d, x2[:n]))

class SAGPoolNet(nn.Module):
    """SAGPool — Lee et al. ICML 2019, self-attention graph pooling."""
    def __init__(self, in_dim=78, hidden=HIDDEN, dropout=0.2, out_dim=2):
        super().__init__(); self.dropout = dropout
        self.conv = GCNConv(in_dim, hidden)
        self.pool = SAGPooling(hidden, ratio=0.5)
        self.head = _xhead(hidden * 2, out_dim, dropout)
    def forward(self, d):
        x = F.relu(self.conv(d.x.float(), d.edge_index))
        x, ei, _, bat, _, _ = self.pool(x, d.edge_index, batch=d.batch)
        return self.head(_pool2(type("B", (), {"batch": bat})(), x))

class TopKPoolNet(nn.Module):
    """TopKPool — Cangea et al. 2018, top-k graph pooling."""
    def __init__(self, in_dim=78, hidden=HIDDEN, dropout=0.2, out_dim=2):
        super().__init__(); self.dropout = dropout
        self.conv = GCNConv(in_dim, hidden)
        self.pool = TopKPooling(hidden, ratio=0.5)
        self.head = _xhead(hidden * 2, out_dim, dropout)
    def forward(self, d):
        x = F.relu(self.conv(d.x.float(), d.edge_index))
        x, ei, _, bat, _, _ = self.pool(x, d.edge_index, batch=d.batch)
        return self.head(_pool2(type("B", (), {"batch": bat})(), x))

class DiffPoolNet(nn.Module):
    """DiffPool-style — Ying et al. NeurIPS 2018, differentiable soft clustering.
    Simplified dense assignment (no auxiliary link-prediction loss)."""
    def __init__(self, in_dim=78, hidden=HIDDEN, dropout=0.2, out_dim=2):
        super().__init__(); self.dropout = dropout
        self.conv = GCNConv(in_dim, hidden)
        self.assign = nn.Linear(hidden, 8)
        self.head = _xhead(hidden * 2, out_dim, dropout)
    def forward(self, d):
        x = torch.relu(self.conv(d.x.float(), d.edge_index))
        xd, mask = to_dense_batch(x, d.batch)
        s = F.softmax(self.assign(xd), dim=1) * mask.unsqueeze(-1)
        h = torch.bmm(s.transpose(1, 2), xd)
        pooled = torch.cat([h.mean(1), h.max(1).values], dim=1)
        return self.head(F.dropout(pooled, self.dropout, self.training))

# ---------- 3) SMILES/sequence LMs — from-scratch architecture analogs (5) ----------
# These are NOT the published pretrained checkpoints: no external weights are used,
# so there is no label-provenance risk. They train on the same char tokens as SmilesCNN.
class TokenTransformerNet(nn.Module):
    def __init__(self, in_dim=80, out_dim=2, hidden=128, dropout=0.2,
                 n_layers=4, n_heads=4, vocab=None, max_len=None):
        super().__init__()
        vocab = int(vocab or globals().get("VOCAB_SIZE", 80))
        max_len = int(max_len or FEATS["chars"].shape[1])
        self.emb = nn.Embedding(vocab, hidden, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, max_len, hidden))
        layer = nn.TransformerEncoderLayer(hidden, n_heads, hidden * 2, dropout,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.head = _xhead(hidden, out_dim, dropout)
    def forward(self, xb):
        tok = xb.long().clamp(0, self.emb.num_embeddings - 1)
        mask = tok.eq(0)
        h = self.enc(self.emb(tok) + self.pos[:, :tok.size(1)])
        h = h.masked_fill(mask.unsqueeze(-1), 0.0)
        pooled = h.sum(1) / mask.eq(False).sum(1, keepdim=True).clamp(min=1)
        return self.head(pooled)

class SmilesBERTNet(TokenTransformerNet):
    """SMILES-BERT — Wang et al. 2019 (bidirectional encoder analog, from scratch)."""
    def __init__(self, in_dim=80, out_dim=2, hidden=128, dropout=0.2, **kw):
        super().__init__(in_dim, out_dim, hidden, dropout, n_layers=4, n_heads=4)
class MolBERTNet(TokenTransformerNet):
    """MolBERT — Fabian et al. 2020 (masked-LM encoder analog, from scratch)."""
    def __init__(self, in_dim=80, out_dim=2, hidden=128, dropout=0.2, **kw):
        super().__init__(in_dim, out_dim, hidden, dropout, n_layers=6, n_heads=4)
class SmilesTransformerNet(TokenTransformerNet):
    """SMILES-Transformer — Honda et al. 2019 (encoder analog, from scratch)."""
    def __init__(self, in_dim=80, out_dim=2, hidden=128, dropout=0.2, **kw):
        super().__init__(in_dim, out_dim, hidden, dropout, n_layers=2, n_heads=4)
class MolBARTNet(TokenTransformerNet):
    """MolBART — Austin et al. 2021 (denoising seq2seq encoder analog, from scratch)."""
    def __init__(self, in_dim=80, out_dim=2, hidden=128, dropout=0.2, **kw):
        super().__init__(in_dim, out_dim, hidden, dropout, n_layers=4, n_heads=8)
class MolT5Net(TokenTransformerNet):
    """MolT5 — Edwards et al. 2022 (span-corruption encoder analog, from scratch)."""
    def __init__(self, in_dim=80, out_dim=2, hidden=128, dropout=0.2, **kw):
        super().__init__(in_dim, out_dim, hidden, dropout, n_layers=4, n_heads=8)

# ---------- 4) Tabular / fingerprint DL (3) ----------
class TabTransformerNet(nn.Module):
    """TabTransformer — Huang et al. 2020, attention over input features."""
    def __init__(self, in_dim=10, out_dim=2, hidden=128, dropout=0.2, **kw):
        super().__init__()
        self.n_feat = min(in_dim, 64)                      # attention over tokens, not 2048 bits
        self.proj = nn.Linear(in_dim, self.n_feat) if in_dim > self.n_feat else nn.Identity()
        self.feat = nn.Linear(1, hidden)
        self.pos = nn.Parameter(torch.zeros(1, self.n_feat, hidden))
        enc = nn.TransformerEncoderLayer(hidden, 4, hidden * 2, dropout,
                                         batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, 2)
        self.head = _xhead(hidden, out_dim, dropout)
    def forward(self, xb):
        h = self.enc(self.feat(self.proj(xb.float()).unsqueeze(-1)) + self.pos[:, :self.n_feat])
        return self.head(h.mean(1))

class NODENet(nn.Module):
    """NODE — Popov et al. ICLR 2020, differentiable oblivious decision tree ensemble
    (simplified soft-threshold leaves; no growing/pruning)."""
    def __init__(self, in_dim=10, out_dim=2, hidden=128, dropout=0.2, **kw):
        super().__init__()
        self.thresh = nn.Parameter(torch.randn(4, in_dim) * 0.1)
        self.scale = nn.Parameter(torch.ones(4, in_dim))
        self.leaves = nn.ModuleList([nn.Linear(in_dim, 1) for _ in range(4)])
        self.head = nn.Linear(4, out_dim)
    def forward(self, xb):
        xb = xb.float()
        outs = [leaf(torch.sigmoid((xb - self.thresh[i]) * self.scale[i]))
                for i, leaf in enumerate(self.leaves)]
        return self.head(torch.cat(outs, dim=1))

class TabNet(nn.Module):
    """TabNet — Arik & Pfister AAAI 2020, sequential attention feature selection
    (simplified two-step attention, no ghost batch norm)."""
    def __init__(self, in_dim=10, out_dim=2, hidden=128, dropout=0.2, **kw):
        super().__init__(); self.dropout = dropout
        self.att = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, in_dim))
        self.feat = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                  nn.Dropout(dropout))
        self.head = _xhead(hidden, out_dim, dropout)
    def forward(self, xb):
        xb = xb.float(); h = 0
        for _ in range(2):
            mask = F.softmax(self.att(xb), dim=1) * xb.size(1)
            h = h + self.feat(xb * mask)
        return self.head(h)

# ---------- 5) 3-D / equivariant hooks — DEFINED, registered only with conformers ----------
class _Need3D(nn.Module):
    """Guard: 3-D models need d.pos (conformers) and e3nn; both are absent by default.
    The classes are kept here so the code is ready when a 3-D data pipeline arrives."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError(
            "3-D model requires conformer coordinates (d.pos) and e3nn. "
            "Set DL_USE_3D=1 after adding 3-D coords to the featurizer and installing e3nn.")
class NequIPNet(_Need3D):    pass   # Batzner et al. Nat. Commun. 2022
class MACENet(_Need3D):      pass   # Batatia et al. NeurIPS 2022
class SphereNet(_Need3D):    pass   # Liu et al. ICLR 2022
class PhysNet(_Need3D):      pass   # Unke & Meuwly 2019

# ---------- Registry ----------
# DL05 is graph-only and never defines FEATS; array models are registered only when
# the notebook actually has a feature matrix for them.
FEATS = globals().setdefault("FEATS", {})
array_key = ("desc" if "desc" in FEATS else
             ("fp" if "fp" in FEATS else ("chars" if "chars" in FEATS else None)))
EXTRA_REGISTRY = {
    "ChebNet": (ChebNet, "graph", None), "MoNet": (MoNet, "graph", None),
    "GGNN": (GGNN, "graph", None), "GCNII": (GCNII, "graph", None),
    "JKNet": (JKNet, "graph", None), "APPNP": (APPNPNet, "graph", None),
    "SGC": (SGConvNet, "graph", None), "MPNNSet2Set": (MPNNSet2Set, "graph", None),
    "GraphNets": (GraphNets, "graph", None), "VirtualGIN": (VirtualGIN, "graph", None),
    "SAGPool": (SAGPoolNet, "graph", None), "TopKPool": (TopKPoolNet, "graph", None),
    "DiffPool": (DiffPoolNet, "graph", None),
}
if array_key:
    EXTRA_REGISTRY.update({
        "TabTransformer": (TabTransformerNet, "array", array_key),
        "NODE": (NODENet, "array", array_key), "TabNet": (TabNet, "array", array_key)})
if "chars" in FEATS:
    EXTRA_REGISTRY.update({
        "SmilesBERT": (SmilesBERTNet, "array", "chars"),
        "MolBERT": (MolBERTNet, "array", "chars"),
        "SmilesTransformerLM": (SmilesTransformerNet, "array", "chars"),
        "MolBART": (MolBARTNet, "array", "chars"),
        "MolT5": (MolT5Net, "array", "chars")})
if USE_3D:
    EXTRA_REGISTRY.update({
        "NequIP": (NequIPNet, "graph", None), "MACE": (MACENet, "graph", None),
        "SphereNet": (SphereNet, "graph", None), "PhysNet": (PhysNet, "graph", None)})

EXTRA_PAPERS = {
    "ChebNet": "Defferrard et al. NIPS 2016", "MoNet": "Monti et al. CVPR 2017",
    "GGNN": "Li et al. ICLR 2016", "GCNII": "Chen et al. ICML 2020",
    "JKNet": "Xu et al. ICML 2018", "APPNP": "Gasteiger et al. ICLR 2019",
    "SGC": "Wu et al. ICML 2019", "MPNNSet2Set": "Gilmer ICML 2017 + Vinyals ICLR 2016",
    "GraphNets": "Battaglia et al. 2018", "VirtualGIN": "virtual-node MPNN lineage",
    "SAGPool": "Lee et al. ICML 2019", "TopKPool": "Cangea et al. 2018",
    "DiffPool": "Ying et al. NeurIPS 2018 (simplified)",
    "TabTransformer": "Huang et al. 2020", "NODE": "Popov et al. ICLR 2020 (simplified)",
    "TabNet": "Arik & Pfister AAAI 2020 (simplified)",
    "SmilesBERT": "Wang et al. 2019 (from-scratch analog, no external weights)",
    "MolBERT": "Fabian et al. 2020 (from-scratch analog)",
    "SmilesTransformerLM": "Honda et al. 2019 (from-scratch analog)",
    "MolBART": "Austin et al. 2021 (from-scratch analog)",
    "MolT5": "Edwards et al. 2022 (from-scratch analog)",
    "NequIP": "Batzner et al. 2022 (needs conformers + e3nn)",
    "MACE": "Batatia et al. 2022 (needs conformers + e3nn)",
    "SphereNet": "Liu et al. 2022 (needs conformers + e3nn)",
    "PhysNet": "Unke & Meuwly 2019 (needs conformers + e3nn)"}

if EXTRA_ON:
    DL_MODELS.update(EXTRA_REGISTRY)
    DL_PAPERS.update(EXTRA_PAPERS)
    print(f"[S07c] extended roster ON: +{len(EXTRA_REGISTRY)} models "
          f"(total registry {len(DL_MODELS)}), array key={array_key}")
else:
    print("[S07c] extended roster OFF: set DL_EXTRA_MODELS=1 to enable 34-model roster")
if not HAS_3D_COORDS:
    print("[S07c] 3-D models defined but NOT registered: no conformer coordinates (d.pos). "
          "Add 3-D coords to the featurizer, install e3nn, then set DL_USE_3D=1.")
'''


SSL_EXTRA_SRC = r'''
# ## 7d. Alternative 2-D self-supervised pretraining (opt-in: DL_SSL_METHOD)
# [USER] GraphCL / JOAO / GCC / Hu-attr — reuse the label-free MolCLR corpus and encoder.
# Default remains MolCLR untouched; any alternative overwrites WARM_START_* only when asked.
SSL_METHOD = os.environ.get("DL_SSL_METHOD", "molclr").strip().lower()
if EXTRA_ON and SSL_METHOD != "molclr":
    assert SSL_METHOD in ("graphcl", "joao", "gcc", "hu"), f"unknown DL_SSL_METHOD={SSL_METHOD}"
    def _ssl_view(g, method, epoch=0):
        if method == "graphcl":      # graph-cluster-style: only subgraph + node drop
            return molclr_augment(g, p_mask=0.20, p_bond=0.0, p_sub=0.20)
        if method == "gcc":          # subgraph instance discrimination
            return molclr_augment(g, p_mask=0.0, p_bond=0.0, p_sub=1.0)
        if method == "joao":         # alternate the two policies per epoch (simplified JOAO)
            return molclr_augment(g, p_mask=0.15, p_bond=0.15, p_sub=0.25) if epoch % 2 == 0 \
                   else molclr_augment(g, p_mask=0.30, p_bond=0.05, p_sub=0.10)
        return molclr_augment(g, p_mask=0.35, p_bond=0.0, p_sub=0.0)   # Hu-style attribute masking proxy

    enc_s = GINNet().to(DEVICE); enc_s.return_pooled = True
    proj_s = nn.Sequential(nn.Linear(2 * HIDDEN, HIDDEN), nn.ReLU(),
                           nn.Linear(HIDDEN, HIDDEN)).to(DEVICE)
    opt_s = torch.optim.Adam(list(enc_s.parameters()) + list(proj_s.parameters()), lr=1e-3)
    _ps = list(enc_s.parameters()) + list(proj_s.parameters())
    for ep in range(1, max(1, PRE_EPOCHS // 2) + 1):
        enc_s.train(); proj_s.train()
        perm = _rng.permutation(len(_pick)); tot, nb = 0.0, 0
        for i0 in range(0, len(perm), PRE_BATCH):
            va = [_ssl_view(mol_to_graph(_pool[int(i)]), SSL_METHOD, ep)
                  for i in _pick[perm[i0:i0 + PRE_BATCH]]]
            vb = [_ssl_view(mol_to_graph(_pool[int(i)]), SSL_METHOD, ep)
                  for i in _pick[perm[i0:i0 + PRE_BATCH]]]
            if len(va) < 2:
                continue
            z1 = proj_s(enc_s(Batch.from_data_list(va).to(DEVICE)))
            z2 = proj_s(enc_s(Batch.from_data_list(vb).to(DEVICE)))
            if SSL_METHOD == "hu":   # attribute reconstruction proxy alongside NT-Xent
                z1 = F.normalize(z1, dim=1); z2 = F.normalize(z2, dim=1)
                loss = F.mse_loss(z1, z2)
            else:
                loss = nt_xent(z1, z2)
            opt_s.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(_ps, 5.0); opt_s.step()
            tot += float(loss); nb += 1
        print(f"[S07d] {SSL_METHOD} epoch {ep}: loss {tot / max(nb, 1):.4f} ({nb} batches)")
    WARM_START_SD = {k: v.detach().cpu().clone() for k, v in enc_s.state_dict().items()
                     if not k.startswith("head")}
    WARM_START_NAME, WARM_START_TAG = f"GIN-{SSL_METHOD}", f"{SSL_METHOD} pretrain ({_src_used})"
    torch.save(WARM_START_SD, OUT_DIR / f"{SSL_METHOD}_gin_pretrain.pt")
    print(f"[S07d] WARM_START overridden -> {WARM_START_NAME}; default MolCLR run was unchanged")
elif EXTRA_ON:
    print("[S07d] SSL alternatives available: DL_SSL_METHOD=graphcl|joao|gcc|hu (default molclr)")
'''

UNCERTAINTY_SRC = r'''
# ## 16b. Uncertainty / validation-only diagnostics (opt-in: DL_UNCERTAINTY=1)
# [USER] No test split is touched here: every number below comes from TRAINED models on
# the validation rows. Deep ensembles, MC-dropout (Bayesian proxy), conformal intervals,
# evidential-proxy calibration and a dual-head multi-task model.
import json
if EXTRA_ON and os.environ.get("DL_UNCERTAINTY", "0") == "1" and len(base_df):
    best_name = str(base_df.iloc[0]["model"])
    _, best_kind, best_fkey = DL_MODELS[best_name]
    va_ds = (_graph_batch_y(va_idx, _y_task(), best_fkey) if best_kind == "graph" else
             TensorDataset(torch.tensor(FEATS[best_fkey][va_idx]),
                           torch.tensor(_y_task())[torch.tensor(va_idx)]))
    va_ld = make_loader(va_ds, BATCH_ARRAY if best_kind == "array" else BATCH_GRAPH,
                        False, best_kind)
    peers = [n for n in TRAINED if DL_MODELS[n][1] == best_kind][:3] or [best_name]
    _, y_true = predict(TRAINED[best_name], va_ld, best_kind, DEVICE)
    preds = [predict(TRAINED[n], va_ld, best_kind, DEVICE)[1] for n in peers]
    ens = np.mean(np.stack(preds), axis=0)
    unc = {"models": peers, "kind": best_kind,
           "ensemble_val": task_metrics(y_true, ens)}

    # MC-dropout Bayesian proxy: 20 stochastic forward passes on the validation loader.
    m = TRAINED[best_name]; m.train()
    draws = []
    with torch.no_grad():
        for _ in range(20):
            ps = []
            for batch in va_ld:
                if best_kind == "graph":
                    b = batch.to(DEVICE); out = m(b); yy = b.y.view(-1).cpu().numpy()
                else:
                    xb, yb = batch; out = m(xb.to(DEVICE)); yy = yb.numpy()
                ps.append(torch.softmax(out.float(), 1)[:, 1].cpu().numpy() if TASK == "cls"
                          else out.view(-1).cpu().numpy())
            draws.append(np.concatenate(ps))
    unc["mc_dropout"] = {"val_mean_auc": task_metrics(y_true, np.mean(draws, 0)).get("AUC"),
                         "mean_predictive_std": float(np.mean(np.std(draws, 0))),
                         "draws": 20}
    m.eval()                                   # restore deterministic mode for screening

    # Conformal interval from validation absolute residuals (calibration only).
    resid = np.abs(y_true - ens)
    unc["conformal"] = {"level": 0.90, "width": float(np.quantile(resid, 0.90)),
                        "val_coverage": float(np.mean(resid <= np.quantile(resid, 0.90))),
                        "note": "validation-calibrated; test split not re-scored"}
    # Evidential proxy: does predictive uncertainty track error?
    sd = np.std(draws, 0)
    try:
        from scipy.stats import spearmanr
        unc["evidential_proxy"] = {"spearman_std_vs_abs_error":
                                   float(spearmanr(sd, resid).correlation)}
    except Exception as e:
        unc["evidential_proxy"] = {"error": str(e)}

    # Dual-head multi-task cls+reg MLP, validation-only selection.
    fk = "desc" if "desc" in FEATS else ("fp" if "fp" in FEATS else None)
    if fk:
        mt = nn.Sequential(nn.Linear(FEATS[fk].shape[1], 64), nn.ReLU(),
                           nn.Linear(64, 2)).to(DEVICE)
        opt = torch.optim.Adam(mt.parameters(), lr=1e-3)
        yr = torch.tensor(((y_reg_all - Y_REG_MEAN) / Y_REG_STD).astype(np.float32))
        yc = torch.tensor(y_cls_all.astype(np.float32))
        for _ in range(3 if SMOKE else 15):
            mt.train(); opt.zero_grad()
            xtr = torch.tensor(FEATS[fk][tr_idx], dtype=torch.float32).to(DEVICE)
            out = mt(xtr)                      # train rows ONLY: test labels never seen
            loss = F.cross_entropy(out, yc[tr_idx].to(DEVICE).long()) + \
                   F.mse_loss(out[:, 0], yr[tr_idx].to(DEVICE))
            loss.backward(); opt.step()
        mt.eval()
        with torch.no_grad():
            out = mt(torch.tensor(FEATS[fk][va_idx], dtype=torch.float32).to(DEVICE)).cpu().numpy()
        unc["multitask"] = {"val_cls_auc": task_metrics(y_cls_all[va_idx], out[:, 1]).get("AUC"),
                            "val_reg_rmse": float(np.sqrt(np.mean(
                                (out[:, 0] * Y_REG_STD + Y_REG_MEAN - y_reg_all[va_idx]) ** 2)))}

    p = OUT_DIR / "uncertainty.json"
    p.write_text(json.dumps(unc, indent=2, default=str))
    print(f"[S16b] uncertainty diagnostics -> {p} | ensemble peers={peers}")
elif EXTRA_ON:
    print("[S16b] uncertainty diagnostics available: set DL_UNCERTAINTY=1 (validation only)")
'''


def build(nb_id):
    meta = NB_TITLES[nb_id]
    num = meta["num"]
    title_short = meta["title"].splitlines()[0].lstrip("# ")
    base = {"NB_ID": nb_id, "NB_NUM": num, "NB_TITLE_SHORT": title_short}
    cells = [md_cell(meta["title"] + meta["subtitle"])]

    # 1. chemistry primer
    if nb_id == "dl01":
        cells.append(md_cell(PRIMER_NB1))
    elif nb_id == "dl04":
        cells.append(md_cell(PRIMER_T1))
    elif nb_id == "dl05":
        cells.append(md_cell(PRIMER_T2))
    else:
        cells.append(md_cell(sub(PRIMER_FP_TEMPLATE, {
            "NB_NUM": num, "FP_PRIMER": FP_PRIMER_SENTENCES[nb_id]})))

    # 2-5 common
    cells += [new_code_cell(SETUP_SRC.strip()), new_code_cell(REPRO_SRC.strip()),
              new_code_cell(UTIL_SRC.strip()), new_code_cell(LOAD_SRC.strip())]

    # 6. split FIRST (defines tr_idx for the scaler), then graphs, then features
    if nb_id in ("dl01", "dl04"):
        feat = FEAT_NB1_SRC
    elif nb_id == "dl05":
        feat = ""
    else:
        feat = sub(FEAT_FP_SRC, {
            "NB_FP_TITLE": FP_META[nb_id]["fp_title"], "FP_NAME": FP_META[nb_id]["fp_name"],
            "FP_DIM": str(FP_META[nb_id]["fp_dim"]),
            "FP_IMPORT": FP_META[nb_id]["fp_import"], "FP_GEN": FP_META[nb_id]["fp_gen"]})
    cells.append(new_code_cell(SPLIT_SRC.strip() + "\n\n" + GRAPH_FEAT_SRC.strip() +
                               ("\n\n" + feat.strip() if feat else "")))

    # 7. models
    if nb_id == "dl01":
        models = MODELS_NB1_SRC
    elif nb_id == "dl04":
        models = MODELS_T1_SRC
    elif nb_id == "dl05":
        models = MODELS_T2_SRC
    else:
        models = sub(MODELS_FP_SRC, {
            "FP_NAME": FP_META[nb_id]["fp_name"], "FP_DIM": str(FP_META[nb_id]["fp_dim"]),
            "NB_NUM": num, "FP_PAPER": FP_META[nb_id]["fp_paper"]})
    cells.append(new_code_cell(models.strip()))
    cells.append(new_code_cell(MODELS_EXTRA_SRC.strip()))  # [USER] 34-model extended roster (opt-in)

    # 7b. DL 05: MolCLR self-supervised pretraining (defines WARM_START_* before the engine)
    if nb_id == "dl05":
        cells.append(new_code_cell(PRETRAIN_T2_SRC.strip()))
        cells.append(new_code_cell(SSL_EXTRA_SRC.strip()))  # [USER] GraphCL/JOAO/GCC/Hu (opt-in)

    # 8-9. engine + baseline
    cells += [new_code_cell(ENGINE_SRC.strip()), new_code_cell(BASELINE_SRC.strip())]
    if nb_id == "dl01":
        cells.append(new_code_cell(DEEPCHEM_SRC.strip()))

    # 10. stepwise fine-tuning (+ pretrain hook everywhere; HF ladder in NB1)
    cells.append(new_code_cell(FT_SRC.strip()))
    cells.append(new_code_cell(PRETRAIN_HOOK_SRC.strip()))
    if nb_id == "dl01":
        cells.append(new_code_cell(HF_FT_SRC.strip()))

    # 11-16
    cells += [new_code_cell(CV_SRC.strip()), new_code_cell(YRAND_SRC.strip()),
              new_code_cell(TEST_SRC.strip()),
              new_code_cell(UNCERTAINTY_SRC.strip()),  # [USER] validation-only uncertainty (opt-in)
              new_code_cell(SUMMARY_SRC.strip()),
              new_code_cell(SCREEN_SRC.strip()), new_code_cell(DIARY_SRC.strip())]

    cells.append(md_cell(CLOSING_MD[nb_id]))

    # substitute per-notebook tokens in every code cell
    for c in cells:
        if c.cell_type == "code":
            c.source = sub(c.source, base)
    nb = new_notebook(cells=cells,
                      metadata={"kernelspec": {"display_name": "Python 3", "language": "python",
                                               "name": "python3"},
                                "language_info": {"name": "python", "version": "3.13"}})
    return nb


NOTEBOOK_FILES = {
    "dl01": "DL_01_Graph_Models_No_Fingerprints.ipynb",
    "dl02": "DL_02_Morgan_Fingerprint_DL.ipynb",
    "dl03": "DL_03_MACCS_Keys_DL.ipynb",
    "dl04": "DL_04_Tier1_Advanced_Architectures.ipynb",
    "dl05": "DL_05_MolCLR_Pretrain_Finetune.ipynb",
}


def validate(nb, path):
    nbformat.validate(nb)
    for i, c in enumerate(nb.cells):
        if c.cell_type == "code":
            try:
                ast.parse(c.source)
            except SyntaxError as e:
                raise SystemExit(f"AST error in {path.name} cell {i}: {e}")
    left = sum(("%%" in c.source) for c in nb.cells if c.cell_type == "code")
    if left:
        raise SystemExit(f"{path.name}: {left} code cells still contain %%tokens%%")


def main():
    for nb_id, fname in NOTEBOOK_FILES.items():
        nb = build(nb_id)
        path = HERE / fname
        validate(nb, path)
        nbformat.write(nb, path)
        n_code = sum(c.cell_type == "code" for c in nb.cells)
        n_md = sum(c.cell_type == "markdown" for c in nb.cells)
        print(f"built {fname}: {len(nb.cells)} cells ({n_code} code / {n_md} md), nbformat+AST clean")


if __name__ == "__main__":
    main()
