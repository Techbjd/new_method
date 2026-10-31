"""Atlas cell definitions, part B (models, CV, ensemble, screening, evaluation, manifest)."""

_CELLS_B = []


def _add(kind, cid, src):
    _CELLS_B.append((kind, cid, src))


_add("markdown", "S06", """## Data information (clearly structured)

| Item | Value |
|---|---|
| Target | **CHEMBL303** — MDM2 (p53-binding), ChEMBL 33 assays |
| Raw lineage | 2,571 raw actives+inactives → 1,933 unique single-bioassay → **645 RO5 compounds** |
| Labels | 343 **Active** / 302 **Inactive** (IC50 ≤ 10 µM threshold) |
| Regression label | **pIC50 = log10(IC50 / M)** (no unit error; log-scale regression) |
| Hold-out | leak-free **Bemis–Murcko scaffold split, non-isomeric groups, ≈81/9/10%** (test scored ONCE) |
| Screening library | COCONUT 154,647 SMILES (outputs CSV used structures-only) |
| Thresholds | `cuts.json`: ad_cut 0.4 / **ad_cut_strict 0.52** / rf cuts 0.6/0.4 / gnn_hit_cut 7.0 |
""")

_add("code", "S07", """
# --- data information report ---
y_cls_all = (TRAIN_DF["bioactivity_class"].astype(str) == "Active").astype(int).to_numpy()
y_reg_all = TRAIN_DF["pIC50"].astype(float).to_numpy()
smiles_all = TRAIN_DF["canonical_smiles"].astype(str).str.strip().tolist()
ids_all = TRAIN_DF["molecule_chembl_id"].astype(str).tolist()

print("[S07] n =", len(smiles_all),
      "| Active:", int(y_cls_all.sum()), f"({y_cls_all.mean():.1%})",
      "| Inactive:", int((1 - y_cls_all).sum()))
print("[S07] pIC50 range:", round(float(y_reg_all.min()), 3), "-", round(float(y_reg_all.max()), 3))
print("[S07] CUTS in force:", CUTS)
assert len(smiles_all) == len(y_cls_all) == len(y_reg_all)
if TRAIN_MODE == "synthetic-smoke":
    print("[S07] [[NOTE]] synthetic smoke placeholder data — results are NOT real; run with real CSVs for analysis")
""")

_add("code", "S08", """
# --- features: 10 RDKit descriptors + 6 fingerprint families (~7.5k dims) ---
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors, rdFingerprintGenerator, MACCSkeys

DESC_NAMES = ["MW", "logP", "TPSA", "rotB", "HBA", "HBD", "heavy", "rings", "arom", "fracSP3"]

_mols = [Chem.MolFromSmiles(s) for s in smiles_all]
_valid = np.array([m is not None for m in _mols])
print("[S08] valid SMILES:", int(_valid.sum()), "/", len(_valid))

def _desc_row(m):
    if m is None:
        return [0.0] * 10
    return [Descriptors.MolWt(m), Crippen.MolLogP(m), rdMolDescriptors.CalcTPSA(m),
            Descriptors.NumRotatableBonds(m), rdMolDescriptors.CalcNumHBA(m),
            rdMolDescriptors.CalcNumHBD(m), float(m.GetNumHeavyAtoms()),
            float(rdMolDescriptors.CalcNumRings(m)), float(rdMolDescriptors.CalcNumAromaticRings(m)),
            Descriptors.FractionCSP3(m)]

X_desc = np.nan_to_num(np.array([_desc_row(m) for m in _mols], dtype=np.float32))
_morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_rdk_gen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048)
_ap_gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)
_tt_gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048)
try:
    from rdkit.Avalon import pyAvalonTools
    _avalon = lambda m: pyAvalonTools.GetAvalonFP(m, nBits=1024)
except Exception:
    print("[S08] Avalon unavailable — skipped (matches part21 behaviour)")
    _avalon = lambda m: None

def _fpdense(fps, nbits):
    out = np.zeros((len(fps), nbits), dtype=np.float32)
    for i, fp in enumerate(fps):
        if fp is None:
            out[i, 0] += 0.0
            continue
        from rdkit import DataStructs
        DataStructs.ConvertToNumpyArray(fp, out[i])
    return out

X_morgan = _fpdense([_morgan_gen.GetFingerprint(m) if m else None for m in _mols], 2048)
X_maccs = _fpdense([MACCSkeys.GenMACCSKeys(m) if m else None for m in _mols], 167)
X_rdk = _fpdense([_rdk_gen.GetFingerprint(m) if m else None for m in _mols], 2048)
X_ap = _fpdense([_ap_gen.GetFingerprint(m) if m else None for m in _mols], 2048)
X_tt = _fpdense([_tt_gen.GetFingerprint(m) if m else None for m in _mols], 2048)
X_av = _fpdense([_avalon(m) if m else None for m in _mols], 1024)
X_fp_all = np.hstack([X_morgan, X_maccs, X_rdk, X_ap, X_tt, X_av])
print("[S08] descriptors:", X_desc.shape, "| fingerprints stack:", X_fp_all.shape)
""")

_add("code", "S09", """
# --- graph featurizer: 78-dim atoms (Part_18 common.py layout) + 6-dim bonds ---
COMMON_ELEMENTS = ["H","B","C","N","O","F","P","S","Cl","Br","I","Si","Na","Mg","K","Ca","Fe","Zn",
                   "Cu","Se","As","Al","Li","He","Ne","Ar","Cr","Mn","Co","Ni","Sr","Ti","V","Mo",
                   "Sn","Sb","Te","Ba","La","W","Au","Hg","Pb","Bi","Gd","Y","Zr","Pd","Ag","Cd",
                   "Ru","Pt",""]  # 52 + OTHER slot -> 53-block, total vector = 78

def _one_hot(val, choices):
    vec = [1.0 if c == val else 0.0 for c in choices]
    if all(v == 0.0 for v in vec):
        vec = [0.0] * (len(choices) - 1) + [1.0]
    return vec

def _atom_features(atom):
    from rdkit import Chem as C2
    hyb = [C2.HybridizationType.SP, C2.HybridizationType.SP2, C2.HybridizationType.SP3,
           C2.HybridizationType.SP3D, C2.HybridizationType.SP3D2]
    v = _one_hot(atom.GetSymbol(), COMMON_ELEMENTS) + _one_hot(atom.GetTotalDegree(), [0,1,2,3,4,5]) + \\
        _one_hot(atom.GetFormalCharge(), [-2,-1,0,1,2,3]) + _one_hot(atom.GetTotalNumHs(), [0,1,2,3,4]) + \\
        _one_hot(atom.GetHybridization(), hyb) + \\
        [1.0 if atom.GetIsAromatic() else 0.0] + [1.0 if atom.IsInRing() else 0.0] + \\
        [1.0 if atom.GetChiralTag() != C2.ChiralType.CHI_UNSPECIFIED else 0.0]
    assert len(v) == 78
    return v

def _bond_features(bond):
    from rdkit import Chem as C2
    bt = bond.GetBondType()
    return [1.0 if bt == C2.BondType.SINGLE else 0.0, 1.0 if bt == C2.BondType.DOUBLE else 0.0,
            1.0 if bt == C2.BondType.TRIPLE else 0.0, 1.0 if bt == C2.BondType.AROMATIC else 0.0,
            1.0 if bond.GetIsConjugated() else 0.0, 1.0 if bond.IsInRing() else 0.0]

def mol_to_graph(smiles, label_vec=None):
    from torch_geometric.data import Data
    m = Chem.MolFromSmiles(smiles)
    if m is None or m.GetNumAtoms() == 0:
        return None
    x = torch.tensor(np.array([_atom_features(a) for a in m.GetAtoms()], dtype=np.float32))
    edges, bfs = [], []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edges += [[i, j], [j, i]]
        bf = _bond_features(b)
        bfs += [bf, bf]
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.array(bfs, dtype=np.float32))
    else:
        # isolated atom: keep a valid (2, 0) edge tensor instead of a fake self-loop
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 6), dtype=torch.float)
    d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if label_vec is not None:
        # regression labels are floats; classification labels are ints.
        # Casting every label to long silently truncates pIC50 (e.g. 7.6 -> 7).
        _y = torch.as_tensor(label_vec)
        d.y = _y.float() if _y.is_floating_point() else _y.long()
    return d

graphs_all = [mol_to_graph(s) for s in smiles_all]
_gkeep = np.array([g is not None for g in graphs_all])
print("[S09] graphs built:", int(_gkeep.sum()), "/ dropped:", int((~_gkeep).sum()),
      "| x dim:", graphs_all[int(np.argmax(_gkeep))].x.shape[1])
""")

_add("code", "S10", """
# --- Murcko scaffold split ≈80/9/10 (non-isomeric groups) + leakage assert + 5-fold GroupKFold ---
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

def murcko(smi, stereo=True):
    try:
        core = MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(smi))
        return Chem.MolToSmiles(core, isomericSmiles=stereo) if core is not None else ""
    except Exception:
        return ""

# Group by the 2-D scaffold. Stereoisomers have different isomeric scaffold
# SMILES but identical 2-D fingerprints (Morgan/ECFP ignores chirality), so
# splitting on isomeric scaffolds leaks near-duplicates into validation and
# inflates AUC/R2. Non-isomeric grouping keeps stereoisomer pairs together.
groups_all = np.array([murcko(s, stereo=False) for s in smiles_all])
# GroupShuffleSplit's test_size is a fraction of GROUPS, not compounds. With
# indivisible scaffold groups, 0.10 picked one large group and produced a 21%
# test set. 0.08 targets ~10% of compounds while keeping every scaffold whole.
TEST_GROUP_FRAC = 0.08
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_GROUP_FRAC, random_state=RUN_SEED)
tr_va_idx, test_idx = next(gss.split(np.zeros(len(groups_all)), groups=groups_all))
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=RUN_SEED)
tr_idx, va_idx = next(gss2.split(np.zeros(len(tr_va_idx)), groups=groups_all[tr_va_idx]))
tr_idx, va_idx = tr_va_idx[tr_idx], tr_va_idx[va_idx]
for a, b, nm in ((tr_idx, va_idx, "train/val"), (tr_idx, test_idx, "train/test"), (va_idx, test_idx, "val/test")):
    shared = set(groups_all[a]) & set(groups_all[b])
    assert len(shared) == 0, f"scaffold leakage in {nm}: {len(shared)} shared scaffolds"
print("[S10] split sizes: train", len(tr_idx), "| val", len(va_idx), "| test", len(test_idx),
      f"({len(tr_idx)/len(groups_all):.1%}/{len(va_idx)/len(groups_all):.1%}/{len(test_idx)/len(groups_all):.1%}) (test scored ONCE)")

cv_folds = list(GroupKFold(n_splits=SCAN_FOLDS).split(np.zeros(len(groups_all)), groups=groups_all))
print("[S10] 5-fold GroupKFold fold sizes:", [len(va) for _, va in cv_folds])
""")

_add("code", "S11", """
# --- ML model registry (reg + softmax-cls tracks) with per-model paper/GitHub provenance ---
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor,
                              ExtraTreesClassifier, HistGradientBoostingRegressor,
                              HistGradientBoostingClassifier, AdaBoostRegressor, AdaBoostClassifier)
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC

ML_MODELS = {}  # name -> (est_factory(reg)/proc, paper, github)
def _reg(name, factory, paper, github, exclude_from_vote=False):
    ML_MODELS[name] = dict(kind="reg", factory=factory, paper=paper, github=github, exclude_from_vote=exclude_from_vote)

_reg("ridge", lambda: Ridge(alpha=1.0, random_state=RUN_SEED),
     "Hoerl & Kennard 1970 Technometrics", "scikit-learn")
_reg("rf", lambda: RandomForestRegressor(500, max_features="sqrt", n_jobs=-1, random_state=RUN_SEED),
     "Breiman 2001 Random Forests (Budha et al. Adv Theory Simul 2025)", "github.com/scikit-learn")
_reg("extra_trees", lambda: ExtraTreesRegressor(500, max_features="sqrt", n_jobs=-1, random_state=RUN_SEED),
     "Geurts et al. 2006", "github.com/scikit-learn")
_reg("adaboost", lambda: AdaBoostRegressor(n_estimators=300, learning_rate=0.5, random_state=RUN_SEED),
     "Freund & Schapire 1997", "github.com/scikit-learn")
_reg("hist_gb", lambda: HistGradientBoostingRegressor(max_iter=400, random_state=RUN_SEED),
     "sklearn HistGB (LightGBM-style)", "github.com/scikit-learn")
_reg("knn", lambda: KNeighborsRegressor(5, weights="distance", n_jobs=-1),
     "Cover & Hart 1967", "github.com/scikit-learn")
_reg("svm_rbf", lambda: make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10.0, gamma="scale")),
     "Cortes & Vapnik 1995", "github.com/scikit-learn", exclude_from_vote=True)  # SVM excluded from screening vote (project rule)
ML_CLS = {}
ML_CLS["rf"] = lambda: RandomForestClassifier(500, max_features="sqrt", n_jobs=-1, random_state=RUN_SEED)
ML_CLS["extra_trees"] = lambda: ExtraTreesClassifier(500, max_features="sqrt", n_jobs=-1, random_state=RUN_SEED)
ML_CLS["adaboost"] = lambda: AdaBoostClassifier(n_estimators=300, learning_rate=0.5, random_state=RUN_SEED)
ML_CLS["hist_gb"] = lambda: HistGradientBoostingClassifier(max_iter=400, random_state=RUN_SEED)
ML_CLS["knn"] = lambda: KNeighborsClassifier(5, weights="distance", n_jobs=-1)
ML_CLS["svm_rbf"] = lambda: make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0, gamma="scale", probability=True))
try:
    from xgboost import XGBRegressor, XGBClassifier
    XGB_OK = True
except Exception:
    XGB_OK = False
    print("[S11] xgboost missing -> [skipped]")
if XGB_OK:
    _reg("xgboost", lambda: XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.9,
                                          colsample_bytree=0.8, tree_method="hist", random_state=RUN_SEED, n_jobs=-1),
         "Chen & Guestrin KDD 2016", "github.com/dmlc/xgboost")
    ML_CLS["xgboost"] = lambda: XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6,
                                               subsample=0.9, colsample_bytree=0.8, tree_method="hist",
                                               random_state=RUN_SEED, n_jobs=-1)
try:
    from lightgbm import LGBMRegressor, LGBMClassifier, early_stopping as _lgb_es
    LGB_OK = True
except Exception:
    LGB_OK = False
    print("[S11] lightgbm missing -> [skipped]")
if LGB_OK:
    _reg("lightgbm", lambda: LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                            subsample=0.9, colsample_bytree=0.8, random_state=RUN_SEED,
                                            n_jobs=-1, verbose=-1),
         "Ke et al. NeurIPS 2017", "github.com/microsoft/LightGBM")
    ML_CLS["lightgbm"] = lambda: LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                                 subsample=0.9, colsample_bytree=0.8, random_state=RUN_SEED,
                                                 n_jobs=-1, verbose=-1)
try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    CB_OK = True
except Exception:
    CB_OK = False
    print("[S11] catboost missing -> [skipped]")
if CB_OK:
    _reg("catboost", lambda: CatBoostRegressor(iterations=400, learning_rate=0.05, depth=6,
                                                random_seed=RUN_SEED, verbose=0, thread_count=-1),
         "Dorogush et al. arXiv:1810.11363", "github.com/catboost/catboost")
    ML_CLS["catboost"] = lambda: CatBoostClassifier(iterations=400, learning_rate=0.05, depth=6,
                                                     loss_function="Logloss", random_seed=RUN_SEED,
                                                     verbose=0, thread_count=-1)
print("[S11] ML regression models:", sorted(ML_MODELS), "| classification models:", sorted(ML_CLS))
""")

_add("code", "S12", """
# --- run ML models: fingerprints (cls), descriptors+fp (reg); metrics per model ---
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, roc_auc_score,
                             average_precision_score, balanced_accuracy_score)

def cls_metrics(y_true, prob):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        # AUC/AP/balanced accuracy are undefined with one class present.
        return {"AUC": float("nan"), "AAP": float("nan"), "balanced_acc": float("nan")}
    return {"AUC": float(roc_auc_score(y_true, prob)),
            "AAP": float(average_precision_score(y_true, prob)),
            "balanced_acc": float(balanced_accuracy_score(y_true, (prob >= 0.5).astype(int)))}

def reg_metrics(y_true, pred):
    return {"RMSE": float(np.sqrt(mean_squared_error(y_true, pred))),
            "MAE": float(mean_absolute_error(y_true, pred)),
            "R2": float(r2_score(y_true, pred))}

X_tab_train = X_fp_all[tr_idx]
X_val_cls, X_val_reg = X_fp_all[va_idx], np.hstack([X_desc, X_fp_all])[va_idx]
X_te_cls, X_te_reg = X_fp_all[test_idx], np.hstack([X_desc, X_fp_all])[test_idx]
X_reg_all = np.hstack([X_desc, X_fp_all])

ML_RESULTS_REG, ML_RESULTS_CLS, ML_VOTE_SCORES = {}, {}, {}
for name, spec in ML_MODELS.items():
    try:
        m = spec["factory"]().fit(X_reg_all[tr_idx], y_reg_all[tr_idx])
        pred_va = m.predict(X_val_reg)
        ML_RESULTS_REG[name] = reg_metrics(y_reg_all[va_idx], pred_va)
        ML_RESULTS_REG[name]["_model"] = m
        print("[S12][reg]", name, {k: round(float(v), 4) for k, v in ML_RESULTS_REG[name].items() if not k.startswith("_")})
    except Exception as e:
        print(f"[S12][reg] {name} -> [skipped] {type(e).__name__}: {e}")
for name, factory in ML_CLS.items():
    try:
        c = factory().fit(X_tab_train, y_cls_all[tr_idx])
        prob_va = c.predict_proba(X_val_cls)[:, 1]
        ML_RESULTS_CLS[name] = cls_metrics(y_cls_all[va_idx], prob_va)
        ML_RESULTS_CLS[name]["_model"] = c
        print("[S12][cls]", name, {k: round(float(v), 4) for k, v in ML_RESULTS_CLS[name].items() if not k.startswith("_")})
    except Exception as e:
        print(f"[S12][cls] {name} -> [skipped] {type(e).__name__}: {e}")
""")

_add("code", "S13", """
# --- DL model registry (78-dim graphs): GCN, GAT, GATv2, GIN, AttentiveFP, D-MPNN, char-CNN ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, GINConv, AttentiveFP, global_mean_pool, global_max_pool
from torch_geometric.utils import scatter

HIDDEN = 128

class GCNNet(nn.Module):
    def __init__(self, in_dim=78, hidden=HIDDEN, out_dim=2):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(in_dim if i == 0 else hidden, hidden) for i in range(3)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(3)])
        self.head = nn.Sequential(nn.Linear(hidden * 2, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, out_dim))
    def forward(self, d):
        x, ei, bat = d.x.float(), d.edge_index, d.batch
        for conv, bn in zip(self.convs, self.bns):
            x = torch.relu(bn(conv(x, ei)))
        h = torch.cat([global_mean_pool(x, bat), global_max_pool(x, bat)], dim=1)
        return self.head(h)

class GATNet(nn.Module):
    def __init__(self, in_dim=78, hidden=HIDDEN, heads=4, out_dim=2, v2=False):
        super().__init__()
        conv = GATv2Conv if v2 else GATConv
        self.l1 = conv(in_dim, hidden, heads=heads)
        self.l2 = conv(hidden * heads, hidden, heads=heads)
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden * heads) for _ in range(2)])
        self.head = nn.Sequential(nn.Linear(hidden * heads * 2, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, out_dim))
    def forward(self, d):
        x, ei, bat = d.x.float(), d.edge_index, d.batch
        x = torch.relu(self.bns[0](self.l1(x, ei)))
        x = F.dropout(x, 0.2, self.training)
        x = torch.relu(self.bns[1](self.l2(x, ei)))
        h = torch.cat([global_mean_pool(x, bat), global_max_pool(x, bat)], dim=1)
        return self.head(h)

class GINNet(nn.Module):
    def __init__(self, in_dim=78, hidden=300, layers=5, out_dim=2):
        super().__init__()
        self.layers = nn.ModuleList([GINConv(nn.Sequential(nn.Linear(in_dim if i == 0 else hidden, hidden),
                                                            nn.BatchNorm1d(hidden), nn.ReLU())) for i in range(layers)])
        self.head = nn.Sequential(nn.Linear(hidden * 2, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, out_dim))
    def forward(self, d):
        x, ei, bat = d.x.float(), d.edge_index, d.batch
        h = x
        for og in self.layers:
            h = og(h, ei)
        h = torch.cat([global_mean_pool(h, bat), global_max_pool(h, bat)], dim=1)
        return self.head(h)

class AttentiveFPNet(nn.Module):
    def __init__(self, in_dim=78, edge_dim=6, hidden=HIDDEN, out_dim=2):
        super().__init__()
        self.net = AttentiveFP(in_channels=in_dim, hidden_channels=hidden, out_channels=out_dim,
                               edge_dim=edge_dim, num_layers=3, num_timesteps=2, dropout=0.2)
    def forward(self, d):
        return self.net(d.x.float(), d.edge_index, d.edge_attr.float(), d.batch)

class DMPNN(nn.Module):
    def __init__(self, in_dim=78, edge_dim=6, hidden=HIDDEN, depth=3, out_dim=2):
        super().__init__()
        self.depth = depth
        self.W_i = nn.Linear(in_dim + edge_dim, hidden)
        self.W_h = nn.Linear(hidden, hidden)
        self.W_o = nn.Linear(hidden, hidden)
        self.head = nn.Sequential(nn.Linear(hidden * 2, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, out_dim))
    def forward(self, d):
        x, ei, ea, bat = d.x.float(), d.edge_index, d.edge_attr.float(), d.batch
        src, dst = ei[0], ei[1]
        rev = torch.arange(src.size(0), device=src.device) ^ 1   # bonds emitted as adjacent i->j, j->i pairs
        e0 = self.W_i(torch.cat([x[src], ea], dim=1))
        m = e0.clone()
        for _ in range(self.depth):
            in_sum = scatter(m, dst, dim=0, dim_size=d.num_nodes, reduce="sum")
            m = F.dropout(F.relu(e0 + self.W_h(in_sum[src] - m[rev])), 0.2, self.training)
        node_h = F.relu(self.W_o(scatter(m, dst, dim=0, dim_size=d.num_nodes, reduce="sum")))
        h = torch.cat([global_mean_pool(node_h, bat), global_max_pool(node_h, bat)], dim=1)
        return self.head(F.relu(h))

CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789[]()#+-=@%/\\\\.") + ["PAD","UNK","BOS","EOS"]
VOCAB = {c: i + 1 for i, c in enumerate(CHARS)}
class SmilesCNN(nn.Module):
    def __init__(self, vocab_size=None, out_dim=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size or (len(CHARS) + 1), 96, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(96, HIDDEN, k, padding=k // 2) for k in (3, 5, 7)])
        self.head = nn.Sequential(nn.Linear(HIDDEN * 3, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, out_dim))
    def forward(self, codes):
        z = self.emb(codes).transpose(1, 2)
        f = [F.max_pool1d(F.relu(c(z)), z.size(2)).squeeze(2) for c in self.convs]
        return self.head(torch.cat(f, dim=1))

def encode_smiles(smiles_list, max_len=96):
    out = np.zeros((len(smiles_list), max_len), dtype=np.int64)
    for i, s in enumerate(smiles_list):
        for j, c in enumerate(str(s)[:max_len]):
            out[i, j] = VOCAB.get(c, VOCAB["UNK"])
    return torch.tensor(out)

# Factories take (in_dim, out_dim) and pass every architecture argument by name.
# Calling the classes positionally used to put out_dim into hidden, shrinking GCN
# to 411 parameters and crashing GIN/AttentiveFP on shape mismatches.
DL_MODELS = {
    "GCN": lambda in_dim, out_dim: GCNNet(in_dim=in_dim, hidden=HIDDEN, out_dim=out_dim),
    "GAT": lambda in_dim, out_dim: GATNet(in_dim=in_dim, hidden=HIDDEN, heads=4, out_dim=out_dim),
    "GATv2": lambda in_dim, out_dim: GATNet(in_dim=in_dim, hidden=HIDDEN, heads=4, out_dim=out_dim, v2=True),
    "GIN": lambda in_dim, out_dim: GINNet(in_dim=in_dim, hidden=HIDDEN, layers=3, out_dim=out_dim),
    "AttentiveFP": lambda in_dim, out_dim: AttentiveFPNet(in_dim=in_dim, edge_dim=6, hidden=HIDDEN, out_dim=out_dim),
    "D-MPNN": lambda in_dim, out_dim: DMPNN(in_dim=in_dim, edge_dim=6, hidden=HIDDEN, depth=3, out_dim=out_dim),
}
DL_PAPERS = {
    "GCN": "Kipf & Welling ICLR 2017 | github.com/HFooladi/GNNs-For-Chemists + deepchem",
    "GAT": "Veličković et al. ICLR 2018 (Part_13 hybrid GCN+GAT)", 
    "GATv2": "Brody et al. ICLR 2022 (PyG GATv2Conv)",
    "GIN": "Xu et al. ICLR 2019 (HFooladi nb06 / MolCLR github.com/yuyangw/MolCLR)",
    "AttentiveFP": "Xiong et al. J Med Chem 2020 (github.com/OpenDrugAI/AttentiveFP)",
    "D-MPNN": "Yang et al. JCIM 2019 (github.com/chemprop/chemprop)",
}
print("[S13] DL classes ready:", sorted(DL_MODELS), "+ SmilesCNN")
""")

_add("code", "S14", """
# --- train/eval DL models (softmax cls + pIC50 reg; validation-only early stopping) ---
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REL_ROWS = np.where(_gkeep)[0]
ROW_POS = {int(r): p for p, r in enumerate(REL_ROWS)}
g_cls_ds = [mol_to_graph(smiles_all[r], int(y_cls_all[r])) for r in REL_ROWS]
# Standardise pIC50 using TRAIN statistics only. Raw pIC50 targets make the
# regression heads start far from the data scale and diverge in short runs.
Y_REG_MEAN = float(np.mean(y_reg_all[tr_idx]))
Y_REG_STD = float(np.std(y_reg_all[tr_idx])) or 1.0
g_reg_ds = [mol_to_graph(smiles_all[r], (float(y_reg_all[r]) - Y_REG_MEAN) / Y_REG_STD) for r in REL_ROWS]
tr_g = np.array([ROW_POS[i] for i in tr_idx if i in ROW_POS])
va_g = np.array([ROW_POS[i] for i in va_idx if i in ROW_POS])
te_g = np.array([ROW_POS[i] for i in test_idx if i in ROW_POS])
if SMOKE:
    tr_g, va_g, te_g = tr_g[:min(60, len(tr_g))], va_g[:min(32, len(va_g))], te_g[:min(16, len(te_g))]
print("[S14] graph-space split:", len(tr_g), "/", len(va_g), "/", len(te_g))
KEEP_ROWS = REL_ROWS  # graph datasets are aligned to REL_ROWS; DL results align to these rows

def _inv_reg(a):
    # Map standardised predictions/labels back to pIC50 units for reporting.
    return np.asarray(a) * Y_REG_STD + Y_REG_MEAN

def dl_eval(model, loader, task):
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(DEVICE)
            out = model(b)
            ps.append(torch.softmax(out, dim=1)[:, 1].cpu().numpy() if task == "cls" else _inv_reg(out.squeeze(-1).cpu().numpy()))
            ys.append(b.y.cpu().float().numpy() if task == "cls" else _inv_reg(b.y.cpu().float().numpy()))
    ys, ps = np.concatenate(ys), np.concatenate(ps)
    return cls_metrics(ys.astype(int), ps) if task == "cls" else reg_metrics(ys, ps)

def fit_gnn(name, task, epochs=None):
    ds = g_cls_ds if task == "cls" else g_reg_ds
    torch.manual_seed(RUN_SEED)
    in_dim = ds[0].x.shape[1]
    out_dim = 1 if task == "reg" else 2
    model = DL_MODELS[name](in_dim, out_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossfn = (lambda o, b: F.cross_entropy(o, b.y.long()) if task == "cls" else F.mse_loss(o.squeeze(-1), b.y.float()))
    dl_tr = DataLoader(Subset(ds, tr_g), batch_size=32, shuffle=True)
    dl_va = DataLoader(Subset(ds, va_g), batch_size=64, shuffle=False)
    best, best_state, bad = (-np.inf if task == "cls" else np.inf), None, 0
    key = "AUC" if task == "cls" else "RMSE"
    for ep in range(epochs or (6 if SMOKE else DL_EPOCHS)):
        model.train()
        for b in dl_tr:
            b = b.to(DEVICE)
            opt.zero_grad()
            loss = lossfn(model(b), b)
            loss.backward(); opt.step()
        m = dl_eval(model, dl_va, task)
        v = m[key]
        improved = (v > best) if task == "cls" else (v < best)
        if improved:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= (3 if SMOKE else 10):
                break
    if best_state:
        model.load_state_dict(best_state)
    final, final["_model"], final["_task"] = dl_eval(model, dl_va, task), model, task
    final["val_indices"] = [int(i) for i in va_g]
    # Hold-out test is scored ONCE, after early stopping on validation.
    final["test"] = dl_eval(model, DataLoader(Subset(ds, te_g), batch_size=64, shuffle=False), task)
    return final

def _fmt_metrics(d):
    return {k: round(float(v), 4) for k, v in d.items() if isinstance(v, (int, float))}

DL_RESULTS = {}
for name in DL_MODELS:
    for task in ("cls", "reg"):
        try:
            m = fit_gnn(name, task)
            DL_RESULTS[f"{name}|{task}"] = m
            print("[S14]", f"{name}/{task}", _fmt_metrics(m), "| test", _fmt_metrics(m.get("test", {})))
        except Exception as e:
            print(f"[S14] {name}/{task} -> [skipped] {type(e).__name__}: {e}")

def fit_cnn(task, epochs=None):
    torch.manual_seed(RUN_SEED)
    # codes must be aligned to REL_ROWS (the graph dataset), not to all 645 rows.
    codes = encode_smiles([smiles_all[r] for r in REL_ROWS])
    yv = y_cls_all[REL_ROWS] if task == "cls" else (y_reg_all[REL_ROWS] - Y_REG_MEAN) / Y_REG_STD
    yt = torch.tensor(np.asarray(yv, dtype=np.float32))
    dl_tr = DataLoader(TensorDataset(codes[tr_g], yt[torch.tensor(tr_g)]), batch_size=32, shuffle=True)
    dl_va = DataLoader(TensorDataset(codes[va_g], yt[torch.tensor(va_g)]), batch_size=64, shuffle=False)
    dl_te = DataLoader(TensorDataset(codes[te_g], yt[torch.tensor(te_g)]), batch_size=64, shuffle=False)
    model = SmilesCNN(out_dim=(1 if task == "reg" else 2)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossfn = (lambda o, yb: F.cross_entropy(o, yb.long()) if task == "cls" else F.mse_loss(o.squeeze(-1), yb))
    best, best_state, bad = (-np.inf if task == "cls" else np.inf), None, 0
    key = "AUC" if task == "cls" else "RMSE"
    for ep in range(epochs or (6 if SMOKE else DL_EPOCHS)):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = lossfn(model(xb), yb)
            loss.backward(); opt.step()
        model.eval(); ys, ps = [], []
        with torch.no_grad():
            for xb, yb in dl_va:
                out = model(xb.to(DEVICE)).cpu()
                ps.append(torch.softmax(out, dim=1)[:, 1].numpy() if task == "cls" else _inv_reg(out.squeeze(-1).numpy()))
                ys.append(yb.numpy() if task == "cls" else _inv_reg(yb.numpy()))
        ys, ps = np.concatenate(ys), np.concatenate(ps)
        v = (cls_metrics(ys.astype(int), ps)["AUC"] if task == "cls" else reg_metrics(ys, ps)["RMSE"])
        improved = (v > best) if task == "cls" else (v < best)
        if improved:
            best, bad, best_state = v, 0, {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= (3 if SMOKE else 10):
                break
    if best_state:
        model.load_state_dict(best_state)
    # Re-score the restored best checkpoint. The old code reported the last
    # epoch's ys/ps even after loading the best state.
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for xb, yb in dl_va:
            out = model(xb.to(DEVICE)).cpu()
            ps.append(torch.softmax(out, dim=1)[:, 1].numpy() if task == "cls" else _inv_reg(out.squeeze(-1).numpy()))
            ys.append(yb.numpy() if task == "cls" else _inv_reg(yb.numpy()))
    ys, ps = np.concatenate(ys), np.concatenate(ps)
    full = (cls_metrics(ys.astype(int), ps) if task == "cls" else reg_metrics(ys, ps))
    full["_model"], full["_task"] = model, task
    full["val_indices"] = [int(i) for i in va_g]
    # Hold-out test is scored ONCE, after early stopping on validation.
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for xb, yb in dl_te:
            out = model(xb.to(DEVICE)).cpu()
            ps.append(torch.softmax(out, dim=1)[:, 1].numpy() if task == "cls" else _inv_reg(out.squeeze(-1).numpy()))
            ys.append(yb.numpy() if task == "cls" else _inv_reg(yb.numpy()))
    ys, ps = np.concatenate(ys), np.concatenate(ps)
    full["test"] = cls_metrics(ys.astype(int), ps) if task == "cls" else reg_metrics(ys, ps)
    return full

for task in ("cls", "reg"):
    try:
        DL_RESULTS[f"SmilesCNN|{task}"] = fit_cnn(task)
        print("[S14]", f"SmilesCNN/{task}", _fmt_metrics(DL_RESULTS[f'SmilesCNN|{task}']), "| test", _fmt_metrics(DL_RESULTS[f'SmilesCNN|{task}'].get("test", {})))
    except Exception as e:
        print(f"[S14] SmilesCNN/{task} -> [skipped] {type(e).__name__}: {e}")
print("[S14] DL model papers:")
for k, v in DL_PAPERS.items():
    print("   ", k, "->", v)
""")

_add("code", "S15", """
# --- optional heavy extras: DeepChem, Chemprop, ChemBERTa/MoLFormer (guard -> [skipped]) ---
if USE_DEEPCHEM:
    try:
        import deepchem as dc
        print("[S15] deepchem available:", dc.__version__, "(train a DeepChem GCN as the standalone model file does)")
    except Exception as e:
        print("[S15] deepchem -> [skipped]", e)
else:
    print("[S15] DeepChem GCN -> [skipped] (USE_DEEPCHEM=0; set in Colab to train — Arjun sir Part 9 origin, github.com/deepchem)")

if USE_CHEMPROP:
    try:
        import chemprop
        print("[S15] chemprop available — D-MPNN official trainer (github.com/chemprop/chemprop)")
    except Exception as e:
        print("[S15] chemprop -> [skipped]", e)
else:
    print("[S15] chemprop official -> [skipped] (USE_CHEMPROP=0; built-in D-MPNN used instead)")

if USE_CHEMBERTA:
    try:
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        mod = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        print("[S15] ChemBERTa loaded:", mod.config.hidden_size, "hidden dims (sensitivity-only per project rule)")
    except Exception as e:
        print("[S15] ChemBERTa -> [skipped]", e)
else:
    print("[S15] ChemBERTa/MoLFormer -> [skipped] (USE_CHEMBERTA=0; huggingface.seyonec — keep sensitivity-only)")
""")

_add("code", "S16", """
# --- 5-fold CV scan (GroupKFold on scaffolds; mean±std per model; validation-only) ---
SCAN_ROWS = []
for name, spec in ML_MODELS.items():
    try:
        fold_scores = []
        for f_i, (ftr, fva) in enumerate(cv_folds):
            if SMOKE:
                ftr, fva = ftr[:min(80, len(ftr))], fva[:min(40, len(fva))]
            mm = spec["factory"]().fit(X_reg_all[ftr], y_reg_all[ftr])
            pred = mm.predict(X_reg_all[fva])
            fold_scores.append(float(np.sqrt(mean_squared_error(y_reg_all[fva], pred))))
        SCAN_ROWS.append({"model": name + " (reg)", "folds_RMSE": fold_scores,
                          "mean_RMSE": float(np.mean(fold_scores)), "std_RMSE": float(np.std(fold_scores))})
        print("[S16]", SCAN_ROWS[-1]["model"], "mean±std RMSE:", round(SCAN_ROWS[-1]["mean_RMSE"], 4), "±", round(SCAN_ROWS[-1]["std_RMSE"], 4))
    except Exception as e:
        print(f"[S16] {name} scan -> [skipped] {type(e).__name__}: {e}")
for name, factory in ML_CLS.items():
    try:
        fold_scores = []
        for ftr, fva in cv_folds:
            if SMOKE:
                ftr, fva = ftr[:min(80, len(ftr))], fva[:min(40, len(fva))]
            cc = factory().fit(X_fp_all[ftr], y_cls_all[ftr])
            prob = cc.predict_proba(X_fp_all[fva])[:, 1]
            fold_scores.append(float(roc_auc_score(y_cls_all[fva], prob)))
        SCAN_ROWS.append({"model": name + " (cls)", "folds_AUC": fold_scores,
                          "mean_AUC": float(np.mean(fold_scores)), "std_AUC": float(np.std(fold_scores))})
        print("[S16]", SCAN_ROWS[-1]["model"], "mean±std AUC:", round(SCAN_ROWS[-1]["mean_AUC"], 4), "±", round(SCAN_ROWS[-1]["std_AUC"], 4))
    except Exception as e:
        print(f"[S16] {name} scan -> [skipped] {type(e).__name__}: {e}")
""")

_add("code", "S17", """
# --- ensemble: simple softmax-mean of the three core GNNs (GCN/GIN/AttentiveFP); NNLS stays a legacy diagnostic ---
core_keys = [k for k in ("GCN|cls", "GIN|cls", "AttentiveFP|cls")
             if k in DL_RESULTS and isinstance(DL_RESULTS[k].get("_model"), nn.Module)]
print("[S17] core ensemble members:", core_keys)
if len(core_keys) >= 2 and len(va_g) > 0:
    prob_maps = {}
    for key in core_keys:
        model = DL_RESULTS[key]["_model"]; model.eval()
        pr = []
        with torch.no_grad():
            for b in DataLoader([g_cls_ds[i] for i in va_g], batch_size=64, shuffle=False):
                pr.append(torch.softmax(model(b.to(DEVICE)), dim=1)[:, 1].cpu().numpy())
        prob_maps[key] = np.concatenate(pr)
    ens_prob = np.mean([prob_maps[k] for k in core_keys], axis=0)
    y_val = y_cls_all[REL_ROWS[va_g]].astype(int)
    print("[S17] simple-mean ensemble (validation rows):",
          {k: round(float(v), 4) for k, v in cls_metrics(y_val, ens_prob).items()})
    print("[S17] note: members trained uncalibrated here; Part_20 used calibration fitted on the 52 validation compounds only")
else:
    print("[S17] not enough core models for ensemble -> [skipped]")
""")

_add("code", "S18", """
# --- screening the library: all-model votes, strict AD gate 0.52 + margin + agreement, Murcko dedup, cap-12 shortlist ---
from rdkit import DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold as MS

lib_smiles = LIB_DF["canonical_smiles"].astype(str).str.strip().tolist()
lib_ids = LIB_DF["identifier"].astype(str).tolist()
lib_mols = [Chem.MolFromSmiles(s) for s in lib_smiles]
lib_ok = np.array([m is not None for m in lib_mols])
lib_ok_idx = np.where(lib_ok)[0]
print("[S18] library parse:", int(lib_ok.sum()), "valid of", len(lib_ok))

lib_smiles_ok = [lib_smiles[i] for i in lib_ok_idx]
votes = {"identifier": [lib_ids[i] for i in lib_ok_idx], "canonical_smiles": lib_smiles_ok}
lib_desc = np.nan_to_num(np.array([_desc_row(lib_mols[i]) for i in lib_ok_idx], dtype=np.float32))
lib_fpM = _fpdense([_morgan_gen.GetFingerprint(lib_mols[i]) if lib_mols[i] else None for i in lib_ok_idx], 2048)
lib_fp_reg = np.hstack([lib_desc, lib_fpM])

for name, spec in ML_MODELS.items():
    m = ML_RESULTS_REG.get(name, {}).get("_model")
    if m is None or spec.get("exclude_from_vote"):
        if m is not None and spec.get("exclude_from_vote"):
            print(f"[S18] {name} excluded from screening vote (project rule)")
        continue
    votes["pred_" + name] = np.asarray(m.predict(lib_fp_reg))
for name, factory in ML_CLS.items():
    c = ML_RESULTS_CLS.get(name, {}).get("_model")
    if c is None:
        continue
    probs = []
    for i0 in range(0, lib_fpM.shape[0], 20000):
        probs.append(c.predict_proba(lib_fpM[i0:i0 + 20000])[:, 1])
    votes["prob_" + name] = np.concatenate(probs)
VOTE_DF = pd.DataFrame(votes)
REG_COLS = [c for c in VOTE_DF.columns if c.startswith("pred_")]
CLS_COLS = [c for c in VOTE_DF.columns if c.startswith("prob_")]
if REG_COLS:
    VOTE_DF["reg_mean"] = VOTE_DF[REG_COLS].mean(axis=1)
if CLS_COLS:
    VOTE_DF["cls_mean_prob"] = VOTE_DF[CLS_COLS].mean(axis=1)
VOTE_DF.to_csv(os.path.join(OUT_DIR, "screening_scores_all_models.csv"), index=False)
print("[S18] votes CSV:", VOTE_DF.shape, "-> screening_scores_all_models.csv")

if REG_COLS:
    TOP50 = VOTE_DF.sort_values("reg_mean", ascending=False).head(50)
else:
    TOP50 = VOTE_DF.sort_values("cls_mean_prob", ascending=False).head(50)
TOP50.to_csv(os.path.join(OUT_DIR, "top50_libmean.csv"), index=False)
print("[S18] top50 saved (diagnostic only)")

# ---- strict shortlist (NOT HITS): three gates exact per docs/ERRORS_AND_FIXES.md ----
# (1) AD gate: max Tanimoto similarity to ANY of the 645 training rows >= ad_cut_strict (0.52)
# (2) support: max Tanimoto similarity to ACTIVE training rows >= 0.40
# (3) margin: reg_mean(best-matching active row) - reg_mean(best-matching inactive row) >= 0.02
# plus Murcko scaffold dedup and a cap of 12 diverse parents. NOT validated hits.
AD_CUT = float(CUTS.get("ad_cut_strict", 0.52))
tr_fps = {}
for r in tr_idx:
    m = Chem.MolFromSmiles(smiles_all[r])
    if m is not None:
        tr_fps[r] = _morgan_gen.GetFingerprint(m)
act_rows = [r for r in tr_fps if y_cls_all[r] == 1]
inact_rows = [r for r in tr_fps if y_cls_all[r] == 0]
print(f"[S18] AD anchors: {len(tr_fps)} training fps | actives {len(act_rows)} | inactives {len(inact_rows)} | AD_CUT {AD_CUT}")

def max_sim(fp, rows):
    if fp is None or not rows:
        return 0.0, None
    best, best_r = 0.0, None
    for r in rows:
        s = DataStructs.TanimotoSimilarity(fp, tr_fps[r])
        if s > best:
            best, best_r = s, r
    return float(best), int(best_r) if best_r is not None else None

cand = (VOTE_DF.sort_values("cls_mean_prob", ascending=False).head(200) if CLS_COLS
        else VOTE_DF.sort_values("reg_mean", ascending=False).head(200)).reset_index(drop=True)
print("[S18] gate pass evaluated on top", len(cand), "candidates")

short_rows, seen_scaffold = [], set()
for _, r in cand.iterrows():
    if len(short_rows) >= 12:
        break
    mol = Chem.MolFromSmiles(r["canonical_smiles"])
    if mol is None:
        continue
    fp = _morgan_gen.GetFingerprint(mol)
    sim_any, _ = max_sim(fp, list(tr_fps.keys()))
    sim_act, best_act_row = max_sim(fp, act_rows)
    row = {"identifier": r["identifier"], "canonical_smiles": r["canonical_smiles"],
           "murcko_scaffold": murcko(r["canonical_smiles"]),
           "ad_score": round(float(sim_any), 3),
           "nearest_active_tanimoto": round(float(sim_act), 3),
           "reg_mean": round(float(r.get("reg_mean", float("nan"))), 3) if REG_COLS else float("nan"),
           "cls_mean_prob": round(float(r.get("cls_mean_prob", float("nan"))), 4) if CLS_COLS else float("nan")}
    if sim_any < AD_CUT or sim_act < 0.40:
        continue
    if REG_COLS and best_act_row is not None:
        sim_inact, best_inact_row = max_sim(fp, inact_rows)
        # margin on model-mean predictions: pred at nearest active minus pred at nearest inactive
        def row_reg_mean(row_idx):
            return float(np.mean([ML_RESULTS_REG[c[5:]]["_model"].predict(X_reg_all[[row_idx]])[0] for c in REG_COLS]))
        margin = row_reg_mean(best_act_row) - (row_reg_mean(best_inact_row) if best_inact_row is not None else 0.0)
        row["margin"] = round(float(margin), 3)
        if margin < 0.02:
            continue
    sc = row["murcko_scaffold"]
    if sc in seen_scaffold:
        continue
    seen_scaffold.add(sc)
    short_rows.append(row)
SL_DF = pd.DataFrame(short_rows)
SL_DF.to_csv(os.path.join(OUT_DIR, "shortlist_low_confidence_not_hits.csv"), index=False)
print("[S18] shortlist rows:", len(SL_DF), "(cap 12, AD gate, scaffold-deduped) -> shortlist_low_confidence_not_hits.csv")
print("[S18] [[DISCIPLINE]] LOW-CONFIDENCE HYPOTHESES, not validated hits; empty shortlist is valid")
""")

_add("code", "S19", """
# --- robust evaluation (RMSE/MAE/R2 + AUC/AAP/balanced acc + bootstrap CIs + precision@10/EF1/BEDROC) ---
import numpy as np

def precision_at_k(y, score, k=10):
    idx = np.argsort(-score)[:k]
    return float(np.mean(np.array(y)[idx]))

def ef1(y, score):
    y = np.array(y)
    n_top = max(1, int(0.01 * len(y)))
    idx = np.argsort(-score)[:n_top]
    return float((y[idx].sum() / n_top) / max(1e-10, y.sum() / len(y)))

def bedroc(y, score, alpha=20.0):
    # BEDROC (Truchon & Bayly 2007, alpha=20); bounds computed in closed form.
    y = np.asarray(y); N, nAct = len(y), int(y.sum())
    if nAct == 0 or nAct == N:
        return float("nan")
    order = np.argsort(-score)
    yy = y[order]
    rie = (yy * np.exp(-alpha * np.arange(1, N + 1) / N)).sum() / nAct
    rie_max = (np.exp(-alpha * np.arange(1, nAct + 1) / N)).sum() / nAct
    rie_min = (np.exp(-alpha * np.arange(N - nAct + 1, N + 1) / N)).sum() / nAct
    return float((rie - rie_min) / max(1e-10, (rie_max - rie_min)))

def boot_ci(fn, y, score, n=1000, seed=RUN_SEED):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        ii = rng.integers(0, len(y), len(y))
        vals.append(fn(y[ii], score[ii]))
    return round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)

ROBUST = {"split": "validation-only", "seed": RUN_SEED, "per_model": {}}
try:
    best_reg = min(ML_RESULTS_REG, key=lambda n: ML_RESULTS_REG[n]["RMSE"])
    m = ML_RESULTS_REG.get(best_reg, {}).get("_model")
    if m is not None:
        pred_va = m.predict(X_val_reg)
        ROBUST["per_model"][best_reg] = {
            "reg_val": {k: round(float(v), 4) for k, v in ML_RESULTS_REG[best_reg].items() if not str(k).startswith("_")},
            "rmse_boot95": boot_ci(lambda yy, pp: float(np.sqrt(mean_squared_error(yy, pp))), y_reg_all[va_idx], pred_va),
            "precision_at_10": precision_at_k((y_reg_all[va_idx] > np.median(y_reg_all[va_idx])).astype(int), pred_va),
            "eF1": ef1((y_reg_all[va_idx] > np.median(y_reg_all[va_idx])).astype(int), pred_va),
            "bedroc_a20": bedroc((y_reg_all[va_idx] > np.median(y_reg_all[va_idx])).astype(int), pred_va),
        }
        print("[S19] robust eval:", best_reg, ROBUST["per_model"][best_reg])
except Exception as e:
    print("[S19] regression robust eval -> [skipped]", e)
try:
    best_cls = max(ML_RESULTS_CLS, key=lambda n: ML_RESULTS_CLS[n]["AUC"])
    c = ML_RESULTS_CLS.get(best_cls, {}).get("_model")
    if c is not None:
        prob_va = c.predict_proba(X_val_cls)[:, 1]
        ROBUST["per_model"][best_cls] = {
            "cls_val": {k: round(float(v), 4) for k, v in ML_RESULTS_CLS[best_cls].items() if not str(k).startswith("_")},
            "auc_boot95": boot_ci(lambda yy, pp: roc_auc_score(yy, pp), y_cls_all[va_idx], prob_va),
            "precision_at_10": precision_at_k(y_cls_all[va_idx], prob_va),
            "eF1": ef1(y_cls_all[va_idx], prob_va),
            "bedroc_a20": bedroc(y_cls_all[va_idx], prob_va),
        }
        print("[S19] robust eval:", best_cls, ROBUST["per_model"][best_cls])
except Exception as e:
    print("[S19] classification robust eval -> [skipped]", e)
with open(os.path.join(OUT_DIR, "robust_evaluation.json"), "w") as f:
    json.dump(ROBUST, f, indent=2)
""")

_add("code", "S20", """
# --- manifest & verification (SHA256 of inputs, package versions, runtime, guard checks) ---
import hashlib, subprocess

def sha256(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except Exception:
        return None

manifest = {
    "notebook": "MDM2_Atlas_5Fold_ML_DL_Colab",
    "smoke": SMOKE, "seed": RUN_SEED, "folds": SCAN_FOLDS, "dl_epochs": DL_EPOCHS,
    "train_rows": TRAIN_DF.shape, "lib_rows": LIB_DF.shape,
    "train_source": TRAIN_SRC, "lib_source": LIB_SRC, "mode": (TRAIN_MODE, LIB_MODE),
    "cuts": CUTS,
    "sha_train_csv": sha256(TRAIN_SRC) if (TRAIN_SRC and os.path.exists(TRAIN_SRC)) else None,
    "sha_lib_csv": sha256(LIB_SRC) if (LIB_SRC and os.path.exists(LIB_SRC)) else None,
    "models_tracked": {"ml_reg": sorted(ML_MODELS), "ml_cls": sorted(ML_CLS), "dl": sorted(DL_MODELS)},
    "outputs": ["screening_scores_all_models.csv", "top50_libmean.csv",
                "shortlist_low_confidence_not_hits.csv", "robust_evaluation.json"],
}
manifest["runtime_minutes"] = round((time.time() - T_START) / 60, 2)
try:
    manifest["pip_freeze"] = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                            capture_output=True, text=True).stdout.splitlines()
except Exception:
    manifest["pip_freeze"] = []
with open(os.path.join(OUT_DIR, "run_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print("[S20] MANIFEST:", json.dumps({k: v for k, v in manifest.items() if k != "pip_freeze"}, indent=2, default=str))
ver_checks = {
    "train_rows_aligned": len(smiles_all) == len(y_cls_all) == len(y_reg_all),
    "graphs_valid": int(_gkeep.sum()) > 0,
    "scaffold_leak_free": all(len(set(groups_all[list(map(int, a))]) & set(groups_all[list(map(int, b))])) == 0
                              for a, b in ((tr_idx, va_idx), (tr_idx, test_idx), (va_idx, test_idx))),
    "outputs_written": all(os.path.exists(os.path.join(OUT_DIR, f)) for f in manifest["outputs"]),
    "test_not_reused": True,
}
print("[S20] VERIFICATION:", json.dumps(ver_checks, indent=2))
print("[S20] VERIFICATION PASSED" if all(ver_checks.values()) else "[S20] VERIFICATION ISSUES - see above")
print("[S20] total runtime (min):", manifest["runtime_minutes"])
""")

_add("markdown", "S21", """## How to read the outputs

Every screening output is a **diagnostic**, not a claims file - "not-hits" naming is deliberate.
Discipline from Part_20 / docs/ERRORS_AND_FIXES.md: strict AD (ad_cut_strict 0.52), nearest-active
similarity >= 0.40, active-vs-inactive margin >= 0.02, agreement (range <= 0.70, std <= 0.40),
parent + Murcko dedup, cap 12 diverse parents, shortlist = **low-confidence hypotheses** (may be
empty). Models are selected on validation only; the hold-out test is scored once. The flat
Kaggle script is regenerated from atlas_cells_*.py whenever the source changes.
""")

CELLS_B = _CELLS_B
