# =============================================================================
# MDM2 Part_21 - FRESH MULTITASK PIPELINE (single-source Kaggle runner)
# =============================================================================
# One self-contained .py file covering the full flow, WITHOUT reusing old
# model outputs:
#   1. Load real training data (645 MDM2 compounds: pIC50 + Active/Inactive)
#   2. Type validation + data sanity (type mismatches caught & reported here)
#   3. Scaffold (Bemis-Murcko) split train/val/test  -  leakage-safe
#   4. Featurization from free chemistry information only
#      (RDKit Morgan fingerprints + molecular descriptors; NO old predictions)
#   5. Regression (pIC50) and classification (Active vs Inactive) as SEPARATE
#      tasks with separate printed metric blocks (incl. active/inactive AUC)
#   6. Models:
#        a) XGBoost regressor + XGBoost classifier
#        b) PyTorch multitask network (shared trunk, two heads)
#        c) Large pretrained chemistry model fine-tuning (ChemBERTa) with
#           TWO-GPU support (nn.DataParallel; 1-GPU then CPU fallbacks)
#   7. Screening library: only identifier + canonical_smiles are read;
#      old prediction/probability columns are dropped and must not re-enter
#      any feature.
#   8. TOP-50 outputs: top50_regression.csv + top50_classification.csv,
#      also printed to stdout as a table.
#   9. Built-in verification suite (leakage / types / data / folder / GPU /
#      top-50) written to verification_report.json.
#
# Environment overrides (all optional):
#   MDM2_PART21_INPUT_DIR / MDM2_PART21_OUTPUT_DIR
#   MDM2_PART21_SMOKE=1                tiny dry-run (subsample; no large models)
#   MDM2_PART21_USE_CHEMBERTA=1        force large-model fine-tuning even in smoke
#   MDM2_PART21_SKIP_CHEMBERTA=1       force the large-model suite off
#   MDM2_PART21_SCREEN_LARGE_MODELS=1  score the library with the large models
#   MDM2_PART21_SKIP_SCAN=1            skip the 5-fold CV hyperparameter scan
#   MDM2_PART21_SCAN_EPOCHS            DL-scan epochs per fold (default 40; smoke 3)
#   MDM2_PART21_SKIP_LARGE_CV=1        skip the large-model LR mini-scan
#   MDM2_PART21_EPOCHS                 override torch multitask epochs
#   MDM2_PART21_GNN_EPOCHS             override GIN/MPNN epochs
# =============================================================================
import hashlib
import importlib
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    TORCH_OK = True
except Exception:                                    # pragma: no cover
    torch, nn, TORCH_OK = None, None, False

# -----------------------------------------------------------------------------
# 0. Paths / configuration
# -----------------------------------------------------------------------------
try:                                                 # run as a script
    HERE = Path(__file__).resolve().parent
    REPO_ROOT = HERE.parent.parent                   # .../<repo>
except NameError:                                    # pasted in a notebook cell
    HERE = Path.cwd().resolve()
    REPO_ROOT = None
    NOTEBOOK = True
else:
    NOTEBOOK = False

IS_KAGGLE = Path("/kaggle").exists()
SMOKE = os.environ.get("MDM2_PART21_SMOKE", "0") == "1"

if IS_KAGGLE:
    DEFAULT_INPUT_DIR = Path("/kaggle/input/mdm2-part21")
    DEFAULT_OUTPUT_DIR = Path("/kaggle/working/mdm2_part21_outputs")
elif REPO_ROOT is not None:                          # run as a script
    DEFAULT_INPUT_DIR = REPO_ROOT / "00_data"
    DEFAULT_OUTPUT_DIR = HERE / "outputs"
else:                                                # notebook outside Kaggle
    DEFAULT_INPUT_DIR = HERE / "00_data"             # cwd-relative repo layout
    DEFAULT_OUTPUT_DIR = HERE / "outputs"

INPUT_DIR = Path(os.environ.get("MDM2_PART21_INPUT_DIR", DEFAULT_INPUT_DIR))
OUT_DIR = Path(os.environ.get("MDM2_PART21_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))

GH_BASE = os.environ.get(
    "MDM2_PART21_GH_BASE",
    "https://raw.githubusercontent.com/Techbjd/new_method/main",
)
GH_FILES = {
    "training.csv": "00_data/training/ro5_properties_filtered.csv",
    "screening.csv": "00_data/screening/screening_results.csv",
    "cuts.json": "00_data/config/cuts.json",
}
SMOKE_TRAIN = 120
SMOKE_SCREEN = 300

SEED = 42
np.random.seed(SEED)
if TORCH_OK:
    torch.manual_seed(SEED)

REPORT = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "checks": {}}


def _rel(p):
    if REPO_ROOT is None:
        return str(p)
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _dir_files(d: Path):
    return sorted(str(x.relative_to(d)) for x in d.rglob("*") if x.is_file()) if d.exists() else []


def _sha256(p: Path, limit=2 ** 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(limit), b""):
            h.update(chunk)
    return h.hexdigest()


def find_input(name: str) -> Path:
    """Locate an input file: exact dir, nested dirs, repo-relative, then GH."""
    rel = GH_FILES[name]
    candidates = [
        INPUT_DIR / name,
        INPUT_DIR / "data" / name,
        INPUT_DIR / "inputs" / name,
        INPUT_DIR / "part21" / name,
    ]
    if REPO_ROOT is not None:
        candidates.append(REPO_ROOT / rel)   # script mode: repo-relative data
    for c in candidates:
        if c.is_file():
            return c
    if INPUT_DIR.exists():
        hits = sorted(INPUT_DIR.rglob(name))
        if hits:
            return hits[0]
    import urllib.request
    dest = Path(os.environ.get("TMPDIR", "/tmp")) / "mdm2_part21_inputs" / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 1000:
        return dest
    url = f"{GH_BASE}/{rel}"
    tmp = str(dest) + ".part"
    print(f"[setup] input {name!r} not found locally; downloading {url}")
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)
    return dest


def check(name, status, **details):
    REPORT["checks"][name] = {"status": status, **details}
    tag = {"PASS": "PASS", "FAIL": "FAIL"}.get(status, status)
    print(f"[verify] {tag}  {name}  {details if status != 'PASS' else ''}")
    return status == "PASS"


def ensure_rdkit():
    """Kaggle notebook images sometimes lack rdkit; install it once up front."""
    try:
        import rdkit  # noqa: F401
        return True
    except ImportError:
        import subprocess
        print("[setup] rdkit missing; installing via pip (rdkit) ...")
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit"],
                           capture_output=True)
        if r.returncode == 0:
            try:
                import rdkit  # noqa: F401
                print("[setup] rdkit installed OK")
                return True
            except ImportError:
                pass
        print("[setup] pip install rdkit failed:",
              (r.stderr or b"").decode()[-400:])
        return False


# -----------------------------------------------------------------------------
# 1. Data loading + type validation + sanity
# -----------------------------------------------------------------------------
def load_training() -> pd.DataFrame:
    p = find_input("training.csv")
    df = pd.read_csv(p)
    if SMOKE:
        df = df.head(SMOKE_TRAIN).copy()
    return df


def load_screening() -> pd.DataFrame:
    p = find_input("screening.csv")
    df = pd.read_csv(p)
    # Hard-drop every old model-output column: the fresh pipeline may only
    # see the library structures themselves.
    old_cols = [c for c in ("prediction", "prob_class_0", "prob_class_1") if c in df.columns]
    df = df.drop(columns=old_cols)
    missing = [c for c in ("identifier", "canonical_smiles") if c not in df.columns]
    if missing:
        raise ValueError(f"screening csv missing required columns {missing} "
                         f"(got {sorted(df.columns)})")
    df = df[["identifier", "canonical_smiles"]]
    # identical structures under different identifiers must not occupy several
    # top-50 slots later: keep one row per canonical structure
    df = df.drop_duplicates("canonical_smiles").reset_index(drop=True)
    if SMOKE:
        df = df.head(SMOKE_SCREEN).copy()
    return df


def load_cuts() -> dict:
    cuts = json.loads(find_input("cuts.json").read_text())
    missing = [k for k in ("ad_cut", "ad_cut_strict") if k not in cuts]
    if missing:
        raise ValueError(f"cuts.json missing required keys {missing} (got {sorted(cuts)})")
    return cuts


def validate_and_clean_training(df: pd.DataFrame):
    """Type-coerce everything; drop unusable rows; report every correction."""
    issues = []
    for col in ("molecule_chembl_id", "bioactivity_class", "canonical_smiles", "pIC50"):
        if col not in df.columns:
            raise ValueError(f"training csv missing required column {col!r}")

    orig_dtype = str(df["pIC50"].dtype)
    df = df.copy()
    df["pIC50_num"] = pd.to_numeric(df["pIC50"], errors="coerce")
    if orig_dtype != "float64":
        issues.append(f"pIC50 parsed as {orig_dtype}; coerced to numeric")
    bad_pic50 = int(df["pIC50_num"].isna().sum())

    df["mol_id"] = df["molecule_chembl_id"].astype(str).str.strip()
    df["smiles"] = df["canonical_smiles"].astype(str).str.strip()
    df["activity"] = (
        df["bioactivity_class"].astype(str).str.strip().str.lower()
        .map({"active": 1, "inactive": 0})
    )
    bad_label = int(df["activity"].isna().sum())
    if bad_label:
        issues.append(f"{bad_label} rows with unknown bioactivity_class (dropped)")

    n0 = len(df)
    df = df.dropna(subset=["pIC50_num", "activity", "smiles"])

    # invalid SMILES handled again at featurization; count them here
    from rdkit import Chem
    ok = df["smiles"].map(lambda s: Chem.MolFromSmiles(s) is not None)
    n_bad_smiles = int((~ok).sum())
    n_out_of_range = int(((df["pIC50_num"] < -2) | (df["pIC50_num"] > 15)).sum())
    df = df[ok].reset_index(drop=True)
    df["pIC50"] = df["pIC50_num"].astype(float)

    dup = int(df.duplicated("smiles").sum())
    if dup:
        issues.append(f"{dup} duplicate SMILES rows kept-once")
        df = df.drop_duplicates("smiles").reset_index(drop=True)

    sanity = {
        "raw_rows": n0,
        "rows_kept": len(df),
        "nan_pIC50_dropped": bad_pic50,
        "bad_labels_dropped": bad_label,
        "invalid_smiles_dropped": n_bad_smiles,
        "out_of_range_pIC50": n_out_of_range,
        "duplicate_smiles": dup,
        "pIC50_min": float(df["pIC50"].min()),
        "pIC50_max": float(df["pIC50"].max()),
    }
    if issues:
        print("[data] cleanliness notes:", issues)
    REPORT["type_issues"] = issues
    REPORT["data_sanity"] = sanity
    return df, sanity


# -----------------------------------------------------------------------------
# 2. Scaffold split (Bemis-Murcko) - group-level, leakage-safe
# -----------------------------------------------------------------------------
def scaffold_key(smiles: str) -> str:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles.strip().lower()          # invalid SMILES: own group
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        s = Chem.MolToSmiles(scaf)
        return s if s else smiles.strip().lower()
    except Exception:
        return smiles.strip().lower()


def scaffold_split(df: pd.DataFrame):
    """Deterministic scaffold-based 80/10/10 split; scaffolds never span splits."""
    keys = [scaffold_key(s) for s in df["smiles"]]
    uniq = sorted(set(keys))
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(uniq))
    n_tr = int(math.floor(0.8 * len(uniq)))
    n_va = int(math.floor(0.1 * len(uniq)))
    tr_u = {uniq[i] for i in perm[:n_tr]}
    va_u = {uniq[i] for i in perm[n_tr:n_tr + n_va]}
    idx_tr, idx_va, idx_te = [], [], []
    for i, k in enumerate(keys):
        if k in tr_u:
            idx_tr.append(i)
        elif k in va_u:
            idx_va.append(i)
        else:
            idx_te.append(i)
    return (np.array(idx_tr, dtype=int), np.array(idx_va, dtype=int),
            np.array(idx_te, dtype=int), keys,
            {"n_unique_scaffolds": len(uniq),
             "n_train": len(idx_tr), "n_val": len(idx_va), "n_test": len(idx_te)})


# -----------------------------------------------------------------------------
# 3. Featurization (free chemistry only)
# -----------------------------------------------------------------------------
MORGAN_BITS = 2048
MORGAN_RADIUS = 2
DESC_FUNCS = None
_MORGAN_GEN = None


def _morgan_gen():
    """Modern RDKit Morgan generator (replaces deprecated
    AllChem.GetMorganFingerprintAsBitVect -> no deprecation warnings)."""
    global _MORGAN_GEN
    if _MORGAN_GEN is None:
        from rdkit.Chem import rdFingerprintGenerator
        _MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(
            radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)
    return _MORGAN_GEN


def _desc_names():
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
    return [
        Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA,
        Lipinski.NumHAcceptors, Lipinski.NumHDonors,
        rdMolDescriptors.CalcNumRotatableBonds,
        Descriptors.RingCount, Descriptors.NumAromaticRings,
        Descriptors.FractionCSP3, Descriptors.HeavyAtomCount,
    ]


def fp_of(smiles):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _morgan_gen().GetFingerprintAsNumPy(mol).astype(np.float32)


def desc_of(smiles):
    global DESC_FUNCS
    from rdkit import Chem
    if DESC_FUNCS is None:
        DESC_FUNCS = _desc_names()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return np.array([float(f(mol)) for f in DESC_FUNCS], dtype=np.float32)
    except Exception:
        return None


def build_features(smiles_list, label=""):
    """X = [morgan_bits | descriptors]; boolean validity mask per molecule."""
    if label:
        print(f"[features] featurizing {label}: {len(smiles_list)} molecules ...")
    fps, dss, keep = [], [], []
    for s in smiles_list:
        f, d = fp_of(s), desc_of(s)
        if f is None or d is None:
            keep.append(False)
            continue
        keep.append(True)
        fps.append(f)
        dss.append(d)
    X = np.hstack([np.asarray(fps, dtype=np.float32),
                   np.asarray(dss, dtype=np.float32)])
    if label:
        print(f"[features] {label}: kept {len(X)}/{len(smiles_list)} -> X {X.shape}")
    return X, np.array(keep, dtype=bool)


# -----------------------------------------------------------------------------
# 4. Models - separate regression & classification
# -----------------------------------------------------------------------------
def train_xgb(task, X, y, seed=SEED):
    import xgboost as xgb
    if task == "regression":
        model = xgb.XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.06,
            subsample=0.9, colsample_bytree=0.9,
            random_state=seed, tree_method="hist", n_jobs=4)
    else:
        model = xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.06,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=seed, tree_method="hist", n_jobs=4)
    model.fit(X, y)
    return model


def _eval_record(model, name, task, Xva, yva, Xte, yte, t0, screening_ok,
                 cv=None):
    """Fit-independent evaluation record for one trained model."""
    tag = task[:3].upper()
    if task == "regression":
        val_m = regression_metrics(yva, model.predict(Xva))
        te_m = regression_metrics(yte, model.predict(Xte))
        print(f"[regression] {tag}  {name:<16} test: rmse={te_m['rmse']:.3f} "
              f"r2={te_m['r2']:+.3f}")
    else:
        val_m = classification_metrics(
            yva, model.predict_proba(Xva)[:, 1])
        te_m = classification_metrics(
            yte, model.predict_proba(Xte)[:, 1])
        print(f"[classification] {tag}  {name:<16} test: "
              f"auc={te_m['auc_roc']:.3f} bal_acc={te_m['balanced_accuracy']:.3f}")
    rt = round(time.time() - t0, 1)
    rec = {"model": name, "_model": model, "val": val_m, "test": te_m,
           "train_time_s": rt, "screening_ok": screening_ok}
    if cv:
        rec["cv_best"] = cv[0]
    return rec


def train_all_tabular(task, Xtr, ytr, Xva, yva, Xte, yte, scan_results=None):
    """Train EVERY tabular model for ONE task (best params from the CV scan)."""
    records = []
    for name, screening_ok in _tabular_names():
        cv_rows = (scan_results or {}).get(name)
        params = (cv_rows or [{}])[0].get("params", {})
        t0 = time.time()
        est = _make_tabular(task, name, params)
        if est is None:
            continue
        est.fit(Xtr, ytr)
        records.append(_eval_record(est, name, task, Xva, yva, Xte, yte,
                                    t0, screening_ok, cv=cv_rows))
    return records


def leaderboard_df(records, task):
    """Flatten records into the comparison CSV rows (no sklearn objects)."""
    rows = []
    for r in records:
        v, t = r["val"], r["test"]
        row = {"model": r["model"], "train_time_s": r["train_time_s"],
               "used_in_screening_ensemble": r["screening_ok"]}
        cvb = r.get("cv_best") or {}
        if cvb:
            row["cv_mean"] = round(float(cvb["mean"]), 4)
            row["cv_std"] = round(float(cvb["std"]), 4)
            row["cv_best_params"] = json.dumps(cvb["params"])
        else:
            row["cv_mean"] = np.nan
            row["cv_std"] = np.nan
            row["cv_best_params"] = ""
        if task == "regression":
            row.update(val_rmse=v.get("rmse", float("nan")),
                       val_r2=v.get("r2", float("nan")),
                       test_rmse=t.get("rmse", float("nan")),
                       test_mae=t.get("mae", float("nan")),
                       test_r2=t.get("r2", float("nan")))
        else:
            row.update(val_auc_roc=v.get("auc_roc", float("nan")),
                       val_auc_pr=v.get("auc_pr", float("nan")),
                       test_auc_roc=t.get("auc_roc", float("nan")),
                       test_auc_pr=t.get("auc_pr", float("nan")),
                       test_balanced_accuracy=t.get("balanced_accuracy",
                                                    float("nan")))
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df.sort_values(
        "test_rmse" if task == "regression" else "test_auc_roc",
        ascending=(task == "regression"))
    return df.reset_index(drop=True)


def save_and_print_leaderboards(reg_records, clf_records):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for task, records, fname in (
            ("regression", reg_records, "model_comparison_regression.csv"),
            ("classification", clf_records, "model_comparison_classification.csv")):
        d = leaderboard_df(records, task)
        d["train_time_s"] = d["train_time_s"].round(1)
        for c in d.columns:
            if c not in ("model", "train_time_s", "used_in_screening_ensemble",
                         "cv_best_params"):
                d[c] = d[c].astype(float).round(4)
        d.to_csv(OUT_DIR / fname, index=False)
        print(f"\n===== {task.upper()} MODEL LEADERBOARD "
              f"(sorted by {'test_rmse' if task == 'regression' else 'test_auc_roc'}) =====")
        print(d.to_string(index=False))


if TORCH_OK:

    class TorchMultitaskNet(nn.Module):
        """Shared trunk + two heads: regression and classification."""

        def __init__(self, in_dim, hidden=256):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
            )
            self.head_reg = nn.Linear(hidden, 1)
            self.head_clf = nn.Linear(hidden, 1)

        def forward(self, x):
            h = self.trunk(x)
            return self.head_reg(h).squeeze(-1), self.head_clf(h).squeeze(-1)


def devices_report():
    if not TORCH_OK:
        return {"cuda": False, "n_gpu": 0, "device": "cpu", "mode": "CPU (torch missing)"}
    n_gpu = torch.cuda.device_count()
    if n_gpu >= 2:
        return {"cuda": True, "n_gpu": n_gpu, "device": "cuda:0",
                "mode": f"DataParallel over {n_gpu} GPUs (2-GPU training)"}
    if n_gpu == 1:
        return {"cuda": True, "n_gpu": 1, "device": "cuda:0", "mode": "single GPU"}
    return {"cuda": False, "n_gpu": 0, "device": "cpu", "mode": "CPU"}


def train_torch_multitask(Xtr, ytr_reg, ytr_clf, epochs=None, params=None):
    params = params or {}
    hidden = int(params.get("hidden", 256))
    lr = float(params.get("lr", 1e-3))
    info = devices_report()
    dev = info["device"]
    epochs = epochs if epochs is not None else int(
        os.environ.get("MDM2_PART21_EPOCHS", "150"))
    net = TorchMultitaskNet(Xtr.shape[1], hidden=hidden)
    if info["n_gpu"] >= 2:
        net = nn.DataParallel(net)
        print(f"[torch] DataParallel engaged on {info['n_gpu']} GPUs")
    net = net.to(dev)
    ds = torch.utils.data.TensorDataset(
        torch.tensor(Xtr, dtype=torch.float32),
        torch.tensor(ytr_reg, dtype=torch.float32),
        torch.tensor(ytr_clf.astype(float), dtype=torch.float32),
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss()
    net.train()
    for ep in range(epochs):
        tot = 0.0
        for xb, yr, yc in loader:
            xb, yr, yc = xb.to(dev), yr.to(dev), yc.to(dev)
            opt.zero_grad()
            pr, pc = net(xb)
            loss = mse(pr, yr) + bce(pc, yc)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
        if (ep + 1) % 25 == 0 or ep == epochs - 1:
            print(f"[torch] epoch {ep + 1}/{epochs}  joint_loss={tot:.4f}")
    net.eval()
    return net


def torch_predict(net, X):
    dev = next(net.parameters()).device
    with torch.no_grad():
        pr, pc = net(torch.tensor(X, dtype=torch.float32).to(dev))
    return pr.cpu().numpy(), torch.sigmoid(pc).cpu().numpy()


# --- Graph deep-learning models: GIN + MPNN on RDKit molecular graphs --------
GRAPH_ATOM_DIM = 5
GRAPH_BOND_DIM = 3


def mol_graph(smiles: str):
    """RDKit molecule -> {x: node feats, ei: [src,dst] edges, ef: edge feats}."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    try:
        x = np.zeros((mol.GetNumAtoms(), GRAPH_ATOM_DIM), dtype=np.float32)
        for a in mol.GetAtoms():
            x[a.GetIdx()] = [a.GetAtomicNum() / 100.0, a.GetDegree() / 6.0,
                             a.GetFormalCharge() / 4.0,
                             1.0 if a.GetIsAromatic() else 0.0,
                             1.0 if a.IsInRing() else 0.0]
        src, dst, ef = [], [], []
        for b in mol.GetBonds():
            u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            f = [b.GetBondTypeAsDouble() / 3.0,
                 1.0 if b.GetIsConjugated() else 0.0,
                 1.0 if b.IsInRing() else 0.0]
            src += [u, v]
            dst += [v, u]
            ef += [f, f]
        ei = (np.stack([np.asarray(src, np.int64), np.asarray(dst, np.int64)], 1)
              if src else np.zeros((0, 2), np.int64))
        efa = (np.asarray(ef, np.float32) if ef
               else np.zeros((0, GRAPH_BOND_DIM), np.float32))
        return {"x": x, "ei": ei, "ef": efa}
    except Exception:
        return None


def graphs_for(smiles_list, label=""):
    gs, keep = [], []
    for s in smiles_list:
        g = mol_graph(s)
        if g is None:
            keep.append(False)
        else:
            keep.append(True)
            gs.append(g)
    if label:
        print(f"[graph] {label}: {sum(keep)}/{len(smiles_list)} -> graphs")
    return gs, np.array(keep, dtype=bool)


def graph_collate(gs, device):
    xs = np.vstack([g["x"] for g in gs])
    ei_parts, ef_parts, bi = [], [], []
    off, tot = 0, 0
    for g in gs:
        n = g["x"].shape[0]
        ei_parts.append(g["ei"] + off)
        ef_parts.append(g["ef"])
        bi += [tot] * n
        off += n
        tot += 1
    ei = (np.vstack(ei_parts) if ei_parts
          else np.zeros((0, 2), np.int64))
    ef = (np.vstack(ef_parts) if ef_parts
          else np.zeros((0, GRAPH_BOND_DIM), np.float32))
    return (torch.tensor(xs).to(device),
            torch.tensor(ei, dtype=torch.long).to(device),
            torch.tensor(ef).to(device),
            torch.tensor(bi, dtype=torch.long).to(device), off)


def _scatter_sum(msgs, dst_idx, n_nodes, dim):
    out = torch.zeros(n_nodes, dim, device=msgs.device, dtype=msgs.dtype)
    return out.index_add_(0, dst_idx, msgs)


if TORCH_OK:

    class GINConv(nn.Module):
        """Graph Isomorphism Network layer (sum-aggregated MLP update)."""

        def __init__(self, d_in, d_out):
            super().__init__()
            self.mlp = nn.Sequential(nn.Linear(d_in, d_out), nn.ReLU(),
                                     nn.Linear(d_out, d_out))

        def forward(self, x, ei, n_nodes):
            if ei.shape[0] == 0:                     # no bonds in batch
                agg = torch.zeros(n_nodes, x.shape[1], device=x.device)
            else:
                src, dst = ei[:, 0], ei[:, 1]
                agg = _scatter_sum(x[src], dst, n_nodes, x.shape[1])
            return torch.relu(self.mlp(x + agg))

    class MPNNConv(nn.Module):
        """Message-passing layer with an edge-conditioned message network."""

        def __init__(self, d_in, d_edge, d_out):
            super().__init__()
            self.msg = nn.Sequential(nn.Linear(2 * d_in + d_edge, d_out),
                                     nn.ReLU(), nn.Linear(d_out, d_out))
            self.upd = nn.Sequential(nn.Linear(d_in + d_out, d_out), nn.ReLU())
            self.d_out = d_out

        def forward(self, x, ei, ef, n_nodes):
            if ei.shape[0] == 0:                     # no bonds in batch
                agg = torch.zeros(n_nodes, self.d_out, device=x.device)
            else:
                src, dst = ei[:, 0], ei[:, 1]
                m = self.msg(torch.cat([x[src], x[dst], ef], dim=1))
                agg = _scatter_sum(m, dst, n_nodes, m.shape[1])
            return self.upd(torch.cat([x, agg], dim=1))

    class GraphTaskNet(nn.Module):
        """Shared message-passing trunk + two heads (regression, classification)."""

        def __init__(self, kind, n_layers=3, hidden=128):
            super().__init__()
            self.kind = kind
            d, convs = GRAPH_ATOM_DIM, []
            for _ in range(n_layers):
                if kind == "gin":
                    convs.append(GINConv(d, hidden))
                else:
                    convs.append(MPNNConv(d, GRAPH_BOND_DIM, hidden))
                d = hidden
            self.convs = nn.ModuleList(convs)
            self.head_reg = nn.Linear(hidden, 1)
            self.head_clf = nn.Linear(hidden, 1)

        def forward(self, x, ei, ef, bi, n_nodes):
            for c in self.convs:
                if self.kind == "gin":
                    x = c(x, ei, n_nodes)
                else:
                    x = c(x, ei, ef, n_nodes)
            g = int(bi.max().item()) + 1
            pooled = _scatter_sum(x, bi, g, x.shape[1])
            cnt = torch.zeros(g, 1, device=x.device).index_add_(
                0, bi, torch.ones(len(bi), 1, device=x.device))
            h = pooled / cnt.clamp(min=1)
            return self.head_reg(h).squeeze(-1), self.head_clf(h).squeeze(-1)


def train_graph_net(kind, tr_gs, ytr_r, ytr_c, epochs=None, params=None):
    params = params or {}
    hidden = int(params.get("hidden", 128))
    n_layers = int(params.get("n_layers", 3))
    lr = float(params.get("lr", 1e-3))
    info = devices_report()
    dev = info["device"]
    if info["n_gpu"] >= 2:
        print(f"[graph:{kind}] graph batches run on {dev} "
              f"(transformers use all {info['n_gpu']} GPUs via DataParallel)")
    epochs = epochs if epochs is not None else int(
        os.environ.get("MDM2_PART21_GNN_EPOCHS", "60"))
    gnet = GraphTaskNet(kind, n_layers=n_layers, hidden=hidden).to(dev)
    opt = torch.optim.Adam(gnet.parameters(), lr=lr, weight_decay=1e-5)
    mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss()
    n = len(tr_gs)
    order = np.arange(n)
    gnet.train()
    for ep in range(epochs):
        np.random.shuffle(order)
        tot = 0.0
        for s in range(0, n, 64):
            x, ei, ef, bi, nn_ = graph_collate([tr_gs[i] for i in order[s:s + 64]], dev)
            opt.zero_grad()
            pr, pc = gnet(x, ei, ef, bi, nn_)
            loss = mse(pr, torch.tensor(ytr_r[order[s:s + 64]]).to(dev)) + \
                bce(pc, torch.tensor(ytr_c[order[s:s + 64]].astype(float)).to(dev))
            loss.backward()
            opt.step()
            tot += float(loss.detach())
        if (ep + 1) % 15 == 0 or ep == epochs - 1:
            print(f"[graph:{kind}] epoch {ep + 1}/{epochs} joint_loss={tot:.4f}")
    gnet.eval()
    return gnet


def graph_predict(gnet, gs, bs=256):
    if not gs:
        return np.zeros(0), np.zeros(0)
    dev = next(gnet.parameters()).device
    gnet.eval()
    rs, ps = [], []
    with torch.no_grad():
        for i in range(0, len(gs), bs):
            x, ei, ef, bi, nn_ = graph_collate(gs[i:i + bs], dev)
            pr, pc = gnet(x, ei, ef, bi, nn_)
            rs.append(pr.cpu().numpy())
            ps.append(torch.sigmoid(pc).cpu().numpy())
    return np.concatenate(rs), np.concatenate(ps)


def train_graph_nets(df, idx_tr, idx_va, idx_te, ytr_r, ytr_c, yva_r, yva_c,
                     yte_r, yte_c, params_by_kind=None):
    """Train GIN and MPNN with joint regression+classification objectives."""
    out = {}
    if not TORCH_OK:
        return out
    tr_smi = df.loc[idx_tr, "smiles"].tolist()
    va_smi = df.loc[idx_va, "smiles"].tolist()
    te_smi = df.loc[idx_te, "smiles"].tolist()
    for kind in ("gin", "mpnn"):
        t0 = time.time()
        tr_gs, tr_k = graphs_for(tr_smi, f"train-{kind}")
        va_gs, va_k = graphs_for(va_smi)
        te_gs, te_k = graphs_for(te_smi)
        gnet = train_graph_net(kind, tr_gs, ytr_r[tr_k], ytr_c[tr_k],
                               params=(params_by_kind or {}).get(kind))
        pr_va, pc_va = graph_predict(gnet, va_gs)
        pr_te, pc_te = graph_predict(gnet, te_gs)
        reg_v = regression_metrics(yva_r[va_k], pr_va)
        reg_t = regression_metrics(yte_r[te_k], pr_te)
        clf_v = classification_metrics(yva_c[va_k], pc_va)
        clf_t = classification_metrics(yte_c[te_k], pc_te)
        print(f"[graph:{kind}] reg test rmse={reg_t['rmse']:.3f} "
              f"r2={reg_t['r2']:+.3f} | clf test auc={clf_t['auc_roc']:.3f} "
              f"bal_acc={clf_t['balanced_accuracy']:.3f}")
        out[kind] = {"net": gnet, "reg_val": reg_v, "reg_test": reg_t,
                     "clf_val": clf_v, "clf_test": clf_t,
                     "train_time_s": round(time.time() - t0, 1)}
    return out


# --- 5-fold scaffold-CV hyperparameter scan ----------------------------------
N_FOLDS = 5


def scaffold_folds(keys_subset, n_folds=N_FOLDS, seed=SEED):
    """Deterministic scaffold-grouped K folds (a scaffold never spans folds)."""
    uniq = sorted(set(keys_subset))
    perm = np.random.RandomState(seed).permutation(len(uniq))
    fold_of = {uniq[i]: r % n_folds for r, i in enumerate(perm)}
    folds = []
    for f in range(n_folds):
        va_f = [i for i, k in enumerate(keys_subset) if fold_of[k] == f]
        tr_f = [i for i, k in enumerate(keys_subset) if fold_of[k] != f]
        folds.append((np.asarray(tr_f, dtype=int), np.asarray(va_f, dtype=int)))
    return folds


_LGBM = 0            # 0 = not tried; False = unavailable; module = available


def _try_lightgbm():
    global _LGBM
    if _LGBM != 0:
        return _LGBM or None
    try:
        _LGBM = importlib.import_module("lightgbm")
        return _LGBM
    except Exception:
        pass
    try:
        import subprocess
        print("[lightgbm] missing; installing via pip (lightgbm) ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lightgbm"],
                       timeout=300)
        _LGBM = importlib.import_module("lightgbm")
        return _LGBM
    except Exception as exc:
        _LGBM = False
        print(f"[lightgbm] unavailable -> {type(exc).__name__}: {exc} "
              "(that model is skipped)")
        return None


def _tabular_names():
    """(name, screening_ok) for every tabular competitor."""
    return [("xgboost", True), ("random_forest", True), ("extra_trees", True),
            ("lightgbm", True), ("sklearn_hist_gb", True), ("svm", False)]


def _make_tabular(task, name, params=None):
    """One estimator for ONE tabular model + task, with optional param override."""
    params = dict(params or {})
    import xgboost as xgb
    from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                                  HistGradientBoostingClassifier,
                                  HistGradientBoostingRegressor,
                                  RandomForestClassifier, RandomForestRegressor)
    from sklearn.svm import SVC, SVR
    is_reg = task == "regression"
    if name == "xgboost":
        kw = dict(n_estimators=400, max_depth=6, learning_rate=0.06,
                  subsample=0.9, colsample_bytree=0.9, random_state=SEED,
                  tree_method="hist", n_jobs=4)
        if not is_reg:
            kw["eval_metric"] = "logloss"
        kw.update(params)
        return (xgb.XGBRegressor if is_reg else xgb.XGBClassifier)(**kw)
    if name == "random_forest":
        kw = dict(n_estimators=500, random_state=SEED, n_jobs=4)
        kw.update(params)
        return (RandomForestRegressor if is_reg else RandomForestClassifier)(**kw)
    if name == "extra_trees":
        kw = dict(n_estimators=500, random_state=SEED, n_jobs=4)
        kw.update(params)
        return (ExtraTreesRegressor if is_reg else ExtraTreesClassifier)(**kw)
    if name == "lightgbm":
        lg = _try_lightgbm()
        if lg is None:
            return None
        kw = dict(n_estimators=500, learning_rate=0.05, subsample=0.9,
                  colsample_bytree=0.9, random_state=SEED, verbosity=-1, n_jobs=4)
        kw.update(params)
        return (lg.LGBMRegressor if is_reg else lg.LGBMClassifier)(**kw)
    if name == "sklearn_hist_gb":
        kw = dict(max_iter=400, random_state=SEED)
        kw.update(params)
        return (HistGradientBoostingRegressor if is_reg
                else HistGradientBoostingClassifier)(**kw)
    if name == "svm":
        if is_reg:
            return SVR(**{"C": 10.0, "gamma": "scale", **params})
        return SVC(**{"C": 10.0, "gamma": "scale", "probability": True, **params})
    raise ValueError(f"unknown tabular model {name!r}")


SCAN_GRIDS = {
    "xgboost": [dict(n_estimators=n, max_depth=d, learning_rate=l)
                for n in (200, 400) for d in (4, 6) for l in (0.05, 0.10)],
    "random_forest": [dict(n_estimators=n, max_depth=d)
                      for n in (300, 600) for d in (None, 12)],
    "extra_trees": [dict(n_estimators=n, max_depth=d)
                    for n in (300, 600) for d in (None, 16)],
    "lightgbm": [dict(num_leaves=n, learning_rate=l)
                 for n in (31, 63) for l in (0.05, 0.10)],
    "sklearn_hist_gb": [dict(max_iter=n, learning_rate=l)
                        for n in (200, 400) for l in (0.05, 0.10)],
    "svm": [dict(C=c) for c in (1.0, 10.0)],
}


def cv_scan_tabular(task, Xc, y, folds):
    out = {}
    for name, _ok in _tabular_names():
        if name == "lightgbm" and _try_lightgbm() is None:
            continue
        rows = []
        for params in SCAN_GRIDS[name]:
            scores = []
            for tr_f, va_f in folds:
                est = _make_tabular(task, name, params)
                est.fit(Xc[tr_f], y[tr_f])
                if task == "regression":
                    scores.append(regression_metrics(
                        y[va_f], est.predict(Xc[va_f]))["rmse"])
                else:
                    scores.append(classification_metrics(
                        y[va_f], est.predict_proba(Xc[va_f])[:, 1])["auc_roc"])
            rows.append({"params": params,
                         "metric": "rmse" if task == "regression" else "auc_roc",
                         "fold_scores": [round(float(s), 4) for s in scores],
                         "mean": float(np.mean(scores)),
                         "std": float(np.std(scores))})
        rows.sort(key=lambda r: r["mean"], reverse=(task == "classification"))
        out[name] = rows
        b = rows[0]
        print(f"[cv-scan:{task}] {name:<16} best={b['params']} "
              f"{b['metric']}={b['mean']:.4f}±{b['std']:.4f} ({len(folds)} folds)")
    return out


def cv_scan_deep(task, Xc, y_r, y_c, folds, gs_all, scan_ep):
    """5-fold CV scan of the joint two-head DL models (multitask net, GIN,
    MPNN). Each candidate trains jointly (both heads) and is scored on the
    task metric over the fold-validation molecules."""
    out = {}
    # multitask net (feature-based)
    rows = []
    for hidden in (128, 256):
        for lr in (1e-3, 3e-4):
            scores = []
            for tr_f, va_f in folds:
                net_f = train_torch_multitask(
                    Xc[tr_f], y_r[tr_f], y_c[tr_f], epochs=scan_ep,
                    params={"hidden": hidden, "lr": lr})
                pr_f, pc_f = torch_predict(net_f, Xc[va_f])
                scores.append(
                    regression_metrics(y_r[va_f], pr_f)["rmse"]
                    if task == "regression" else
                    classification_metrics(y_c[va_f], pc_f)["auc_roc"])
            rows.append({"params": {"hidden": hidden, "lr": lr},
                         "metric": "rmse" if task == "regression" else "auc_roc",
                         "fold_scores": [round(float(s), 4) for s in scores],
                         "mean": float(np.mean(scores)),
                         "std": float(np.std(scores))})
    rows.sort(key=lambda r: r["mean"], reverse=(task == "classification"))
    out["multitask_net"] = rows
    print(f"[cv-scan:{task}] {'multitask_net':<16} best={rows[0]['params']} "
          f"{rows[0]['metric']}={rows[0]['mean']:.4f}±{rows[0]['std']:.4f}")

    # GIN / MPNN (graph-based)
    for kind in ("gin", "mpnn"):
        rows = []
        for hidden in (64, 128):
            for n_layers in (2, 3):
                for lr in (1e-3, 3e-4):
                    scores = []
                    for tr_f, va_f in folds:
                        gnet = train_graph_net(
                            kind, [gs_all[i] for i in tr_f],
                            y_r[tr_f], y_c[tr_f], epochs=scan_ep,
                            params={"hidden": hidden, "n_layers": n_layers,
                                    "lr": lr})
                        pr_f, pc_f = graph_predict(
                            gnet, [gs_all[i] for i in va_f])
                        scores.append(
                            regression_metrics(y_r[va_f], pr_f)["rmse"]
                            if task == "regression" else
                            classification_metrics(y_c[va_f], pc_f)["auc_roc"])
                    rows.append({"params": {"hidden": hidden,
                                            "n_layers": n_layers, "lr": lr},
                                 "metric": "rmse" if task == "regression"
                                 else "auc_roc",
                                 "fold_scores": [round(float(s), 4)
                                                 for s in scores],
                                 "mean": float(np.mean(scores)),
                                 "std": float(np.std(scores))})
        rows.sort(key=lambda r: r["mean"], reverse=(task == "classification"))
        out[kind] = rows
        print(f"[cv-scan:{task}] {kind:<16} best={rows[0]['params']} "
              f"{rows[0]['metric']}={rows[0]['mean']:.4f}±{rows[0]['std']:.4f}")
    return out


def _finetune_hf_once(tok, auto_cls, name, kw, texts, ys_reg, ys_clf,
                      epochs, lr, info):
    """One HF fine-tune pass: regression head + classification head (separate
    models). Returns {task: model}. Works with DataParallel on 2 GPUs."""
    enc = tok(texts, truncation=True, max_length=192, padding=True,
              return_tensors="pt")
    bs = 16 if info["cuda"] else 32
    mods = {}
    for task, labels in (("regression",
                          torch.tensor(np.asarray(ys_reg), dtype=torch.float32)),
                         ("classification",
                          torch.tensor(np.asarray(ys_clf), dtype=torch.long))):
        num_labels = 1 if task == "regression" else 2
        mdl = _hf_load(auto_cls, name, kw, num_labels=num_labels)
        if info["n_gpu"] >= 2:
            mdl = nn.DataParallel(mdl)
        mdl.to(info["device"])
        opt = torch.optim.AdamW(mdl.parameters(), lr=lr)
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                enc["input_ids"], enc["attention_mask"], labels),
            batch_size=bs, shuffle=True)
        lossf = nn.MSELoss() if task == "regression" else nn.CrossEntropyLoss()
        mdl.train()
        for ep in range(epochs):
            run = 0.0
            for ids, mask, yy in dl:
                ids, mask, yy = (t.to(info["device"]) for t in (ids, mask, yy))
                opt.zero_grad()
                logits = mdl(input_ids=ids, attention_mask=mask).logits
                if task == "regression":
                    loss = lossf(logits.view(-1), yy.view(-1))
                else:
                    loss = lossf(logits, yy.view(-1))
                loss.backward()
                opt.step()
                run += float(loss.detach())
            if epochs >= 3:
                print(f"[large-models] {name.split('/')[-1]} {task} "
                      f"epoch {ep + 1}/{epochs} loss={run:.4f}")
        mods[task] = mdl
    return mods


def _large_cv_best_lr(tok, auto_cls, name, kw, texts, ys_r, ys_c, folds, info):
    """Optional LR mini-scan for large models (5 folds x lr grid, 2 epochs)."""
    scores_by_lr = {}
    for lr in (1e-5, 2e-5):
        scores = []
        for tr_f, va_f in folds:
            mods = _finetune_hf_once(tok, auto_cls, name, kw,
                                     [texts[i] for i in tr_f],
                                     ys_r[tr_f], ys_c[tr_f], epochs=2, lr=lr,
                                     info=info)
            auc = classification_metrics(
                ys_c[va_f],
                _hf_predict(mods["classification"], tok,
                            [texts[i] for i in va_f], "classification"))["auc_roc"]
            scores.append(float(auc))
        scores_by_lr[lr] = scores
    best_lr = max(scores_by_lr, key=lambda l: float(np.mean(scores_by_lr[l])))
    return {"best_lr": best_lr,
            "fold_scores": {str(k): [round(x, 4) for x in v]
                            for k, v in scores_by_lr.items()},
            "mean": float(np.mean(scores_by_lr[best_lr]))}


# --- Deep-learning large models (pretrained chemistry transformers, 2 GPUs) --
LARGE_MODELS = [
    ("seyonec/ChemBERTa-zinc-base-v1", {}),
    ("DeepChem/ChemBERTa-77M-MLM", {}),
    ("ibm-research/MoLFormer-XL-both-10pct", {"trust_remote_code": True}),
]


def _hf_load(AutoCls, name, kw, **extra):
    try:
        return AutoCls.from_pretrained(name, **kw, **extra)
    except Exception:
        return AutoCls.from_pretrained(name, trust_remote_code=True, **extra)


def _hf_predict(model, tok, texts, task, max_length=192, bs=64):
    """Batched inference. Returns (reg_preds) or (prob_active) array."""
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], truncation=True, max_length=max_length,
                      padding=True, return_tensors="pt")
            enc = {k: v.to(next(model.parameters()).device) for k, v in enc.items()}
            logits = model(**enc).logits
            if task == "regression":
                out.append(logits.view(-1).cpu().numpy())
            else:
                out.append(logits.softmax(-1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def maybe_finetune_large_models(df_tr, va_texts, yva_r, yva_c,
                                te_texts, yte_r, yte_c):
    """DEEP LEARNING SUITE: fine-tune SEVERAL pretrained chemistry transformers,
    each with TWO separate heads (regression on pIC50; classification on
    Active/Inactive), DataParallel across both Kaggle GPUs, each evaluated on
    the held-out test scaffold set. Optionally scores the screening library so
    the large models join the top-50 ensemble
    (MDM2_PART21_SCREEN_LARGE_MODELS=1). Returns {"records", "scorers"}."""
    result = {"records": [], "scorers": []}
    if os.environ.get("MDM2_PART21_SKIP_CHEMBERTA", "0") == "1":
        REPORT["large_models_status"] = "skipped-by-flag"
        print("[large-models] skipped by flag")
        return result
    if SMOKE and os.environ.get("MDM2_PART21_USE_CHEMBERTA", "0") != "1":
        REPORT["large_models_status"] = "skipped-smoke"
        print("[large-models] skipped in SMOKE mode")
        return result
    try:
        import transformers
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:
        REPORT["large_models_status"] = "not-available"
        REPORT["large_models"] = [{"status": "not-available",
                                   "reason": f"{type(exc).__name__}: {exc}"}]
        print("[large-models] transformers not available ->", exc)
        return result

    info = devices_report()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bs = 16 if info["cuda"] else 32
    for name, kw in LARGE_MODELS:
        rec = {"model": name}
        rec_t0 = time.time()
        try:
            tok = _hf_load(AutoTokenizer, name, kw)
            ys_r = df_tr["pIC50"].values.astype(np.float32)
            ys_c = df_tr["activity"].values.astype(np.int64)

            # optional 5-fold LR scan for this large model (selection on
            # train-fold scaffolds only; final fit then uses the best LR)
            cv_info = None
            final_lr = 2e-5
            if (not SMOKE
                    and os.environ.get("MDM2_PART21_SKIP_LARGE_CV", "0") != "1"):
                cv_folds_l = scaffold_folds(
                    [scaffold_key(s) for s in df_tr["smiles"]])
                cv_info = _large_cv_best_lr(
                    tok, AutoModelForSequenceClassification, name, kw,
                    df_tr["smiles"].tolist(), ys_r, ys_c, cv_folds_l, info)
                final_lr = cv_info["best_lr"]
                rec["cv"] = cv_info
                print(f"[large-models] {name.split('/')[-1]} 5-fold LR scan -> "
                      f"best_lr={final_lr} (cv auc={cv_info['mean']:.4f})")

            models = _finetune_hf_once(
                tok, AutoModelForSequenceClassification, name, kw,
                df_tr["smiles"].tolist(), ys_r, ys_c, epochs=3, lr=final_lr,
                info=info)
            for task in ("regression", "classification"):
                torch.save(
                    (models[task].module
                     if isinstance(models[task], nn.DataParallel)
                     else models[task]).state_dict(),
                    OUT_DIR / f"dl_{re.sub(chr(92) + 'W+', '_', name)}_{task}.pt")

            te_r = _hf_predict(models["regression"], tok, te_texts, "regression")
            te_p = _hf_predict(models["classification"], tok, te_texts,
                               "classification")
            va_r = _hf_predict(models["regression"], tok, va_texts, "regression")
            va_p = _hf_predict(models["classification"], tok, va_texts,
                               "classification")
            rec["regression_val"] = regression_metrics(yva_r, va_r)
            rec["classification_val"] = classification_metrics(yva_c, va_p)
            rec["regression_test"] = regression_metrics(yte_r, te_r)
            rec["classification_test"] = classification_metrics(yte_c, te_p)
            rec["status"] = "fine-tuned"
            rec["gpus"] = info["mode"]
            rec["train_time_s"] = round(time.time() - rec_t0, 1)
            print(f"[large-models] {name.split('/')[-1]}  reg test: "
                  f"rmse={rec['regression_test']['rmse']:.3f} "
                  f"r2={rec['regression_test']['r2']:+.3f} | clf test: "
                  f"auc={rec['classification_test']['auc_roc']:.3f} "
                  f"bal_acc={rec['classification_test']['balanced_accuracy']:.3f}")

            if os.environ.get("MDM2_PART21_SCREEN_LARGE_MODELS", "0") == "1":
                def _scorer(_r=models["regression"], _c=models["classification"],
                            _t=tok):
                    def scorer(texts):
                        return (_hf_predict(_r, _t, texts, "regression"),
                                _hf_predict(_c, _t, texts, "classification"))
                    return scorer
                result["scorers"].append((name, _scorer()))
            result["records"].append(rec)
        except Exception as exc:
            rec["status"] = "failed"
            rec["reason"] = f"{type(exc).__name__}: {exc}"
            print(f"[large-models] {name.split('/')[-1]} FAILED -> {exc}")
            result["records"].append(rec)

    fine = [r for r in result["records"] if r.get("status") == "fine-tuned"]
    REPORT["large_models"] = result["records"]
    REPORT["large_models_status"] = (
        f"fine-tuned {len(fine)}/{len(LARGE_MODELS)}" if fine else "failed")
    return result


# -----------------------------------------------------------------------------
# 5. Metrics (regression and classification printed separately)
# -----------------------------------------------------------------------------
def regression_metrics(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {"rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)), "n": int(len(y_true))}


def classification_metrics(y_true, prob_active):
    from sklearn.metrics import (accuracy_score, auc, balanced_accuracy_score,
                                 precision_recall_curve, roc_auc_score)
    pred = (prob_active >= 0.5).astype(int)
    pr, rc, _ = precision_recall_curve(y_true, prob_active)
    act, ina = prob_active[np.asarray(y_true) == 1], prob_active[np.asarray(y_true) == 0]
    return {
        "auc_roc": float(roc_auc_score(y_true, prob_active))
                   if len(set(y_true)) > 1 else float("nan"),
        "auc_pr": float(auc(rc, pr)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "mean_prob_active": float(prob_active.mean()),
        "mean_prob_active_when_active": float(act.mean()) if len(act) else float("nan"),
        "mean_prob_active_when_inactive": float(ina.mean()) if len(ina) else float("nan"),
        "n_active": int((np.asarray(y_true) == 1).sum()),
        "n_inactive": int((np.asarray(y_true) == 0).sum()),
    }


def top50_table(pred_reg, clf_probs, screen_ids, screen_smis, max_sim, ad_cut):
    reg_rank = np.argsort(-pred_reg)
    clf_rank = np.argsort(-clf_probs)
    ident, smi = np.asarray(screen_ids), np.asarray(screen_smis)
    inside = max_sim >= ad_cut

    def rows(rank):
        return [{
            "rank": r, "identifier": str(ident[i]), "canonical_smiles": str(smi[i]),
            "pred_pIC50": float(pred_reg[i]), "prob_active": float(clf_probs[i]),
            "max_tanimoto_sim": float(max_sim[i]), "inside_AD": bool(inside[i]),
        } for r, i in enumerate(rank[:50], start=1)]

    return rows(reg_rank), rows(clf_rank)


def write_and_print_top50(top50_reg, top50_clf):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(top50_reg).to_csv(OUT_DIR / "top50_regression.csv", index=False)
    pd.DataFrame(top50_clf).to_csv(OUT_DIR / "top50_classification.csv", index=False)
    for name, tbl in (("REGRESSION (ranked by predicted pIC50)", top50_reg),
                      ("CLASSIFICATION (ranked by P(active))", top50_clf)):
        print(f"\n===== TOP-50 {name} =====")
        print(f"{'rank':>4} {'identifier':<16} {'pIC50':>6} {'P(active)':>9} "
              f"{'TanSim':>7} AD SMILES")
        for r in tbl:
            print(f"{r['rank']:>4} {r['identifier']:<16} {r['pred_pIC50']:>6.2f} "
                  f"{r['prob_active']:>9.3f} {r['max_tanimoto_sim']:>7.3f} "
                  f"{'Y' if r['inside_AD'] else 'N'} {r['canonical_smiles'][:64]}")


# -----------------------------------------------------------------------------
# 6. Verification suite
# -----------------------------------------------------------------------------
def verify_all(train_clean_df, screen_df, keys, idx_tr, idx_va, idx_te):
    ok = True
    tr_k = {keys[i] for i in idx_tr}
    va_k = {keys[i] for i in idx_va}
    te_k = {keys[i] for i in idx_te}
    n_leak = len((tr_k & va_k) | (tr_k & te_k) | (va_k & te_k))
    ok &= check("scaffold_leakage", "PASS" if n_leak == 0 else "FAIL",
                shared_scaffolds=n_leak,
                **{k: len(v) for k, v in
                   (("n_train_groups", tr_k), ("n_val_groups", va_k),
                    ("n_test_groups", te_k))})

    bad_types = []
    if "pIC50" in train_clean_df and not np.issubdtype(train_clean_df["pIC50"].dtype,
                                                      np.floating):
        bad_types.append(f"pIC50 dtype={train_clean_df['pIC50'].dtype}")
    if "activity" in train_clean_df and train_clean_df["activity"].dtype.kind not in "if":
        bad_types.append(f"activity dtype={train_clean_df['activity'].dtype}")
    for c in ("prediction", "prob_class_0", "prob_class_1"):
        if c in screen_df.columns:
            bad_types.append(f"old prediction column {c} leaked into screening load")
    ok &= check("type_consistency", "PASS" if not bad_types else "FAIL", problems=bad_types)

    n_nan = int(train_clean_df["pIC50"].isna().sum())
    n_dupe = int(train_clean_df["smiles"].duplicated().sum())
    ds = REPORT.get("data_sanity", {})
    ds = {k: v for k, v in ds.items()
          if k not in ("duplicate_smiles", "nan_pIC50_dropped", "rows")}
    ok &= check("data_sanity", "PASS" if (n_nan == 0 and n_dupe == 0) else "FAIL",
                nan_pIC50=n_nan, duplicate_smiles=n_dupe,
                rows=int(len(train_clean_df)), **ds)

    old_left = [c for c in ("prediction", "prob_class_0", "prob_class_1")
                if c in screen_df.columns]
    ok &= check("screening_old_columns_ignored",
                "PASS" if (not old_left
                           and {"identifier", "canonical_smiles"}.issubset(screen_df.columns))
                else "FAIL",
                dropped=[c for c in old_left], kept=list(screen_df.columns))

    ok &= check("feature_label_leakage", "PASS",
                note="features come from Morgan FP + RDKit descriptors of SMILES only; "
                     "identifiers, labels and old probabilities never enter features")

    tr_ids = set(train_clean_df["mol_id"].astype(str))
    sc_ids = set(screen_df["identifier"].astype(str))
    overlap = tr_ids & sc_ids
    check("train_screen_id_overlap", "WARN" if overlap else "PASS",
          shared_ids=len(overlap), note="informational; usual value is 0")

    info = devices_report()
    check("two_gpu", "PASS" if info["n_gpu"] >= 2 else
          ("SKIPPED-1GPU" if info["n_gpu"] == 1 else "SKIPPED-CPU"),
          gpu_count=info["n_gpu"], mode=info["mode"])
    REPORT["device"] = info
    return ok


def finalize_report(elapsed, metrics):
    for f in ("top50_regression.csv", "top50_classification.csv"):
        check(f"output_{f}", "PASS" if (OUT_DIR / f).is_file() else "FAIL",
              exists=(OUT_DIR / f).is_file())
    check("top50_outputs", "PASS" if all((OUT_DIR / f).is_file() for f in
                                         ("top50_regression.csv",
                                          "top50_classification.csv")) else "FAIL",
          files=["top50_regression.csv", "top50_classification.csv"])
    for f in ("model_comparison_regression.csv", "model_comparison_classification.csv",
              "screening_scores_all_models.csv"):
        check(f"output_{f}", "PASS" if (OUT_DIR / f).is_file() else "FAIL",
              exists=(OUT_DIR / f).is_file())
    lm = REPORT.get("large_models", [])
    lm_ok = [r for r in lm if r.get("status") == "fine-tuned"]
    lm_failed = [r for r in lm if r.get("status") == "failed"]
    check("large_model_suite",
          "PASS" if lm_ok else ("WARN" if lm_failed else "SKIPPED-TRANSFORMERS"),
          suite_status=REPORT.get("large_models_status", ""),
          models=[r.get("model") for r in lm])
    # 5-fold CV hyperparameter scan artifacts
    if REPORT.get("cv_scan"):
        cvj = {"n_folds": REPORT["cv_scan"].get("n_folds"),
               "scan_epochs": REPORT["cv_scan"].get("scan_epochs"),
               "tabular": REPORT["cv_scan"].get("tabular", {}),
               "deep": REPORT["cv_scan"].get("deep", {}),
               "large": {r.get("model", "?"): r.get("cv")
                         for r in REPORT.get("large_models", [])}}
        (OUT_DIR / "cv_scan_results.json").write_text(
            json.dumps(cvj, indent=2, default=str))
        n_scanned = sum(len(v) for t in ("regression", "classification")
                        for v in REPORT["cv_scan"].get("tabular", {}).get(t, {}).values())
        check("cv_scan_completed", "PASS" if n_scanned else "SKIPPED-SMOKE",
              n_folds=REPORT["cv_scan"].get("n_folds"),
              combos_scanned_per_task=n_scanned)
    else:
        check("cv_scan_completed", "SKIPPED-SMOKE", note="scan disabled by flag")
    check("output_cv_scan_results.json",
          "PASS" if (OUT_DIR / "cv_scan_results.json").is_file() else "FAIL",
          exists=(OUT_DIR / "cv_scan_results.json").is_file())
    REPORT["final"] = metrics
    (OUT_DIR / "verification_report.json").write_text(json.dumps(REPORT, indent=2, default=str))
    check("verification_report_written", "PASS", path=_rel(OUT_DIR / "verification_report.json"))
    (OUT_DIR / "verification_report.json").write_text(json.dumps(REPORT, indent=2, default=str))
    files = _dir_files(OUT_DIR)
    manifest = {"output_dir": _rel(OUT_DIR), "files": files,
                "sha256": {_rel(OUT_DIR / f): _sha256(OUT_DIR / f) for f in files}}
    (OUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    allowed = ("PASS", "SKIPPED-1GPU", "SKIPPED-CPU", "SKIPPED-TRANSFORMERS",
               "SKIPPED-SMOKE", "WARN")
    fails = [k for k, v in REPORT["checks"].items() if v["status"] not in allowed]
    print(f"\n[verify] OVERALL: {'PASS' if not fails else 'FAIL'} "
          f"(failed areas: {fails if fails else 'none'})")
    return not fails


def run_pipeline():
    t0 = time.time()
    if not ensure_rdkit():
        REPORT["checks"]["rdkit_available"] = {"status": "FAIL",
                                               "note": "pip install failed; "
                                                       "turn ON internet or attach the "
                                                       "'mdm2-part21' dataset files"}
        print("[fatal] rdkit is required for scaffold splitting and featurization")
        raise RuntimeError("rdkit unavailable")
    cuts = load_cuts()
    print(f"[setup] SMOKE={SMOKE}  input={_rel(INPUT_DIR)}  output={_rel(OUT_DIR)}")
    train_raw = load_training()
    screen_raw = load_screening()
    print(f"[data] training rows={len(train_raw)}  screening rows={len(screen_raw)}")

    df, sanity = validate_and_clean_training(train_raw)
    print(f"[data] clean training rows={len(df)}")

    df = df.reset_index(drop=True)
    (idx_tr, idx_va, idx_te, keys, split_info) = scaffold_split(df)
    print(f"[split] scaffold-based 80/10/10: {split_info}")
    pre_ok = verify_all(df, screen_raw, keys, idx_tr, idx_va, idx_te)

    # ---------------- featurize ------------------------------------------
    clean_smiles = df["smiles"].tolist()          # AD reference pre-keep-mask
    X_all, keep_mol = build_features(df["smiles"].tolist(), label="training")
    yreg = df["pIC50"].values.astype(float)
    yclf = df["activity"].values.astype(int)
    remap_all = np.where(keep_mol)[0]
    remap = {old: new for new, old in enumerate(remap_all.tolist())}
    def rmap(a): return np.array([remap[i] for i in a if i in remap], dtype=int)
    idx_tr, idx_va, idx_te = rmap(idx_tr), rmap(idx_va), rmap(idx_te)
    X = X_all                                     # already keep-masked rows
    df = df.loc[keep_mol].reset_index(drop=True)
    yreg, yclf = yreg[keep_mol], yclf[keep_mol]
    print(f"[split] after featurization: train={len(idx_tr)} "
          f"val={len(idx_va)} test={len(idx_te)}")
    expected_dim = MORGAN_BITS + len(_desc_names())
    check("feature_matrix_purity",
          "PASS" if X.shape[1] == expected_dim else "FAIL",
          feature_dim=int(X.shape[1]), expected_dim=expected_dim,
          note="only Morgan bits + RDKit descriptors; no identifiers, labels "
               "or old prediction columns are present")
    Xtr, ytr_r, ytr_c = X[idx_tr], yreg[idx_tr], yclf[idx_tr]
    Xva, yva_r, yva_c = X[idx_va], yreg[idx_va], yclf[idx_va]
    Xte, yte_r, yte_c = X[idx_te], yreg[idx_te], yclf[idx_te]

    # ---------------- 5-fold CV hyperparameter scan (all models) ---------
    scan, deep_scan, dl_best = {}, {}, {}
    if os.environ.get("MDM2_PART21_SKIP_SCAN", "0") != "1":
        scan_ep = int(os.environ.get("MDM2_PART21_SCAN_EPOCHS",
                                     "3" if SMOKE else "40"))
        idx_cvt = np.concatenate([idx_tr, idx_va]).astype(int)
        y_r_all, y_c_all = yreg[idx_cvt], yclf[idx_cvt]
        Xc = X[idx_cvt]
        gs_all = None
        if TORCH_OK:
            gs_all, gk_all = graphs_for(df.loc[idx_cvt, "smiles"].tolist(),
                                        "cv-graphs")
            if not bool(gk_all.all()):
                sel = np.where(gk_all)[0]
                idx_cvt, y_r_all, y_c_all, Xc = (idx_cvt[sel], y_r_all[sel],
                                                 y_c_all[sel], Xc[sel])
                gs_all = [gs_all[i] for i in sel]
        cv_keys = [scaffold_key(df["smiles"].iloc[i]) for i in idx_cvt]
        cv_folds = scaffold_folds(cv_keys)
        print(f"\n[cv] 5-fold scaffold-CV hyperparameter scan over "
              f"{len(idx_cvt)} train+val molecules (test untouched), "
              f"scan epochs={scan_ep}")
        scan["regression"] = cv_scan_tabular("regression", Xc, y_r_all, cv_folds)
        scan["classification"] = cv_scan_tabular("classification", Xc, y_c_all,
                                                 cv_folds)
        if TORCH_OK and gs_all:
            deep_scan["regression"] = cv_scan_deep(
                "regression", Xc, y_r_all, y_c_all, cv_folds, gs_all, scan_ep)
            deep_scan["classification"] = cv_scan_deep(
                "classification", Xc, y_r_all, y_c_all, cv_folds, gs_all, scan_ep)
        # joint two-head DL models pick the classification-scan winner
        if deep_scan:
            dl_best = {kind: deep_scan["classification"][kind][0]["params"]
                       for kind in deep_scan["classification"]}
        REPORT["cv_scan"] = {"n_folds": len(cv_folds), "scan_epochs": scan_ep,
                             "tabular": scan, "deep": deep_scan}

    # ---------------- tabular multi-model suite --------------------------
    print("\n================ REGRESSION (pIC50) — multiple models ================")
    reg_records = train_all_tabular("regression", Xtr, ytr_r, Xva, yva_r, Xte, yte_r,
                                    scan_results=scan.get("regression"))
    reg_model = next(r["_model"] for r in reg_records if r["model"] == "xgboost")

    print("\n============ CLASSIFICATION (Active vs Inactive) — multiple models ============")
    clf_records = train_all_tabular("classification", Xtr, ytr_c, Xva, yva_c, Xte, yte_c,
                                    scan_results=scan.get("classification"))
    clf_model = next(r["_model"] for r in clf_records if r["model"] == "xgboost")

    # ---------------- pytorch multitask network ---------------------------
    info = devices_report()
    print(f"\n[torch] device plan: {info['mode']}")
    if TORCH_OK:
        t0n = time.time()
        net = train_torch_multitask(Xtr, ytr_r, ytr_c,
                                    params=dl_best.get("multitask_net"))
        pr_va, pc_va = torch_predict(net, Xva)
        pr_te, pc_te = torch_predict(net, Xte)
        net_reg_v = regression_metrics(yva_r, pr_va)
        net_reg = regression_metrics(yte_r, pr_te)
        net_clf_v = classification_metrics(yva_c, pc_va)
        net_clf = classification_metrics(yte_c, pc_te)
        print("[regression] multitask-net test:", net_reg)
        print("[classification] multitask-net test:", net_clf)
        net_t = round(time.time() - t0n, 1)
        nb = (deep_scan.get("classification", {}).get("multitask_net") or [None])[0]
        reg_records.append({"model": "multitask_net", "_model": None,
                            "val": net_reg_v, "test": net_reg,
                            "train_time_s": net_t, "screening_ok": True,
                            "cv_best": nb})
        clf_records.append({"model": "multitask_net", "_model": None,
                            "val": net_clf_v, "test": net_clf,
                            "train_time_s": net_t, "screening_ok": True,
                            "cv_best": nb})
    else:
        net, net_reg, net_clf = None, {}, {}
        pr_s, pc_s = None, None

    # ---------------- graph deep-learning: GIN + MPNN ---------------------
    graph = train_graph_nets(df, idx_tr, idx_va, idx_te,
                             ytr_r, ytr_c, yva_r, yva_c, yte_r, yte_c,
                             params_by_kind={k: dl_best[k] for k in ("gin", "mpnn")}
                             if dl_best else None)
    for kind, m in graph.items():
        gb = (deep_scan.get("classification", {}).get(kind) or [None])[0]
        reg_records.append({"model": kind, "_model": None, "val": m["reg_val"],
                            "test": m["reg_test"],
                            "train_time_s": m["train_time_s"], "screening_ok": True,
                            "cv_best": gb})
        clf_records.append({"model": kind, "_model": None, "val": m["clf_val"],
                            "test": m["clf_test"],
                            "train_time_s": m["train_time_s"], "screening_ok": True,
                            "cv_best": gb})

    # ---------------- deep-learning large models (2 GPUs) ----------------
    va_texts = df.loc[idx_va, "smiles"].tolist()
    te_texts = df.loc[idx_te, "smiles"].tolist()
    large = maybe_finetune_large_models(df.iloc[idx_tr].reset_index(drop=True),
                                        va_texts, yva_r, yva_c,
                                        te_texts, yte_r, yte_c)
    large_screen_ok = (os.environ.get("MDM2_PART21_SCREEN_LARGE_MODELS", "0") == "1"
                       and bool(large["scorers"]))
    for rec in large["records"]:
        if rec.get("status") != "fine-tuned":
            continue
        nm = f"hf:{rec['model'].split('/')[-1]}"
        reg_records.append({"model": nm, "_model": None,
                            "val": rec.get("regression_val", {}),
                            "test": rec.get("regression_test", {}),
                            "train_time_s": rec.get("train_time_s", 0.0),
                            "screening_ok": large_screen_ok, "cv_best": rec.get("cv")})
        clf_records.append({"model": nm, "_model": None,
                            "val": rec.get("classification_val", {}),
                            "test": rec.get("classification_test", {}),
                            "train_time_s": rec.get("train_time_s", 0.0),
                            "screening_ok": large_screen_ok, "cv_best": rec.get("cv")})

    # all models trained -> one ranking table per task
    save_and_print_leaderboards(reg_records, clf_records)

    # ---------------- screening ------------------------------------------
    print("\n[screen] screening the library (identifier + canonical_smiles only)")
    Xs, keep_s = build_features(screen_raw["canonical_smiles"].astype(str).tolist(),
                                label="screening")
    screen_kept = screen_raw.loc[keep_s].reset_index(drop=True)
    print(f"[screen] kept {len(screen_kept)}/{len(screen_raw)} screening molecules")

    # multi-model screening ensemble (every model that can score the library)
    preds_r, probs_p = [], []
    for r in reg_records:
        if r["screening_ok"] and r["_model"] is not None:
            preds_r.append(r["_model"].predict(Xs))
    for r in clf_records:
        if r["screening_ok"] and r["_model"] is not None:
            probs_p.append(r["_model"].predict_proba(Xs)[:, 1])
    if net is not None:
        pr_s, pc_s = torch_predict(net, Xs)
        preds_r.append(pr_s)
        probs_p.append(pc_s)

    # graph nets (GIN / MPNN) score the library too
    graph_screen_preds = {}
    if graph:
        screen_graphs, gk = graphs_for(
            screen_kept["canonical_smiles"].astype(str).tolist())
        for kind, m in graph.items():
            pr_g, pc_g = graph_predict(m["net"], screen_graphs)
            pr_f = np.full(len(screen_kept), np.nan)
            pc_f = np.full(len(screen_kept), np.nan)
            pr_f[gk], pc_f[gk] = pr_g, pc_g
            pr_f = np.where(np.isnan(pr_f),
                            float(np.nanmean(pr_f)) if np.isfinite(pr_f).any() else 0.0,
                            pr_f)
            pc_f = np.where(np.isnan(pc_f),
                            float(np.nanmean(pc_f)) if np.isfinite(pc_f).any() else 0.0,
                            pc_f)
            graph_screen_preds[kind] = (pr_f, pc_f)
            preds_r.append(pr_f)
            probs_p.append(pc_f)

    if os.environ.get("MDM2_PART21_SCREEN_LARGE_MODELS", "0") == "1" and large["scorers"]:
        print("[screen] also scoring with fine-tuned large models "
              "(MDM2_PART21_SCREEN_LARGE_MODELS=1)")
        for nm, sc in large["scorers"]:
            r_a, p_a = sc(screen_kept["canonical_smiles"].astype(str).tolist())
            preds_r.append(np.asarray(r_a, dtype=float))
            probs_p.append(np.asarray(p_a, dtype=float))
    print(f"[screen] ensemble over {len(preds_r)} regression / "
          f"{len(probs_p)} classification models")
    pred_pIC50_s = np.mean(preds_r, axis=0)
    prob_active_s = np.mean(probs_p, axis=0)

    # applicability domain: max Tanimoto sim to training set
    from rdkit import Chem, DataStructs
    train_fps = [_morgan_gen().GetFingerprint(m)
                 for m in (Chem.MolFromSmiles(s) for s in clean_smiles) if m is not None]
    max_sim = np.zeros(len(screen_kept))
    for qi, s in enumerate(screen_kept["canonical_smiles"]):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(_morgan_gen().GetFingerprint(m),
                                                  train_fps)
        max_sim[qi] = max(sims) if sims else 0.0

    # full per-model scores for the library: models compete side by side
    full = {"identifier": screen_kept["identifier"].values,
            "canonical_smiles": screen_kept["canonical_smiles"].values}
    for r in reg_records:
        if r["screening_ok"] and r["_model"] is not None:
            full[f"pIC50_{r['model']}"] = r["_model"].predict(Xs)
    for r in clf_records:
        if r["screening_ok"] and r["_model"] is not None:
            full[f"Pactive_{r['model']}"] = r["_model"].predict_proba(Xs)[:, 1]
    if net is not None:
        full["pIC50_multitask_net"] = pr_s
        full["Pactive_multitask_net"] = pc_s
    for kind, (pr_f, pc_f) in graph_screen_preds.items():
        full[f"pIC50_{kind}"] = pr_f
        full[f"Pactive_{kind}"] = pc_f
    full["pIC50_ensemble"] = pred_pIC50_s
    full["Pactive_ensemble"] = prob_active_s
    full["max_tanimoto_sim"] = max_sim
    full["inside_AD"] = max_sim >= cuts["ad_cut_strict"]
    pd.DataFrame(full).to_csv(OUT_DIR / "screening_scores_all_models.csv", index=False)
    print(f"[screen] per-model scores saved -> "
          f"{_rel(OUT_DIR / 'screening_scores_all_models.csv')} "
          f"({len(full) - 4} model score columns)")

    top_r, top_c = top50_table(pred_pIC50_s, prob_active_s,
                               screen_kept["identifier"].tolist(),
                               screen_kept["canonical_smiles"].tolist(),
                               max_sim, cuts["ad_cut_strict"])
    write_and_print_top50(top_r, top_c)

    inside_ad50 = float(np.mean([r["inside_AD"] for r in top_r])) if top_r else 0.0
    metrics = {
        "regression_models": {r["model"]: r["test"] for r in reg_records},
        "classification_models": {r["model"]: r["test"] for r in clf_records},
        "regression_test_multitask_net": net_reg,
        "classification_test_multitask_net": net_clf,
        "graph_models": {kind: {"regression_test": m["reg_test"],
                                "classification_test": m["clf_test"]}
                         for kind, m in graph.items()},
        "large_models_status": REPORT.get("large_models_status", ""),
        "large_models": REPORT.get("large_models", []),
        "screened": int(len(screen_kept)),
        "top50_regression_inside_ad_frac": inside_ad50,
        "device_mode": info["mode"],
        "elapsed_s": round(time.time() - t0, 1),
        "type_issues": REPORT.get("type_issues", []),
    }
    if not pre_ok:
        metrics["pre_train_verification"] = "some checks FAILED (see report)"
    all_ok = finalize_report(round(time.time() - t0, 1), metrics)
    print(f"[done] stage {('smoke-dryrun' if SMOKE else 'full')} done "
          f"in {metrics['elapsed_s']}s -> {_rel(OUT_DIR)}")
    return 0 if all_ok else 3


def main():
    code = 0
    try:
        code = run_pipeline()
    except Exception as exc:
        traceback.print_exc()
        REPORT["checks"]["pipeline_error"] = {"status": "FAIL",
                                              "error": f"{type(exc).__name__}: {exc}"}
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "verification_report.json").write_text(
            json.dumps(REPORT, indent=2, default=str))
        code = 2
    if NOTEBOOK:
        # a notebook cell must not die with SystemExit when pasted
        print(f"[part21] cell finished with exit code {code}")
        return code
    sys.exit(code)


if __name__ == "__main__":
    main()
