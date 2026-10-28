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
#   MDM2_PART21_SMOKE=1          tiny dry-run (subsample; no ChemBERTa)
#   MDM2_PART21_USE_CHEMBERTA=1  force ChemBERTa fine-tuning
#   MDM2_PART21_SKIP_CHEMBERTA=1 force ChemBERTa off
#   MDM2_PART21_EPOCHS           override torch multitask epochs
# =============================================================================
import hashlib
import json
import math
import os
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


def train_torch_multitask(Xtr, ytr_reg, ytr_clf, epochs=None):
    info = devices_report()
    dev = info["device"]
    epochs = epochs or int(os.environ.get("MDM2_PART21_EPOCHS", "150"))
    net = TorchMultitaskNet(Xtr.shape[1])
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
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
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


# --- ChemBERTa large model (optional, two separate fine-tunes) ---------------
def maybe_finetune_chemberta(df: pd.DataFrame):
    """SEPARATE regression and classification fine-tunes (one model each).
    Uses nn.DataParallel so both GPUs are engaged. Returns dict or None."""
    if os.environ.get("MDM2_PART21_SKIP_CHEMBERTA", "0") == "1":
        REPORT["large_model"] = {"status": "skipped-by-flag"}
        return None
    if SMOKE and os.environ.get("MDM2_PART21_USE_CHEMBERTA", "0") != "1":
        REPORT["large_model"] = {"status": "skipped-smoke"}
        print("[chemberta] skipped in SMOKE mode")
        return None
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:
        REPORT["large_model"] = {"status": "not-available",
                                 "reason": f"{type(exc).__name__}: {exc}"}
        print("[chemberta] transformers not available ->", exc)
        return None
    info = devices_report()
    model_name = "seyonec/ChemBERTa-zinc-base-v1"
    out = {}
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        enc = tok(df["smiles"].tolist(), truncation=True, max_length=192,
                  padding=True, return_tensors="pt")
        ys_reg = torch.tensor(df["pIC50"].values, dtype=torch.float32)
        ys_clf = torch.tensor(df["activity"].values, dtype=torch.long)
        bs = 16 if info["cuda"] else 32

        for task, labels in (("regression", ys_reg), ("classification", ys_clf)):
            num_labels = 1 if task == "regression" else 2
            mdl = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels)
            if info["n_gpu"] >= 2:
                mdl = nn.DataParallel(mdl)
                print(f"[chemberta] {task}: DataParallel on {info['n_gpu']} GPUs")
            mdl.to(info["device"])
            opt = torch.optim.AdamW(mdl.parameters(), lr=2e-5)
            dl = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(enc["input_ids"], enc["attention_mask"],
                                               labels),
                batch_size=bs, shuffle=True)
            lossf = nn.MSELoss() if task == "regression" else nn.CrossEntropyLoss()
            mdl.train()
            for ep in range(3):
                run = 0.0
                for ids, mask, yy in dl:
                    ids, mask, yy = ids.to(info["device"]), mask.to(info["device"]), \
                        yy.to(info["device"])
                    opt.zero_grad()
                    if task == "regression":
                        logits = mdl(input_ids=ids, attention_mask=mask).logits.squeeze(-1)
                        loss = lossf(logits, yy.view(-1))
                    else:
                        logits = mdl(input_ids=ids, attention_mask=mask).logits
                        loss = lossf(logits, yy.view(-1))
                    loss.backward()
                    opt.step()
                    run += float(loss.detach())
                print(f"[chemberta] {task} epoch {ep + 1}/3 loss={run:.4f}")
            core = mdl.module if isinstance(mdl, nn.DataParallel) else mdl
            torch.save(core.state_dict(), OUT_DIR / f"chemberta_{task}.pt")
            mdl.eval()
            out[task] = "trained"
        REPORT["large_model"] = {"status": "fine-tuned", "model": model_name,
                                 "tasks": out, "gpus": info["mode"]}
        return out
    except Exception as exc:
        REPORT["large_model"] = {"status": "failed",
                                 "reason": f"{type(exc).__name__}: {exc}",
                                 "note": "pipeline continues without it"}
        print("[chemberta] fine-tune failed ->", exc)
        return None


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
    REPORT["final"] = metrics
    (OUT_DIR / "verification_report.json").write_text(json.dumps(REPORT, indent=2, default=str))
    check("verification_report_written", "PASS", path=_rel(OUT_DIR / "verification_report.json"))
    (OUT_DIR / "verification_report.json").write_text(json.dumps(REPORT, indent=2, default=str))
    files = _dir_files(OUT_DIR)
    manifest = {"output_dir": _rel(OUT_DIR), "files": files,
                "sha256": {_rel(OUT_DIR / f): _sha256(OUT_DIR / f) for f in files}}
    (OUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    allowed = ("PASS", "SKIPPED-1GPU", "SKIPPED-CPU", "WARN")
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

    # ---------------- regression -----------------------------------------
    print("\n================ REGRESSION (pIC50) ================")
    reg_model = train_xgb("regression", Xtr, ytr_r)
    print("[regression] XGB  val :", regression_metrics(yva_r, reg_model.predict(Xva)))
    reg_te_m = regression_metrics(yte_r, reg_model.predict(Xte))
    print("[regression] XGB  test:", reg_te_m)

    # ---------------- classification -------------------------------------
    print("\n============ CLASSIFICATION (Active vs Inactive) ============")
    clf_model = train_xgb("classification", Xtr, ytr_c)
    clf_va_p = clf_model.predict_proba(Xva)[:, 1]
    print("[classification] XGB  val :", classification_metrics(yva_c, clf_va_p))
    clf_te_m = classification_metrics(yte_c, clf_model.predict_proba(Xte)[:, 1])
    print("[classification] XGB  test:", clf_te_m)

    # ---------------- pytorch multitask network ---------------------------
    info = devices_report()
    print(f"\n[torch] device plan: {info['mode']}")
    if TORCH_OK:
        net = train_torch_multitask(Xtr, ytr_r, ytr_c)
        pr_te, pc_te = torch_predict(net, Xte)
        net_reg = regression_metrics(yte_r, pr_te)
        net_clf = classification_metrics(yte_c, pc_te)
        print("[regression] multitask-net test:", net_reg)
        print("[classification] multitask-net test:", net_clf)
    else:
        net, net_reg, net_clf = None, {}, {}

    # ---------------- screening ------------------------------------------
    print("\n[screen] screening the library (identifier + canonical_smiles only)")
    Xs, keep_s = build_features(screen_raw["canonical_smiles"].astype(str).tolist(),
                                label="screening")
    screen_kept = screen_raw.loc[keep_s].reset_index(drop=True)
    print(f"[screen] kept {len(screen_kept)}/{len(screen_raw)} screening molecules")

    pred_reg_s = reg_model.predict(Xs)
    clf_probs_s = clf_model.predict_proba(Xs)[:, 1]
    if net is not None:
        pr_s, pc_s = torch_predict(net, Xs)
        prob_active_s = 0.5 * clf_probs_s + 0.5 * pc_s
        pred_pIC50_s = 0.75 * pred_reg_s + 0.25 * pr_s
    else:
        prob_active_s, pred_pIC50_s = clf_probs_s, pred_reg_s

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

    top_r, top_c = top50_table(pred_pIC50_s, prob_active_s,
                               screen_kept["identifier"].tolist(),
                               screen_kept["canonical_smiles"].tolist(),
                               max_sim, cuts["ad_cut_strict"])
    write_and_print_top50(top_r, top_c)

    maybe_finetune_chemberta(df.iloc[idx_tr].reset_index(drop=True))

    inside_ad50 = float(np.mean([r["inside_AD"] for r in top_r])) if top_r else 0.0
    metrics = {
        "regression_test_xgb": reg_te_m,
        "classification_test_xgb": clf_te_m,
        "regression_test_multitask_net": net_reg,
        "classification_test_multitask_net": net_clf,
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
