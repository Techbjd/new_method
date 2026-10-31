"""Shared featurization for the traditional-ML layer (seed 57, leak-free)."""
from __future__ import annotations

import numpy as np
import pandas as pd

RANDOM_SEED = 57

TRAIN_URL_FALLBACKS = [
    ("../00_data/training/ro5_properties_filtered.csv", "local"),
    ("00_data/training/ro5_properties_filtered.csv", "local"),
    ("https://raw.githubusercontent.com/Techbjd/new_method/main/00_data/training/ro5_properties_filtered.csv", "github"),
]
LIB_URL_FALLBACKS = [
    ("../00_data/screening/screening_results.csv", "local"),
    ("00_data/screening/screening_results.csv", "local"),
    ("https://raw.githubusercontent.com/Techbjd/new_method/main/00_data/screening/screening_results.csv", "github"),
]

DESCRIPTORS = [
    "molecular_weight", "logp", "tpsa", "rotatable_bonds", "n_hba",
    "n_hbd", "heavy_atoms", "rings", "aromatic_rings", "fraction_sp3",
]

FINGERPRINTS = {
    "morgan": dict(radius=2, nbits=2048),      # ECFP4 (Rogers & Hahn 2010, JCIM)
    "maccs": dict(nbits=167),                  # RDKit MACCS 167 keys
    "rdk": dict(fpSize=2048, path_hash_len=4), # RDKit topological
    "atom_pair": dict(fpSize=2048),            # Carhart 1985 AtomPair
    "torsion": dict(fpSize=2048),              # Nilakantan 1987 TopologicalTorsion
    "avalon": dict(nBits=1024),                # Avalon (Gedeck 2006) — auto-skip if missing
}


def _with_rdkit():
    from rdkit import Chem, DataStructs
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors, rdFingerprintGenerator
    return Chem, DataStructs, Crippen, Descriptors, rdMolDescriptors, rdFingerprintGenerator


def load_training_table(path: str | None = None) -> pd.DataFrame:
    """Load the 645-compound RO5 table (cols: molecule_chembl_id, biomactivity_class, pIC50, canonical_smiles, ...)."""
    candidates = ([path] if path else [p for p, _ in TRAIN_URL_FALLBACKS])
    for cand in candidates:
        if cand.startswith("http"):
            try:
                return pd.read_csv(cand)
            except Exception:
                continue
        try:
            return pd.read_csv(cand)
        except Exception:
            continue
    raise FileNotFoundError("ro5_properties_filtered.csv not found locally or on GitHub raw")


def smiles_column(df: pd.DataFrame) -> pd.Series:
    col = "canonical_smiles" if "canonical_smiles" in df.columns else "SMILES"
    return df[col].astype(str).str.strip()


def label_encode(df: pd.DataFrame) -> np.ndarray:
    return (df["bioactivity_class"].astype(str) == "Active").astype(int).to_numpy()


def smiles_to_descriptors(smiles: list[str]) -> np.ndarray:
    """10 RDKit descriptors -> float matrix (NaN->0); invalid SMILES become all-zero rows."""
    Chem, _, Crippen, Descriptors, rdMolDescriptors, _ = _with_rdkit()
    cols = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            cols.append([0.0] * len(DESCRIPTORS)); continue
        cols.append([
            Descriptors.MolWt(mol), Crippen.MolLogP(mol), rdMolDescriptors.CalcTPSA(mol),
            Descriptors.NumRotatableBonds(mol), rdMolDescriptors.CalcNumHBA(mol),
            rdMolDescriptors.CalcNumHBD(mol), float(mol.GetNumHeavyAtoms()), float(rdMolDescriptors.CalcNumRings(mol)),
            float(rdMolDescriptors.CalcNumAromaticRings(mol)),
            Descriptors.FractionCSP3(mol),
        ])
    X = np.asarray(cols, dtype=np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def smiles_to_fingerprints(smiles: list[str], family: str = "morgan") -> np.ndarray:
    """One fingerprint family to a dense fp array (avalon auto-skipped if absent)."""
    Chem, DataStructs, _, _, _, _ = _with_rdkit()
    from rdkit.Chem import rdFingerprintGenerator as rfg
    cfg = FINGERPRINTS[family]
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    if family == "maccs":
        from rdkit.Chem import MACCSkeys
        fps = [MACCSkeys.GenMACCSKeys(m) for m in mols]
    elif family == "morgan":
        gen = rfg.GetMorganGenerator(radius=cfg["radius"], fpSize=cfg["nbits"])
        fps = [gen.GetFingerprint(m) for m in mols]
    elif family == "rdk":
        gen = rfg.GetRDKitFPGenerator(fpSize=cfg["fpSize"])
        fps = [gen.GetFingerprint(m) for m in mols]
    elif family == "atom_pair":
        gen = rfg.GetAtomPairGenerator(fpSize=cfg["fpSize"])
        fps = [gen.GetFingerprint(m) for m in mols]
    elif family == "torsion":
        gen = rfg.GetTopologicalTorsionGenerator(fpSize=cfg["fpSize"])
        fps = [gen.GetFingerprint(m) for m in mols]
    elif family == "avalon":
        try:
            from rdkit.Avalon import pyAvalonTools
            fps = [pyAvalonTools.GetAvalonFP(m, nBits=cfg["nBits"]) for m in mols]
        except Exception:
            return np.zeros((len(mols), 1), dtype=np.float32)
    else:
        raise KeyError(family)
    out = np.zeros((len(fps), fps[0].GetNumBits()), dtype=np.float32)
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, out[i])
    return out


def murcko_scaffold(smiles: str) -> str:
    """Bemis–Murcko scaffold SMILES (2013 JCIM 19(4):192; RDKit implementation)."""
    from rdkit.Chem.Scaffolds import MurckoScaffold
    Chem, _, _, _, _, _ = _with_rdkit()
    try:
        core = MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(smiles))
        return Chem.MolToSmiles(core) if core is not None else ""
    except Exception:
        return ""


def scaffold_split(smiles: list[str], y_reg, y_cls, test_frac=0.10, val_frac=0.10, seed=RANDOM_SEED):
    """Group-split by Murcko scaffold: 80/10/10, zero scaffold leakage (verified upstream)."""
    from sklearn.model_selection import GroupShuffleSplit
    groups = np.array([murcko_scaffold(s) for s in smiles])
    n = len(y_cls)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    train_val_idx, test_idx = next(gss.split(np.zeros(n), groups=groups))
    n_tv = len(train_val_idx)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=int(np.ceil(val_frac / (1 - test_frac) * n)), random_state=seed)
    train_idx, val_idx = next(gss2.split(np.zeros(n_tv), groups=np.asarray(groups)[train_val_idx]))
    train_idx = train_val_idx[train_idx]; val_idx = train_val_idx[val_idx]
    def _no_leak(a, b):
        return len(set(np.asarray(groups)[a]) & set(np.asarray(groups)[b])) == 0
    assert _no_leak(train_idx, val_idx) and _no_leak(train_idx, test_idx) and _no_leak(val_idx, test_idx)
    return train_idx, val_idx, test_idx


def five_fold_group_cv_indices(smiles: list[str], n_splits: int = 5, seed=RANDOM_SEED):
    """5-fold GroupKFold over Murcko scaffolds — the '5 fold scan' used everywhere."""
    from sklearn.model_selection import GroupKFold
    groups = [murcko_scaffold(s) for s in smiles]
    gkf = GroupKFold(n_splits=n_splits)
    for tr, va in gkf.split(np.zeros(len(groups)), groups=groups):
        yield tr, va


def load_screening_library(path: str | None = None) -> pd.DataFrame:
    """154k COCONUT library, structures-only (old prediction columns dropped)."""
    candidates = ([path] if path else [p for p, _ in LIB_URL_FALLBACKS])
    for cand in candidates:
        try:
            if cand.startswith("http"):
                df = pd.read_csv(cand)
            else:
                df = pd.read_csv(cand)
        except Exception:
            continue
        return df[["identifier", "canonical_smiles"]].copy()
    raise FileNotFoundError("screening_results.csv not found locally or on GitHub raw")
