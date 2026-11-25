#!/usr/bin/env python3
# [USER]/[GENERATED] validation_package.py -- created 2026-11-23
#
# Provenance:
#   [USER]      Scientific protocol, nested-CV design, enrichment/uncertainty
#               specification and conservative reporting rules.
#   [GENERATED] Single-file, CPU-feasible implementation and all files it writes
#               under outputs/validation_package/.
#
# IMPORTANT SCIENTIFIC CAVEATS -- READ BEFORE USING ANY NUMBER FROM THIS FILE:
#   1. Fast nested-CV AUC/AP/BalAcc/MCC values from the small classical models
#      below are SANITY CHECKS ONLY for the 645-compound training pool. They are
#      not validation of any screening hit and not prospective performance.
#   2. The official 129-compound scaffold-disjoint test set is NEVER used for
#      model selection, hyperparameter choice, threshold choice or early
#      stopping anywhere in this script. Model/threshold selection happens only
#      inside the inner folds carved from the 516 train+val compounds.
#   3. The existing DL_01..DL_05 "fast" test predictions are re-scored ONCE and
#      DESCRIPTIVELY (single split, already-finished runs, no retraining, no
#      architecture/hyperparameter selection). Their enrichment numbers must
#      not be used to pick a "best" model or to claim validated hits.
#   4. The official build_notebooks.py scaffold split leaves train and val
#      sharing some scaffolds (test is disjoint from both). That historical fact
#      is reported, not hidden; all CV here uses GroupKFold on scaffolds.
#
# No network access. No writes to source data, notebooks, build_notebooks.py,
# docs or any archive folder. All outputs go to outputs/validation_package/.

"""Reproducible validation package for the MDM2 natural-product project.

Run from this directory:
    python3 validation_package.py

Optional environment overrides:
    VALID_REPEATS=3   number of seeded outer repetitions
    VALID_FOLDS=5     number of outer GroupKFold folds
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

# Keep thread use modest and deterministic on an 8-CPU machine; set before
# numpy/sklearn are imported so their thread pools honour the value.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Scoring import Scoring
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.warning")

HERE = Path(__file__).resolve().parent
DATA_CSV = HERE.parent / "00_data" / "training" / "ro5_properties_filtered.csv"
FAST_OUTPUTS = HERE / "outputs"
OUT_DIR = FAST_OUTPUTS / "validation_package"
# The default-23 fast predictions are preserved here before any extended-roster
# run overwrites outputs/dl0X/fast/. Re-running this script therefore reproduces
# the documented enrichment numbers regardless of later notebook runs.
SNAPSHOT_DIR = OUT_DIR / "source_fast_predictions"

RANDOM_SEED = 57  # project convention, identical to build_notebooks.py
TEST_SIZE, VAL_FRAC = 0.20, 0.10
N_REPEATS = int(os.environ.get("VALID_REPEATS", "3"))
N_FOLDS = int(os.environ.get("VALID_FOLDS", "5"))
N_INNER = 3
BOOT_N = 2000
BOOT_SEED = 57
ALPHA_BEDROC = 20.0

DATE = "2026-11-23"


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
def say(msg: str) -> None:
    print(msg, flush=True)


def rel_str(path: Path) -> str:
    """Readable path for logs/metadata, safe even for paths outside this repo."""
    try:
        return str(path.relative_to(HERE))
    except ValueError:
        return str(path)


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2 or not np.isfinite(p).all():
        return float("nan")
    return float(roc_auc_score(y, p))


def safe_ap(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2 or not np.isfinite(p).all():
        return float("nan")
    return float(average_precision_score(y, p))


def metrics_at_threshold(y: np.ndarray, p: np.ndarray, thr: float) -> tuple[float, float]:
    yhat = (p >= thr).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    return (
        float(balanced_accuracy_score(y, yhat)),
        float(matthews_corrcoef(y, yhat)),
    )


def mean_std(vals: list[float]) -> dict:
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": float(np.mean(v)),
        "std": float(np.std(v, ddof=0)),
        "n": int(v.size),
    }


def seeded_group_codes(groups: np.ndarray, seed: int) -> np.ndarray:
    """Permute the *order* of distinct scaffold labels, never their identity.

    sklearn GroupKFold is deterministic for identical group labels. Remapping
    labels through a seeded permutation gives reproducible, different fold
    compositions while keeping every scaffold entirely inside one fold.
    """
    uniq = np.unique(groups)
    perm = np.random.RandomState(seed).permutation(len(uniq))
    mapping = {g: int(i) for g, i in zip(uniq, perm)}
    return np.asarray([mapping[g] for g in groups], dtype=int)


# ---------------------------------------------------------------------------
# 1. EXACT leak-free scaffold split copied from build_notebooks.py SPLIT_SRC
# ---------------------------------------------------------------------------
def murcko(smi: str) -> str:
    # RDKit 2026 API: keyword mol= required (Part_20 fix). Whole scaffolds
    # never straddle splits.
    try:
        m = Chem.MolFromSmiles(str(smi))
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m is not None else "INVALID"
    except Exception:
        return "INVALID"


def canon(smi: str) -> str:
    m = Chem.MolFromSmiles(str(smi))
    return Chem.MolToSmiles(m) if m is not None else str(smi)


def load_official_split():
    df = pd.read_csv(DATA_CSV)
    smiles_all = df["canonical_smiles"].astype(str).tolist()
    y_cls_all = (df["bioactivity_class"].astype(str).str.title() == "Active").astype(int).to_numpy()

    valid = np.array([Chem.MolFromSmiles(s) is not None for s in smiles_all])
    if not valid.all():
        keep = np.where(valid)[0]
        smiles_all = [smiles_all[i] for i in keep]
        y_cls_all = y_cls_all[keep]
        say(f"[SPLIT] dropped {int((~valid).sum())} unparseable rows")

    scaffolds_all = np.array([murcko(s) for s in smiles_all])
    n_uniq = len(set(scaffolds_all))
    say(f"[SPLIT] molecules={len(scaffolds_all)} unique Murcko scaffolds={n_uniq} "
        f"INVALID={int((scaffolds_all == 'INVALID').sum())}")
    assert n_uniq >= 50, "[SPLIT] scaffold collapse detected - Murcko API may have changed!"

    groups = defaultdict(list)
    for _i, _s in enumerate(scaffolds_all):
        groups[_s].append(_i)

    n_test_target = int(round(TEST_SIZE * len(scaffolds_all)))
    order = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))  # largest scaffolds first
    te_list = []
    for _s, _ii in order:
        if len(te_list) + len(_ii) <= n_test_target:
            te_list.extend(_ii)
    test_idx = np.array(sorted(te_list), dtype=int)
    tr_all = np.array(sorted(set(range(len(scaffolds_all))) - set(te_list)), dtype=int)
    tr_idx, va_idx = train_test_split(
        tr_all, test_size=VAL_FRAC, random_state=RANDOM_SEED, stratify=y_cls_all[tr_all]
    )
    tr_idx, va_idx = np.array(tr_idx), np.array(va_idx)

    assert (len(tr_idx), len(va_idx), len(test_idx)) == (464, 52, 129), (
        "official 464/52/129 scaffold split was not reproduced"
    )
    sets = [set(tr_idx.tolist()), set(va_idx.tolist()), set(test_idx.tolist())]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2]), \
        "index overlap between splits"
    scaffold_sets = [
        {scaffolds_all[i] for i in sets[0]},
        {scaffolds_all[i] for i in sets[1]},
        {scaffolds_all[i] for i in sets[2]},
    ]
    ov_tr_te = len(scaffold_sets[0] & scaffold_sets[2])
    ov_va_te = len(scaffold_sets[1] & scaffold_sets[2])
    ov_tr_va = len(scaffold_sets[0] & scaffold_sets[1])
    assert ov_tr_te == 0 and ov_va_te == 0, "scaffold overlap with official test set"

    canon_tr = {canon(smiles_all[i]) for i in tr_idx}
    canon_te = {canon(smiles_all[i]) for i in test_idx}
    assert not (canon_tr & canon_te), "canonical SMILES overlap between train and test!"

    tr_va_idx = np.array(sorted(sets[0] | sets[1]), dtype=int)
    ids = df["molecule_chembl_id"].astype(str).to_numpy()
    say(f"[SPLIT] train {len(tr_idx)} | val {len(va_idx)} | test {len(test_idx)} | "
        f"scaffold overlap train-test={ov_tr_te}, val-test={ov_va_te}, train-val={ov_tr_va} "
        f"(train-val overlap is inherited from the official split; ALL CV below is GroupKFold)")
    say("[SPLIT] official test set is NEVER used for selection in this script")

    split_meta = {
        "source_csv": str(DATA_CSV.relative_to(HERE.parent)),
        "algorithm": "build_notebooks.py SPLIT_SRC (seed 57), replicated verbatim",
        "n_total": int(len(scaffolds_all)),
        "unique_murcko_scaffolds": int(n_uniq),
        "n_train": int(len(tr_idx)),
        "n_val": int(len(va_idx)),
        "n_test": int(len(test_idx)),
        "n_actives_train_val_test": [
            int(y_cls_all[tr_idx].sum()),
            int(y_cls_all[va_idx].sum()),
            int(y_cls_all[test_idx].sum()),
        ],
        "scaffold_overlap_train_test": ov_tr_te,
        "scaffold_overlap_val_test": ov_va_te,
        "scaffold_overlap_train_val": ov_tr_va,
        "train_val_pool_for_cv": int(len(tr_va_idx)),
        "official_test_ids": sorted(ids[test_idx].tolist()),
    }
    return (
        smiles_all, y_cls_all, scaffolds_all, tr_idx, va_idx, test_idx, tr_va_idx, split_meta
    )


# ---------------------------------------------------------------------------
# 2. Features: ECFP4 (radius 2, 2048 bits) and a small descriptor block
# ---------------------------------------------------------------------------
DESCRIPTORS = {
    "MW": Descriptors.MolWt,
    "LogP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA,
    "HBD": rdMolDescriptors.CalcNumHBD,
    "HBA": rdMolDescriptors.CalcNumHBA,
    "RotB": rdMolDescriptors.CalcNumRotatableBonds,
    "Rings": rdMolDescriptors.CalcNumRings,
    "AromRings": rdMolDescriptors.CalcNumAromaticRings,
    "HeavyAtoms": Descriptors.HeavyAtomCount,
    "FracCsp3": Descriptors.FractionCSP3,
    "MolMR": Descriptors.MolMR,
}


def build_features(smiles_list):
    t0 = time.time()
    ecfp = np.zeros((len(smiles_list), 2048), dtype=np.float32)
    desc = np.zeros((len(smiles_list), len(DESCRIPTORS)), dtype=np.float64)
    for i, s in enumerate(smiles_list):
        m = Chem.MolFromSmiles(str(s))
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048)
        DataStructs.ConvertToNumpyArray(fp, ecfp[i])
        for j, fn in enumerate(DESCRIPTORS.values()):
            try:
                v = float(fn(m))
                desc[i, j] = v if math.isfinite(v) else 0.0
            except Exception:
                desc[i, j] = 0.0
    say(f"[FEAT] ECFP4 2048-bit + {len(DESCRIPTORS)} descriptors built in "
        f"{time.time() - t0:.1f}s")
    return ecfp, desc


# ---------------------------------------------------------------------------
# 3. Repeated nested scaffold CV on train+val only
# ---------------------------------------------------------------------------
def make_model(family: str, params: dict):
    if family == "ecfp_logreg":
        return LogisticRegression(
            C=params["C"], solver="liblinear", max_iter=2000,
            class_weight="balanced", random_state=RANDOM_SEED,
        )
    if family == "ecfp_random_forest":
        return RandomForestClassifier(
            n_estimators=150, min_samples_leaf=params["min_samples_leaf"],
            max_features="sqrt", class_weight="balanced_subsample",
            n_jobs=2, random_state=RANDOM_SEED,
        )
    if family == "descriptor_logreg":
        return LogisticRegression(
            C=params["C"], solver="liblinear", max_iter=2000,
            class_weight="balanced", random_state=RANDOM_SEED,
        )
    raise ValueError(f"unknown model family: {family}")


CONFIGS = {
    "ecfp_logreg": [{"C": 0.1}, {"C": 1.0}, {"C": 10.0}],
    "ecfp_random_forest": [{"min_samples_leaf": 1}, {"min_samples_leaf": 4}],
    "descriptor_logreg": [{"C": 0.1}, {"C": 1.0}, {"C": 10.0}],
}


def choose_threshold(y_oof: np.ndarray, p_oof: np.ndarray) -> float:
    """Validation-only threshold: maximise balanced accuracy on inner OOF."""
    grid = np.linspace(0.05, 0.95, 91)
    best = (-1.0, 0.5)
    for thr in grid:
        ba, _ = metrics_at_threshold(y_oof, p_oof, float(thr))
        score = -1.0 if not np.isfinite(ba) else float(ba)
        # deterministic tie-break: threshold closest to 0.5
        tie = -abs(float(thr) - 0.5)
        if (score, tie) > (best[0], -abs(best[1] - 0.5)):
            best = (score, float(thr))
    return best[1]


def tune_family(family, configs, x_train, y_train, g_train, inner_seed):
    """Inner GroupKFold tuning; returns best config, threshold and inner scores."""
    inner_codes = seeded_group_codes(g_train, inner_seed)
    splits = list(GroupKFold(n_splits=N_INNER).split(x_train, y_train, inner_codes))
    results = []
    oof_cache = {}
    for params in configs:
        oof_p = np.full(len(y_train), np.nan)
        for it, iv in splits:
            xit, xiv = x_train[it], x_train[iv]
            if family == "descriptor_logreg":
                sc = StandardScaler().fit(xit)
                xit, xiv = sc.transform(xit), sc.transform(xiv)
            model = make_model(family, params)
            model.fit(xit, y_train[it])
            oof_p[iv] = model.predict_proba(xiv)[:, 1]
        aucs, aps = [], []
        for it, iv in splits:
            aucs.append(safe_auc(y_train[iv], oof_p[iv]))
            aps.append(safe_ap(y_train[iv], oof_p[iv]))
        rec = {
            "params": params,
            "oof_p": oof_p,
            "inner_auc_mean": float(np.nanmean(aucs)) if np.isfinite(aucs).any() else float("nan"),
            "inner_auc_std": float(np.nanstd(aucs)) if np.isfinite(aucs).any() else float("nan"),
            "inner_ap_mean": float(np.nanmean(aps)) if np.isfinite(aps).any() else float("nan"),
            "inner_ap_std": float(np.nanstd(aps)) if np.isfinite(aps).any() else float("nan"),
        }
        results.append(rec)
        oof_cache[id(params)] = oof_p

    def key(r):
        a = r["inner_auc_mean"]
        ap = r["inner_ap_mean"]
        return (-1.0 if not np.isfinite(a) else a,
                -1.0 if not np.isfinite(ap) else ap)

    best = max(results, key=key)
    thr = choose_threshold(y_train, best["oof_p"])
    return best, thr


def run_nested_cv(smiles, y_all, scaffolds, tr_va_idx, split_meta):
    t0 = time.time()
    x_fp, x_desc = build_features([smiles[i] for i in tr_va_idx])
    y = y_all[tr_va_idx]
    groups = scaffolds[tr_va_idx]
    rows = []

    for rep in range(N_REPEATS):
        seed = RANDOM_SEED + rep
        outer_codes = seeded_group_codes(groups, seed)
        outer = GroupKFold(n_splits=N_FOLDS).split(x_fp, y, outer_codes)
        for fold, (fit_rel, test_rel) in enumerate(outer, start=1):
            xf_fp, xt_fp = x_fp[fit_rel], x_fp[test_rel]
            xf_desc, xt_desc = x_desc[fit_rel], x_desc[test_rel]
            yf, yt = y[fit_rel], y[test_rel]
            gf = groups[fit_rel]
            fold_scaffolds = set(gf) & set(groups[test_rel])
            assert not fold_scaffolds, "outer GroupKFold scaffold leakage"
            inner_seed = 10_000 * seed + fold

            tuned = {}
            for family, configs in CONFIGS.items():
                x_fam = x_desc if family == "descriptor_logreg" else x_fp
                best, thr = tune_family(family, configs, x_fam[fit_rel], yf, gf, inner_seed)
                tuned[family] = (best, thr)

            # Family choice is made ONLY from inner validation AUC (tie: AP).
            def family_key(fam):
                r = tuned[fam][0]
                a = r["inner_auc_mean"]
                ap = r["inner_ap_mean"]
                return (-1.0 if not np.isfinite(a) else a,
                        -1.0 if not np.isfinite(ap) else ap)

            # deterministic alphabetical tie-break on family name
            best_families = sorted(CONFIGS)
            selected = max(best_families, key=lambda f: (family_key(f), -best_families.index(f)))

            for family in best_families:
                best, thr = tuned[family]
                xf = x_desc[fit_rel] if family == "descriptor_logreg" else x_fp[fit_rel]
                xt = x_desc[test_rel] if family == "descriptor_logreg" else x_fp[test_rel]
                scaler = None
                if family == "descriptor_logreg":
                    scaler = StandardScaler().fit(xf)
                    xf, xt = scaler.transform(xf), scaler.transform(xt)
                model = make_model(family, best["params"])
                model.fit(xf, yf)
                p = model.predict_proba(xt)[:, 1]
                auc, ap = safe_auc(yt, p), safe_ap(yt, p)
                ba, mcc = metrics_at_threshold(yt, p, thr)
                rows.append({
                    "repeat": rep + 1,
                    "seed": seed,
                    "outer_fold": fold,
                    "model": family,
                    "selected_by_inner_only": family == selected,
                    "chosen_params": json.dumps(best["params"], sort_keys=True),
                    "threshold_from_inner_oof": round(float(thr), 4),
                    "inner_auc_mean": round(best["inner_auc_mean"], 6),
                    "inner_auc_std": round(best["inner_auc_std"], 6),
                    "inner_ap_mean": round(best["inner_ap_mean"], 6),
                    "inner_ap_std": round(best["inner_ap_std"], 6),
                    "n_outer_fit": int(len(yf)),
                    "n_outer_eval": int(len(yt)),
                    "n_pos_outer_fit": int(yf.sum()),
                    "n_pos_outer_eval": int(yt.sum()),
                    "outer_scaffold_overlap": len(fold_scaffolds),
                    "AUC": round(auc, 6) if np.isfinite(auc) else None,
                    "AP": round(ap, 6) if np.isfinite(ap) else None,
                    "BalAcc": round(ba, 6) if np.isfinite(ba) else None,
                    "MCC": round(mcc, 6) if np.isfinite(mcc) else None,
                })
        say(f"[CV] repeat {rep + 1}/{N_REPEATS} done ({time.time() - t0:.1f}s elapsed)")

    folds_df = pd.DataFrame(rows)
    summary = {
        "status": "SANITY CHECK ONLY - fast AUCs are not validation of hits",
        "date": DATE,
        "data_scope": "516 train+val compounds only; official 129-compound test set untouched",
        "protocol": {
            "outer": f"GroupKFold by Murcko scaffold, {N_FOLDS} folds x {N_REPEATS} seeded repeats "
                     "(seeds 57,58,59; scaffold-label permutation keeps folds disjoint)",
            "inner": f"GroupKFold by Murcko scaffold, {N_INNER} folds; hyperparameter and "
                     "probability-threshold selection from inner OOF AUC (tie: AP) only",
            "models": list(CONFIGS),
            "ecfp": "Morgan ECFP4 radius=2, nBits=2048 (RDKit)",
            "descriptors": list(DESCRIPTORS),
            "rf": "150 trees, sqrt features, balanced_subsample, random_state=57, n_jobs=2",
            "threshold_grid": "0.05..0.95 step 0.01, maximise inner-OOF balanced accuracy",
        },
        "split_used": {k: v for k, v in split_meta.items() if k != "official_test_ids"},
        "n_fold_rows": int(len(folds_df)),
        "per_model_all_outer_folds": {},
        "selection_counts_inner_only": {},
        "selected_model_across_outer_folds": {},
        "runtime_seconds": round(time.time() - t0, 1),
        "caveats": [
            "Outer-fold metrics are computed on train+val scaffolds only; no test compound "
            "or test label entered any fit or selection step.",
            "Non-selected models are evaluated on outer folds for diagnostics only; that "
            "information was NOT used to choose a final model.",
            "Train and val share scaffolds in the official split (see split_used); "
            "GroupKFold is used throughout CV to avoid scaffold leakage inside folds.",
            "No calibration, no external validation, no hit validation is claimed.",
        ],
    }
    for fam, grp in folds_df.groupby("model"):
        summary["per_model_all_outer_folds"][fam] = {
            m: mean_std(grp[m].tolist()) for m in ["AUC", "AP", "BalAcc", "MCC"]
        }
    counts = folds_df[folds_df["selected_by_inner_only"]]["model"].value_counts().to_dict()
    summary["selection_counts_inner_only"] = {str(k): int(v) for k, v in counts.items()}
    sel = folds_df[folds_df["selected_by_inner_only"]]
    for fam, grp in sel.groupby("model"):
        summary["selected_model_across_outer_folds"][fam] = {
            m: mean_std(grp[m].tolist()) for m in ["AUC", "AP", "BalAcc", "MCC"]
        }
    return folds_df, summary


# ---------------------------------------------------------------------------
# 4. DESCRIPTIVE enrichment + bootstrap CIs on EXISTING fast test predictions
# ---------------------------------------------------------------------------
def bedroc_from_ranked(y_ranked: np.ndarray) -> float:
    """RDKit CalcBEDROC on an already score-sorted label list (best first)."""
    y_list = [[int(v)] for v in y_ranked]
    try:
        v = float(Scoring.CalcBEDROC(y_list, 0, ALPHA_BEDROC))
        if not math.isfinite(v):
            return float("nan")
        return float(min(1.0, max(0.0, v)))  # clip tiny floating-point overshoot at 1.0
    except Exception:
        return float("nan")


def enrichment_metrics(y: np.ndarray, p: np.ndarray, ids: np.ndarray) -> dict:
    order = np.lexsort((ids.astype(str), -p))  # deterministic tie-break
    ys, ps = y[order], p[order]
    n, npos = len(ys), int(ys.sum())
    k1 = max(1, int(math.ceil(0.01 * n)))
    k10 = min(10, n)
    hits10 = int(ys[:k10].sum())
    hits1 = int(ys[:k1].sum())
    return {
        "n_test": n,
        "n_actives": npos,
        "n_inactives": n - npos,
        "AUC": safe_auc(ys, ps),
        "AP": safe_ap(ys, ps),
        "precision_at_10": float(hits10 / k10),
        "hits_at_10": hits10,
        "k_1pct": k1,
        "hits_at_1pct": hits1,
        "EF_1pct": float((hits1 / k1) / (npos / n)) if npos > 0 else float("nan"),
        "BEDROC_alpha20": bedroc_from_ranked(ys),
        "_sorted_y": ys,
        "_sorted_p": ps,
    }


def bootstrap_cis(y: np.ndarray, p: np.ndarray, ids: np.ndarray, base: dict) -> dict:
    """Paired nonparametric bootstrap of labeled rows; fixed seed, no refitting."""
    rng = np.random.default_rng(BOOT_SEED)
    n = len(y)
    draws = rng.integers(0, n, size=(BOOT_N, n))
    ids_arr = ids.astype(str)
    aucs, p10s, beds = [], [], []
    y_sorted_base = base["_sorted_y"]
    for b in range(BOOT_N):
        idx = draws[b]
        yy, pp = y[idx], p[idx]
        if len(np.unique(yy)) < 2:
            continue
        ord_ = np.lexsort((ids_arr[idx], -pp))
        ys = yy[ord_]
        aucs.append(float(roc_auc_score(ys, pp[ord_])))
        p10s.append(float(ys[:min(10, n)].sum() / min(10, n)))
        if len(y_sorted_base) == n:
            beds.append(bedroc_from_ranked(ys))

    def ci(vals):
        v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
        if v.size < 50:
            return (float("nan"), float("nan"), int(v.size))
        return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), int(v.size))

    a_lo, a_hi, n_a = ci(aucs)
    p_lo, p_hi, n_p = ci(p10s)
    b_lo, b_hi, n_b = ci(beds)
    return {
        "bootstrap_n_requested": BOOT_N,
        "bootstrap_seed": BOOT_SEED,
        "AUC_ci95": [a_lo, a_hi],
        "AUC_ci_valid_draws": n_a,
        "precision_at_10_ci95": [p_lo, p_hi],
        "precision_at_10_ci_valid_draws": n_p,
        "BEDROC_alpha20_ci95": [b_lo, b_hi],
        "BEDROC_ci_valid_draws": n_b,
    }


def run_fast_enrichment(official_test_ids):
    t0 = time.time()
    out_rows, warnings_list, grouped = [], [], {}
    official = set(official_test_ids)
    exclude = {"identifier", "canonical_smiles", "y_true", "prediction", "prob_class_0"}
    score_re = re.compile(r"(prob|score|p_?active|likelihood)", re.I)

    for n in range(1, 6):
        nb = f"dl{n:02d}"
        folder = FAST_OUTPUTS / nb / "fast"
        snap_pred = SNAPSHOT_DIR / f"{nb}_test_predictions.csv"
        snap_res = SNAPSHOT_DIR / f"{nb}_results.json"
        pred_path = snap_pred if snap_pred.exists() else folder / "test_predictions.csv"
        res_path = snap_res if snap_res.exists() else folder / f"{nb}_results.json"
        if not pred_path.exists():
            msg = f"[ENRICH][WARN] {nb}: missing {rel_str(pred_path)} - skipping"
            say(msg)
            warnings_list.append(msg)
            grouped[nb] = {"status": "skipped", "reason": "missing test_predictions.csv"}
            continue
        try:
            df = pd.read_csv(pred_path)
            required = {"y_true"}
            if not required.issubset(df.columns):
                raise ValueError(f"missing required column(s): {sorted(required - set(df.columns))}")
            label_raw = df["y_true"]
            if label_raw.dtype == object:
                y = (label_raw.astype(str).str.strip().str.title() == "Active").astype(int).to_numpy()
            else:
                y = label_raw.astype(int).to_numpy()
            if not set(np.unique(y)).issubset({0, 1}):
                raise ValueError("y_true is not binary 0/1 or Active/Inactive")

            score_cols = [
                c for c in df.columns
                if c not in exclude and score_re.search(str(c))
                and pd.api.types.is_numeric_dtype(df[c])
            ]
            if not score_cols:
                raise ValueError(
                    f"no probability/score column found (columns={list(df.columns)}); "
                    "expected e.g. prob_class_1"
                )

            meta = {}
            if res_path.exists():
                with open(res_path, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
            else:
                msg = f"[ENRICH][WARN] {nb}: missing {res_path.name}; using CSV columns only"
                say(msg)
                warnings_list.append(msg)

            final_model = meta.get("final_model") if isinstance(meta, dict) else None
            pred_ids = (df["identifier"].astype(str).to_numpy()
                        if "identifier" in df.columns else np.arange(len(df)).astype(str))
            id_match = len(set(pred_ids.tolist()) & official)

            nb_entries = []
            for col in score_cols:
                p_raw = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
                ok = np.isfinite(p_raw) & np.isfinite(y)
                if ok.sum() < len(df):
                    msg = (f"[ENRICH][WARN] {nb}/{col}: dropping {int((~ok).sum())} rows with "
                           "non-finite label/score")
                    say(msg)
                    warnings_list.append(msg)
                if len(np.unique(y[ok])) < 2:
                    msg = f"[ENRICH][WARN] {nb}/{col}: single-class labels; AUC/BEDROC undefined"
                    say(msg)
                    warnings_list.append(msg)
                    continue
                yy, pp, ii = y[ok], p_raw[ok], pred_ids[ok]
                base = enrichment_metrics(yy, pp, ii)
                cis = bootstrap_cis(yy, pp, ii, base)
                label = f"{final_model} [{col}]" if final_model and len(score_cols) == 1 else col
                rec = {
                    "notebook": nb,
                    "model_label": label,
                    "score_column": col,
                    "final_model_from_results_json": final_model if isinstance(final_model, str) else None,
                    "results_json_seed": meta.get("seed") if isinstance(meta, dict) else None,
                    "results_json_split_sizes": json.dumps(meta.get("split_sizes", {}), sort_keys=True)
                        if isinstance(meta, dict) else None,
                    "results_json_test_metrics": json.dumps(meta.get("test_metrics", {}), sort_keys=True)
                        if isinstance(meta, dict) else None,
                    "n_ids_matching_official_test": id_match,
                    "n_actives": base["n_actives"],
                    "n_inactives": base["n_inactives"],
                    "n_test": base["n_test"],
                    "AUC": base["AUC"],
                    "AP": base["AP"],
                    "precision_at_10": base["precision_at_10"],
                    "hits_at_10": base["hits_at_10"],
                    "k_1pct": base["k_1pct"],
                    "hits_at_1pct": base["hits_at_1pct"],
                    "EF_1pct": base["EF_1pct"],
                    "BEDROC_alpha20": base["BEDROC_alpha20"],
                    **{k: v for k, v in cis.items()},
                    "descriptive_only": True,
                    "used_for_model_selection": False,
                }
                out_rows.append(rec)
                nb_entries.append({k: v for k, v in rec.items() if not k.startswith("_")})
            grouped[nb] = {
                "status": "scored_descriptively" if nb_entries else "skipped",
                "prediction_file": rel_str(pred_path),
                "results_json": rel_str(res_path) if res_path.exists() else None,
                "rows": nb_entries,
            }
            say(f"[ENRICH] {nb}: scored {len(nb_entries)} score column(s) "
                f"({time.time() - t0:.1f}s elapsed)")
        except Exception as exc:  # robustness: never crash on an unexpected schema
            msg = f"[ENRICH][WARN] {nb}: skipped ({type(exc).__name__}: {exc})"
            say(msg)
            warnings_list.append(msg)
            grouped[nb] = {"status": "skipped", "reason": str(exc)}

    cols = [
        "notebook", "model_label", "score_column", "final_model_from_results_json",
        "results_json_seed", "results_json_split_sizes", "results_json_test_metrics",
        "n_ids_matching_official_test", "n_actives", "n_inactives", "n_test",
        "AUC", "AP", "precision_at_10", "hits_at_10", "k_1pct", "hits_at_1pct",
        "EF_1pct", "BEDROC_alpha20",
        "AUC_ci95", "AUC_ci_valid_draws",
        "precision_at_10_ci95", "precision_at_10_ci_valid_draws",
        "BEDROC_alpha20_ci95", "BEDROC_ci_valid_draws",
        "bootstrap_n_requested", "bootstrap_seed",
        "descriptive_only", "used_for_model_selection",
    ]
    enrich_df = pd.DataFrame(out_rows, columns=cols)
    summary = {
        "status": "DESCRIPTIVE ONLY - single official test split, already-existing fast predictions",
        "date": DATE,
        "what_was_done": "Re-scored existing test_predictions.csv files; no retraining, "
                         "no model/hyperparameter/threshold selection, no test-set tuning.",
        "caveats": [
            "Test was scored once descriptively; these numbers are NOT prospective and "
            "NOT validation of any hit.",
            "No 'best' model is chosen here. Do not rank notebooks by these values.",
            "1% of 129 compounds is ceil(1.29)=2 compounds, so EF1% is extremely "
            "unstable; bootstrap CIs are wide and should be read as uncertainty, not "
            "evidence of superiority.",
            "BEDROC(alpha=20) is RDKit CalcBEDROC on score-sorted labels with "
            "identifier tie-breaking.",
            "Metadata copied from each dl0X_results.json is reported as-is and was "
            "not used for selection.",
        ],
        "per_notebook": grouped,
        "warnings": warnings_list,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    return enrich_df, summary


# ---------------------------------------------------------------------------
# 5. README for the output folder
# ---------------------------------------------------------------------------
def write_readme(cv_summary, enr_summary, elapsed):
    cv_sel = cv_summary.get("selected_model_across_outer_folds", {})
    sel_txt = ", ".join(
        f"{k}: n={v['AUC']['n']}" for k, v in cv_sel.items()
    ) or "none"
    readme = f"""# Validation package -- generated {DATE}

[GENERATED] by `06_deep_learning_models/validation_package.py` in {elapsed:.1f} s.

## Scientific status / caveats

- **Fast nested-CV AUC/AP/BalAcc/MCC numbers are sanity checks only.** They are
  not validation of hits and not prospective performance.
- **The official 129-compound scaffold-disjoint test set was never used for
  selection.** All hyperparameter and threshold choices were made in inner
  GroupKFold folds drawn only from the 516 train+val compounds. The test was
  touched only to *re-score already-existing* fast prediction CSVs, once,
  descriptively.
- **No external test set exists yet.** There is no prospective validation set
  in this repository. Nothing here upgrades a predicted active to a hit.
- The official split leaves train and val sharing scaffolds (test is disjoint
  from both). This is documented in `nested_cv_summary.json`; all CV folds here
  use Murcko-scaffold GroupKFold.

## What WAS run

1. Rebuilt the exact 464/52/129 Murcko scaffold split from
   `build_notebooks.py SPLIT_SRC` (seed 57) and asserted test isolation.
2. Repeated nested scaffold CV on train+val only:
   {cv_summary['protocol']['outer']}
   {cv_summary['protocol']['inner']}
   Models: {', '.join(cv_summary['protocol']['models'])}.
   Inner selection counts: {json.dumps(cv_summary['selection_counts_inner_only'], sort_keys=True)}.
   Selected-model fold counts: {sel_txt}.
3. Re-scored the preserved default-23 snapshots in
   `outputs/validation_package/source_fast_predictions/` (falling back to
   `outputs/dl01..dl05/fast/test_predictions.csv` only if a snapshot is absent;
   no retraining): number of test actives, ROC AUC, precision@10, EF1%
   (top {2} of 129), BEDROC(alpha=20), and 2000-draw fixed-seed bootstrap 95%
   CIs for AUC, precision@10 and BEDROC.
4. Wrote this README plus the CSV/JSON files listed below.

## What was NOT run

- No retraining or re-tuning of any DL_01..DL_05 deep-learning model.
- No new scoring of the official test set for the purpose of choosing a model;
  no comparison-based "best model" selection from test enrichment.
- No external/prospective test set, no experimental assay, no applicability
  domain expansion, no hit validation claim.
- No changes to `build_notebooks.py`, notebooks, docs, source data or archives.

## Exact commands

```bash
cd "{HERE}"
python3 validation_package.py

# optional, e.g. 2 repeats x 3 outer folds
VALID_REPEATS=2 VALID_FOLDS=3 python3 validation_package.py
```

Dependencies: installed numpy, pandas, scikit-learn and rdkit only. No network.

## Outputs (all under `outputs/validation_package/`)

- `nested_cv_folds.csv` -- one row per (repeat, outer fold, model family);
  includes inner AUC/AP, the inner-only-selected flag, chosen parameters and
  threshold, and outer-fold AUC/AP/BalAcc/MCC.
- `nested_cv_summary.json` -- mean +/- std per model across outer folds, plus
  selected-by-inner-only summary (sanity check only).
- `fast_test_enrichment.csv` -- one row per available score column of each
  existing fast test prediction file, with point estimates and bootstrap CIs;
  every row is flagged `descriptive_only=True`, `used_for_model_selection=False`.
- `fast_test_enrichment_summary.json` -- grouped by notebook, including
  warnings for skipped files/schemas.
- `README.md` -- this file.

## Honest limitations

- If a prediction CSV has an unexpected schema it is skipped with a printed
  warning and recorded in `fast_test_enrichment_summary.json`.
- AUC/BEDROC are undefined for single-class label vectors and are reported as
  empty (NaN) rather than fabricated.
- EF1% on 129 compounds uses only 2 top-ranked molecules; treat it as a
  noisy descriptive statistic.
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    say(f"[MAIN] {DATE} | VALID_REPEATS={N_REPEATS} VALID_FOLDS={N_FOLDS} | no test-set selection")

    (smiles, y_all, scaffolds, tr_idx, va_idx, test_idx, tr_va_idx, split_meta) = \
        load_official_split()

    folds_df, cv_summary = run_nested_cv(smiles, y_all, scaffolds, tr_va_idx, split_meta)
    folds_df.to_csv(OUT_DIR / "nested_cv_folds.csv", index=False)
    with open(OUT_DIR / "nested_cv_summary.json", "w", encoding="utf-8") as fh:
        json.dump(cv_summary, fh, indent=2, sort_keys=True)

    enrich_df, enr_summary = run_fast_enrichment(split_meta["official_test_ids"])
    enrich_df.to_csv(OUT_DIR / "fast_test_enrichment.csv", index=False)
    with open(OUT_DIR / "fast_test_enrichment_summary.json", "w", encoding="utf-8") as fh:
        json.dump(enr_summary, fh, indent=2, sort_keys=True)

    total = time.time() - started
    write_readme(cv_summary, enr_summary, total)
    say(f"[MAIN] done in {total:.1f}s | outputs -> {OUT_DIR.relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
