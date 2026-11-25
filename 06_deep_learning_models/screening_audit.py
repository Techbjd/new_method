#!/usr/bin/env python3
# [GENERATED] Implementation written on [USER] request, 2026-11-23.
# Provenance: [USER] requested the audit; [GENERATED] code and outputs. No source
# data, notebooks, docs, Part_* archive folders or build_notebooks.py are modified.
#
# CAVEAT: Every number produced here describes COMPUTATIONAL HYPOTHESES ONLY.
# Nothing in this script or its outputs is assay validation, a confirmed hit,
# or prospective/experimental evidence of MDM2 inhibition.
"""Read-only screening + applicability-domain funnel audit from existing CSVs.

Run:  python3 screening_audit.py   (from this folder or anywhere; paths are
resolved relative to this file).  Outputs go only to ./outputs/screening_audit/.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

DATE = "2026-11-23"
PROVENANCE = "[USER] request / [GENERATED] implementation"
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
OUT_DIR = SCRIPT_DIR / "outputs" / "screening_audit"

AD_CUT_STRICT_FALLBACK = 0.52
NEAREST_ACTIVE_CUT = 0.40
MARGIN_CUT = 0.02
AGREEMENT_RANGE_CUT = 0.70
AGREEMENT_STD_CUT = 0.40

WARNINGS: list[str] = []


def warn(message: str) -> None:
    WARNINGS.append(message)
    print(f"[WARN] {message}", file=sys.stderr)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def pct(part: int | float, whole: int | float) -> float:
    return round(100.0 * float(part) / float(whole), 4) if whole else 0.0


def to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0.0) != 0.0
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "t"})


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def parse_identifier_stats(df: pd.DataFrame) -> dict:
    if "identifier" not in df.columns:
        return {"identifier_column_present": False}
    ids = df["identifier"]
    text = ids.astype("string").str.strip()
    missing = int(text.isna().sum() + (text.fillna("") == "").sum())
    dup_rows = int(text.dropna()[text.dropna().duplicated()].shape[0])
    parents = text.where(text.notna() & (text != ""), pd.Series("<MISSING>" + pd.Index(range(len(df))).astype(str), index=df.index))
    parents = parents.str.replace(r"\.\d+$", "", regex=True)
    return {
        "identifier_column_present": True,
        "missing_identifiers": missing,
        "duplicate_identifier_rows": dup_rows,
        "duplicate_identifier_values": int(text.dropna().nunique() - text.dropna()[~text.dropna().duplicated()].nunique()) if len(text) else 0,
        "unique_parent_identifiers_stripped": int(parents.nunique()),
        "duplicate_parent_rows": int(len(parents) - parents.nunique()),
    }


def read_csv_role(path: Path, role: str, required: bool = False) -> tuple[pd.DataFrame | None, dict]:
    record: dict = {
        "role": role,
        "path": rel(path),
        "exists": path.exists(),
        "required": required,
    }
    if not path.exists():
        msg = f"{role}: missing file {rel(path)}"
        (warn if not required else warn)(msg)
        record["status"] = "missing"
        return None, record
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:  # robust: report and continue
        warn(f"{role}: could not read {rel(path)}: {exc}")
        record.update(status="read_error", error=str(exc))
        return None, record
    record.update(
        status="loaded",
        bytes=path.stat().st_size,
        data_rows=int(len(df)),
        header=list(df.columns),
        schema={str(c): str(t) for c, t in df.dtypes.items()},
        identifier_stats=parse_identifier_stats(df),
    )
    print(f"[READ] {role}: {len(df):,} rows x {len(df.columns)} cols <- {rel(path)}")
    return df, record


def attach_columns(base: pd.DataFrame, other: pd.DataFrame | None, columns: list[str], source_name: str) -> pd.DataFrame:
    """Left-join selected existing columns by identifier; never invent values."""
    if other is None or "identifier" not in other.columns:
        return base
    present = [c for c in columns if c in other.columns]
    absent = [c for c in columns if c not in other.columns]
    if absent:
        warn(f"{source_name}: requested columns absent ({absent}); these gates may be unavailable")
    if not present:
        return base
    sub = other[["identifier"] + present].copy()
    if sub["identifier"].duplicated().any():
        n = int(sub["identifier"].duplicated().sum())
        warn(f"{source_name}: {n} duplicate identifiers; keeping first row per identifier")
        sub = sub.drop_duplicates("identifier", keep="first")
    overlap = [c for c in present if c in base.columns]
    if overlap:
        warn(f"{source_name}: columns already present in base, not overwritten: {overlap}")
        present = [c for c in present if c not in overlap]
    if not present:
        return base
    return base.merge(sub[["identifier"] + present], on="identifier", how="left", validate="many_to_one")


def compare_files(a: pd.DataFrame, b: pd.DataFrame, a_name: str, b_name: str) -> dict:
    out: dict = {"a": a_name, "b": b_name, "a_rows": len(a), "b_rows": len(b)}
    if "identifier" not in a.columns or "identifier" not in b.columns:
        out["status"] = "identifier column missing"
        return out
    sa = set(a["identifier"].astype(str).str.strip())
    sb = set(b["identifier"].astype(str).str.strip())
    out.update(
        unique_identifiers_a=len(sa),
        unique_identifiers_b=len(sb),
        identifiers_in_both=len(sa & sb),
        identifiers_only_in_a=len(sa - sb),
        identifiers_only_in_b=len(sb - sa),
    )
    if "canonical_smiles" in a.columns and "canonical_smiles" in b.columns:
        cols = ["identifier", "canonical_smiles"]
        m = a[cols].merge(b[cols], on="identifier", suffixes=("_a", "_b"))
        if len(m):
            mismatch = m["canonical_smiles_a"].astype(str).str.strip() != m["canonical_smiles_b"].astype(str).str.strip()
            out["smiles_mismatches_on_shared_identifiers"] = int(mismatch.sum())
    return out


def scaffold_of(mol) -> str | None:
    if mol is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return None


def main() -> int:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = {
        "screen_primary": ROOT / "00_data/screening/screening_results.csv",
        "domain_part17": ROOT / "Part_17/results/coconut_with_domain.csv",
        "gnn_calibrated_part17": ROOT / "Part_17/results/coconut_gnn_screening_full_calibrated.csv",
        "gnn_full_part17": ROOT / "Part_17/results/coconut_gnn_screening_full.csv",
        "rankfusion_part17": ROOT / "Part_17/results/coconut_rescore_rankfusion.csv",
        "consensus_part17": ROOT / "Part_17/results/consensus_T1.csv",
        "disagreement_part17": ROOT / "Part_17/results/coconut_disagreement_flags.csv",
        "training": ROOT / "00_data/training/ro5_properties_filtered.csv",
        "cuts": ROOT / "00_data/config/cuts.json",
        "part20_screening_full": ROOT / "03_screening_pipeline/part20_fixed/screening_full_regression.csv",
    }
    inputs: dict[str, dict] = {}
    frames: dict[str, pd.DataFrame | None] = {}
    for role, path in paths.items():
        if role == "cuts":
            continue
        frames[role], inputs[role] = read_csv_role(path, role, required=role in {"screen_primary", "training"})

    cuts: dict = {}
    if paths["cuts"].exists():
        try:
            cuts = json.loads(paths["cuts"].read_text())
            inputs["cuts"] = {"role": "cuts", "path": rel(paths["cuts"]), "exists": True, "status": "loaded", "content": cuts}
        except Exception as exc:
            warn(f"cuts: unreadable JSON ({exc}); using fallback thresholds")
            inputs["cuts"] = {"path": rel(paths["cuts"]), "status": "read_error", "error": str(exc)}
    else:
        warn("cuts.json missing; using fallback thresholds")
        inputs["cuts"] = {"path": rel(paths["cuts"]), "exists": False, "status": "missing"}

    ad_cut_strict = float(cuts.get("ad_cut_strict", AD_CUT_STRICT_FALLBACK))
    if "ad_cut_strict" not in cuts:
        warn(f"ad_cut_strict absent from cuts.json; using documented fallback {AD_CUT_STRICT_FALLBACK}")

    screen = frames["screen_primary"]
    training = frames["training"]
    integrity: dict = {
        "audit": {
            "date": DATE,
            "provenance": PROVENANCE,
            "mode": "read-only audit of EXISTING artifacts; no model training, no network, no source writes",
            "script": rel(Path(__file__)),
            "outputs_dir": rel(OUT_DIR),
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "rdkit": Chem.rdBase.rdkitVersion,
            "thresholds": {
                "ad_cut_strict": ad_cut_strict,
                "nearest_active": NEAREST_ACTIVE_CUT,
                "active_margin": MARGIN_CUT,
                "agreement_range": AGREEMENT_RANGE_CUT,
                "agreement_std": AGREEMENT_STD_CUT,
            },
            "caveat": "Computational hypotheses only, not assay validation; no compound is called a validated hit.",
        },
        "inputs": inputs,
        "warnings": WARNINGS,
    }

    # ---- Basic integrity of the primary screen ---------------------------------
    if screen is None:
        warn("primary screening CSV unavailable; cannot build funnel")
        integrity["fatal"] = "primary screening CSV missing or unreadable"
        (OUT_DIR / "integrity.json").write_text(json.dumps(integrity, indent=2))
        (OUT_DIR / "funnel.json").write_text(json.dumps({"funnel": [], "fatal": integrity["fatal"]}, indent=2))
        (OUT_DIR / "funnel.csv").write_text("stage,gate,status\nFATAL,primary screen missing,unavailable\n")
        (OUT_DIR / "README.md").write_text(f"# Screening audit ({DATE})\n\n{integrity['fatal']}\n\n{WARNINGS}\n")
        return 2

    need = {"identifier", "canonical_smiles"}
    missing_core = need - set(screen.columns)
    if missing_core:
        integrity["fatal"] = f"primary screen missing required columns {sorted(missing_core)}"
        (OUT_DIR / "integrity.json").write_text(json.dumps(integrity, indent=2))
        warn(integrity["fatal"])
        return 2

    ids_text = screen["identifier"].astype("string").str.strip()
    smiles_text = screen["canonical_smiles"].astype("string").str.strip()
    mols: list = []
    scaffolds: list[str | None] = []
    for i, smi in enumerate(smiles_text.tolist()):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None
        mols.append(mol)
        scaffolds.append(scaffold_of(mol))
        if (i + 1) % 50000 == 0:
            print(f"[RDKIT] parsed {i + 1:,}/{len(smiles_text):,} SMILES", flush=True)
    valid_array = np.array([m is not None for m in mols], dtype=bool)
    valid_count = int(valid_array.sum())

    parent_ids = ids_text.where(ids_text.notna() & (ids_text != ""), pd.Series(["<MISSING_ID_%d>" % i for i in range(len(screen))], index=screen.index))
    parent_ids = parent_ids.str.replace(r"\.\d+$", "", regex=True)
    integrity["screen_integrity"] = {
        "data_rows": int(len(screen)),
        "header": list(screen.columns),
        "missing_identifiers": int(ids_text.isna().sum() + (ids_text.fillna("") == "").sum()),
        "missing_canonical_smiles": int(smiles_text.isna().sum() + (smiles_text.fillna("") == "").sum()),
        "duplicate_identifier_rows": int(ids_text.dropna()[ids_text.dropna().duplicated()].shape[0]),
        "unique_identifiers": int(ids_text.nunique()),
        "valid_smiles_rows": valid_count,
        "invalid_smiles_rows": int(len(screen) - valid_count),
        "valid_smiles_fraction": round(valid_count / len(screen), 6),
        "unique_parent_identifiers": int(parent_ids.nunique()),
        "duplicate_parent_rows": int(len(parent_ids) - parent_ids.nunique()),
        "unique_murcko_scaffolds": int(pd.Series(scaffolds).nunique()),
        "scaffold_rule": "BDK Murcko scaffold, includeChirality=False (stereo-blind, matches GCN featurisation)",
    }

    # ---- Exact canonical-string overlap with the 645 training SMILES ------------
    overlap: dict = {
        "rule": "exact string match after str.strip() only; no canonicalisation, no tautomer/stereo normalisation",
        "reported_not_assumed_absent": True,
    }
    if training is not None and "canonical_smiles" in training.columns:
        train_smiles = set(training["canonical_smiles"].astype(str).str.strip())
        hit_mask = smiles_text.isin(train_smiles).fillna(False)
        overlap.update(
            training_rows=int(len(training)),
            unique_training_smiles=len(train_smiles),
            screening_rows_matching_training_smiles=int(hit_mask.sum()),
            unique_screening_smiles_matching_training=int(smiles_text[hit_mask].nunique()),
            training_smiles_found_in_screen=sorted(set(smiles_text[hit_mask].dropna()))[:10],
        )
        train_mols = [Chem.MolFromSmiles(str(s).strip()) for s in training["canonical_smiles"].astype(str)]
        train_scaffolds = {s for s in (scaffold_of(m) for m in train_mols) if s}
        screen_scaffolds = {s for s in scaffolds if s is not None}
        overlap.update(
            training_murcko_scaffolds=len(train_scaffolds),
            screening_murcko_scaffolds=len(screen_scaffolds),
            shared_murcko_scaffolds=len(train_scaffolds & screen_scaffolds),
            interpretation=(
                "Exact-canonical-string training/SMILES overlap is a direct label-leak check; "
                "scaffold overlap is only analogue proximity, not proof of leakage."
            ),
        )
        if int(hit_mask.sum()):
            warn(f"{int(hit_mask.sum())} screening rows exactly match a training canonical SMILES: possible label leakage, reported not assumed absent")
    else:
        overlap["status"] = "unavailable: training SMILES missing"
        warn(overlap["status"])
    integrity["training_smiles_overlap"] = overlap

    # ---- Cross-file schema / identifier checks ---------------------------------
    cross = {}
    for key in ("domain_part17", "gnn_calibrated_part17", "gnn_full_part17", "part20_screening_full"):
        other = frames.get(key)
        if other is not None:
            cross[key] = compare_files(screen, other, "screen_primary", key)
    integrity["cross_file_checks"] = cross

    # ---- AD definition: document what in_domain ACTUALLY is --------------------
    domain = frames.get("domain_part17")
    dvals = to_bool(domain["in_domain"]) if domain is not None and "in_domain" in domain.columns else None
    ad_info: dict = {
        "gate_used_in_funnel": "coconut_with_domain.csv::in_domain == True (per audit instruction)",
        "actual_definition_in_source_data": (
            "Part_17/Part_17_Pretrain_Finetune_v1_regression.ipynb cell 21 / "
            "Part_17_Pretrain_Finetune.ipynb cell 13: mw, logp, n_hba and n_hbd each within "
            "mean +/- 2*SD of the 645 curated training compounds, with non-null mw. "
            "This is a PHYSICOCHEMICAL DESCRIPTOR BOX, not a Tanimoto AD score."
        ),
        "is_maxT_464": False,
        "documented_strict_ad_definition": f"maxT_464 >= {ad_cut_strict} (docs/IMPROVEMENTS.md P1 #8, docs/ERRORS_AND_FIXES.md section 3, 00_data/config/cuts.json)",
        "warning": "Do NOT describe the Part_17 in_domain column as the strict maxT_464 >= 0.52 AD gate.",
        "in_domain_true_rows": int(dvals.sum()) if dvals is not None else None,
        "in_domain_fraction": round(float(dvals.mean()), 6) if dvals is not None else None,
        "in_domain_column_present": dvals is not None,
    }
    integrity["ad_definition"] = ad_info

    # ---- Assemble the audit frame ----------------------------------------------
    base = pd.DataFrame(
        {
            "identifier": ids_text,
            "parent_id": parent_ids,
            "canonical_smiles": smiles_text,
            "valid_smiles": valid_array,
            "murcko_scaffold": pd.Series(scaffolds, index=screen.index),
        }
    )
    base = attach_columns(base, domain, ["in_domain", "mw", "logp", "n_hba", "n_hbd"], "domain_part17")
    part20 = frames.get("part20_screening_full")
    part20_cols = [
        "maxT_464", "maxT_645", "maxT_343act", "nn_464_id", "nn_343act_id",
        "std_pred_pic50", "pred_gcn_pic50", "pred_gin_pic50", "pred_afp_pic50",
        "pred_hfgin_pic50", "pred_chemberta_pic50", "active_margin", "margin",
        "maxT_inactive", "maxT_inact",
    ]
    base = attach_columns(base, part20, part20_cols, "part20_screening_full")
    gnn_cal = frames.get("gnn_calibrated_part17")
    base = attach_columns(base, gnn_cal, ["gnn_pretrain_prob_cal", "gnn_pretrain_prob_raw", "prob_class_1"], "gnn_calibrated_part17")
    gnn_full = frames.get("gnn_full_part17")
    base = attach_columns(base, gnn_full, ["gnn_pretrain_prob", "gnn_scratch_prob", "rf_prob"], "gnn_full_part17")

    if "in_domain" in base.columns:
        base["in_domain_bool"] = to_bool(base["in_domain"])
    for col in ("maxT_464", "maxT_645", "maxT_343act", "std_pred_pic50", "active_margin", "margin", "maxT_inactive", "maxT_inact"):
        if col in base.columns:
            base[col + "_num"] = to_num(base[col])

    pred_cols = [c for c in ("pred_gcn_pic50", "pred_gin_pic50", "pred_afp_pic50", "pred_hfgin_pic50", "pred_chemberta_pic50") if c in base.columns]
    core_pred_cols = [c for c in ("pred_gcn_pic50", "pred_gin_pic50", "pred_afp_pic50") if c in base.columns]
    if pred_cols:
        pmat = base[pred_cols].apply(to_num)
        base["model_range"] = pmat.max(axis=1) - pmat.min(axis=1)
        base["model_std"] = pmat.std(axis=1, ddof=1)
    else:
        base["model_range"] = np.nan
        base["model_std"] = np.nan
    std_series = base["std_pred_pic50_num"] if "std_pred_pic50_num" in base.columns else base["model_std"]

    gate_availability = {
        "strict_AD_in_domain_column": "available" if "in_domain_bool" in base.columns else "unavailable: required columns missing",
        "strict_AD_maxT_464_diagnostic": "available" if "maxT_464_num" in base.columns else "unavailable: required columns missing",
        "nearest_active_support_maxT_343act": "available" if "maxT_343act_num" in base.columns else "unavailable: required columns missing",
        "active_inactive_margin": (
            "available" if ("active_margin_num" in base.columns or "margin_num" in base.columns)
            or ("maxT_343act_num" in base.columns and ("maxT_inactive_num" in base.columns or "maxT_inact_num" in base.columns))
            else "unavailable: required columns missing"
        ),
        "core_model_agreement": (
            "unavailable: required columns missing" if pred_cols == [] and "std_pred_pic50_num" not in base.columns
            else ("available: full 3-model core set" if len(core_pred_cols) >= 3 else f"partial: {len(core_pred_cols)} of 3 core pIC50 model columns in saved CSVs")
        ),
        "margin_required_columns": ["active_margin (explicit) OR maxT_343act + maxT_inactive (or maxT_inact)"],
        "agreement_required_columns": ["std_pred_pic50 (or >=2 model pIC50 columns)", "range across GCN/GIN/AttentiveFP pIC50 predictions"],
    }
    integrity["gate_availability"] = gate_availability

    # maxT_464 crosstab (documented strict AD) vs Part_17 in_domain gate.
    if "in_domain_bool" in base.columns and "maxT_464_num" in base.columns:
        ind = base["in_domain_bool"].fillna(False).astype(bool)
        maxt_ok = (base["maxT_464_num"] >= ad_cut_strict).fillna(False)
        ad_info["maxT_464_rows_missing_in_primary_screen"] = int(base["maxT_464_num"].isna().sum())
        ad_info["crosstab_in_domain_vs_maxT464_ge_cut"] = {
            "in_domain_false_maxt_below_or_missing": int((~ind & ~maxt_ok).sum()),
            "in_domain_false_maxt_above_cut": int((~ind & maxt_ok).sum()),
            "in_domain_true_maxt_below_or_missing": int((ind & ~maxt_ok).sum()),
            "in_domain_true_maxt_above_cut": int((ind & maxt_ok).sum()),
            "note": "Rows without maxT_464 (60 rows from the 154,647-row screen not present in the 154,587-row Part_20 artifact) count as below/missing.",
        }
    if "maxT_464_num" in base.columns:
        ad_info["maxT_464_rows_ge_cut"] = int((base["maxT_464_num"] >= ad_cut_strict).sum())
        ad_info["maxT_464_rows_scored"] = int(base["maxT_464_num"].notna().sum())
        ad_info["maxT_464_source"] = rel(paths["part20_screening_full"])

    # ---- Funnel -----------------------------------------------------------------
    current = base.copy()
    den_raw = len(base)
    den_parent = int(base["parent_id"].nunique())
    den_scaffold = int(base["murcko_scaffold"].nunique())

    def counts(df: pd.DataFrame) -> dict:
        return {
            "raw_rows": int(len(df)),
            "unique_parents": int(df["parent_id"].nunique()),
            "unique_scaffolds": int(df["murcko_scaffold"].nunique()),
            "unique_parent_scaffold_pairs": int(df[["parent_id", "murcko_scaffold"]].drop_duplicates().shape[0]),
        }

    rows: list[dict] = []

    def add_row(stage: str, gate: str, status: str, gate_applied: bool, source: str, definition: str,
                subset: pd.DataFrame | None = None, apply: bool = False, note: str = "", blank_counts: bool = False) -> None:
        nonlocal current
        entering = int(len(current))
        target = subset if subset is not None else current
        c = counts(target)
        if apply:
            current = target.copy()
        row = {
            "stage": stage,
            "gate": gate,
            "status": status,
            "gate_applied": bool(gate_applied),
            "source": source,
            "definition": definition,
            "rows_entering_gate": entering,
            "raw_rows": None if blank_counts else c["raw_rows"],
            "pct_raw_of_input": None if blank_counts else pct(c["raw_rows"], den_raw),
            "unique_parents": None if blank_counts else c["unique_parents"],
            "pct_parents_of_input": None if blank_counts else pct(c["unique_parents"], den_parent),
            "unique_scaffolds": None if blank_counts else c["unique_scaffolds"],
            "pct_scaffolds_of_input": None if blank_counts else pct(c["unique_scaffolds"], den_scaffold),
            "unique_parent_scaffold_pairs": None if blank_counts else c["unique_parent_scaffold_pairs"],
            "notes": note,
        }
        rows.append(row)

    add_row("00_input", "All screening rows (denominator)", "computed", False,
            rel(paths["screen_primary"]), "Every data row of the primary screening CSV; no gate applied",
            note="Stereoisomer/analog rows still present at this stage.")

    add_row("01_valid_smiles", "RDKit-parseable canonical SMILES", "computed", True,
            "RDKit Chem.MolFromSmiles", "SMILES must parse; invalid SMILES are dropped",
            subset=current[current["valid_smiles"]], apply=True,
            note="Prerequisite for any chemotype counting; not a biological filter.")

    if "in_domain_bool" in base.columns:
        add_row("02_strict_ad_in_domain", "Strict AD gate (in_domain column)", "computed: descriptor-box AD, NOT maxT_464", True,
                rel(paths["domain_part17"]), "in_domain == True (Part_17 descriptor-range box; see integrity.json -> ad_definition)",
                subset=current[current["in_domain_bool"]], apply=True,
                note="Instruction required using in_domain; it is not the documented maxT_464 >= 0.52 score. Both are reported.")
    else:
        add_row("02_strict_ad_in_domain", "Strict AD gate (in_domain column)", "unavailable: required columns missing", False,
                rel(paths["domain_part17"]), "in_domain == True", blank_counts=True,
                note="in_domain column absent from available CSVs; gate not applied.")

    if "maxT_343act_num" in base.columns:
        mask = current["maxT_343act_num"] >= NEAREST_ACTIVE_CUT
        add_row("03_nearest_active_support", f"Nearest-active support >= {NEAREST_ACTIVE_CUT}", "computed", True,
                rel(paths["part20_screening_full"]), f"maxT_343act >= {NEAREST_ACTIVE_CUT} (max Tanimoto to ACTIVE training rows)",
                subset=current[mask.fillna(False)], apply=True)
    else:
        add_row("03_nearest_active_support", f"Nearest-active support >= {NEAREST_ACTIVE_CUT}", "unavailable: required columns missing", False,
                rel(paths["part20_screening_full"]), f"maxT_343act >= {NEAREST_ACTIVE_CUT}", blank_counts=True,
                note="maxT_343act column absent; gate not applied.")

    margin_col = next((c for c in ("active_margin_num", "margin_num") if c in base.columns), None)
    inactive_col = next((c for c in ("maxT_inactive_num", "maxT_inact_num") if c in base.columns), None)
    if margin_col:
        mask = current[margin_col] >= MARGIN_CUT
        add_row("04_active_inactive_margin", f"Active-inactive margin >= {MARGIN_CUT}", "computed", True,
                "existing margin column", f"{margin_col} >= {MARGIN_CUT}", subset=current[mask.fillna(False)], apply=True)
    elif inactive_col and "maxT_343act_num" in base.columns:
        margin = current["maxT_343act_num"] - current[inactive_col]
        add_row("04_active_inactive_margin", f"Active-inactive margin >= {MARGIN_CUT}", "computed", True,
                rel(paths["part20_screening_full"]), f"maxT(active) - maxT(inactive) >= {MARGIN_CUT}",
                subset=current[margin >= MARGIN_CUT], apply=True)
    else:
        add_row("04_active_inactive_margin", f"Active-inactive margin >= {MARGIN_CUT}", "unavailable: required columns missing", False,
                "docs/ERRORS_AND_FIXES.md section 3; 03_screening_pipeline/part20_fixed/screening_full_regression.csv",
                f"active_margin = maxT(active) - maxT(inactive) >= {MARGIN_CUT}", blank_counts=True,
                note="No explicit active_margin/maxT_inactive column exists in the available CSVs. A similarity proxy was NOT fabricated; gate not applied.")

    if pred_cols or "std_pred_pic50_num" in base.columns:
        std_ok = std_series <= AGREEMENT_STD_CUT
        rng_ok = base["model_range"] <= AGREEMENT_RANGE_CUT if pred_cols else pd.Series(True, index=base.index)
        mask = (std_ok & rng_ok).fillna(False).reindex(current.index).fillna(False)
        status = ("computed: full 3-model core agreement" if len(core_pred_cols) >= 3
                  else f"computed: PARTIAL proxy, {len(core_pred_cols)}/3 core pIC50 models available")
        add_row("05_core_model_agreement", f"Agreement range <= {AGREEMENT_RANGE_CUT} and std <= {AGREEMENT_STD_CUT}", status, True,
                rel(paths["part20_screening_full"]),
                f"max-min pIC50 across available model columns <= {AGREEMENT_RANGE_CUT} AND std_pred_pic50 <= {AGREEMENT_STD_CUT}",
                subset=current[mask], apply=True,
                note=("Full GCN/GIN/AttentiveFP agreement is NOT reconstructible from saved CSVs; "
                      "this uses only the saved model columns and must not be reported as the documented 3-model gate."
                      if len(core_pred_cols) < 3 else "All three core model columns present."))
    else:
        add_row("05_core_model_agreement", f"Agreement range <= {AGREEMENT_RANGE_CUT} and std <= {AGREEMENT_STD_CUT}",
                "unavailable: required columns missing", False, rel(paths["part20_screening_full"]),
                f"range <= {AGREEMENT_RANGE_CUT}, std <= {AGREEMENT_STD_CUT}", blank_counts=True,
                note="No model prediction or std columns available; gate not applied.")

    if "maxT_464_num" in base.columns:
        mask = current["maxT_464_num"] >= ad_cut_strict
        add_row("06_AD_maxT464_diagnostic", f"Documented strict AD maxT_464 >= {ad_cut_strict} (diagnostic only)",
                "diagnostic: counts only, not carried forward", False, rel(paths["part20_screening_full"]),
                f"maxT_464 >= {ad_cut_strict}", subset=current[mask.fillna(False)],
                note="Confirms the documented AD cut on surviving rows; it does NOT replace gate 02 in the funnel chain.")
    else:
        add_row("06_AD_maxT464_diagnostic", f"Documented strict AD maxT_464 >= {ad_cut_strict}",
                "unavailable: required columns missing", False, rel(paths["part20_screening_full"]),
                f"maxT_464 >= {ad_cut_strict}", blank_counts=True, note="Column absent; no value inferred.")

    add_row("07_survivors_collapsed", "Survivors after applied gates, chemotypes collapsed", "computed (post-processing)", False,
            "RDKit BDK Murcko + parent identifier", "Counts of surviving rows, unique parents, unique scaffolds and unique parent-scaffold pairs",
            note="One row per chemotype is the counting unit; raw rows are retained for transparency. "
                 "These are computational hypotheses only, NEVER validated hits.")

    funnel = {
        "date": DATE,
        "provenance": PROVENANCE,
        "caveat": "Computational hypotheses only, not assay validation; no compound is a validated hit.",
        "denominators": {"raw_rows": den_raw, "unique_parents": den_parent, "unique_scaffolds": den_scaffold},
        "unit_note": "pct_* columns are percentages of the stage-00 input denominators.",
        "gate_availability": gate_availability,
        "funnel": rows,
        "warnings": WARNINGS,
    }

    # ---- Write outputs (only under outputs/screening_audit) ---------------------
    funnel_df = pd.DataFrame(rows)
    funnel_df.to_csv(OUT_DIR / "funnel.csv", index=False)
    (OUT_DIR / "funnel.json").write_text(json.dumps(funnel, indent=2))
    integrity["warnings"] = WARNINGS
    (OUT_DIR / "integrity.json").write_text(json.dumps(integrity, indent=2))

    def r(v, nd=0):
        return "n/a" if v is None else (f"{v:,.{nd}f}" if isinstance(v, (int, float, np.integer, np.floating)) else str(v))

    lines = [
        f"# Screening / applicability-domain audit ({DATE})",
        "",
        f"**Provenance:** {PROVENANCE}. **Generated:** {DATE}.",
        "",
        "> **CAVEAT — computational hypotheses only, not assay validation.** No compound in any "
        "input or output is a validated hit; no prospective or experimental validation was performed.",
        "",
        "## What was run",
        "",
        "A read-only audit of existing CSV artifacts: schema/integrity checks, exact canonical-string "
        "training-overlap check, an AD/support/margin/agreement funnel, and parent + BDK-Murcko "
        "chemotype collapsing. No network, no model training, no source files touched.",
        "",
        "Exact commands (from the repository root):",
        "",
        "```bash",
        'cd "/home/bijay/Desktop/already read paper/new_method/06_deep_learning_models"',
        "python3 screening_audit.py",
        "python3 -c 'import json; [json.load(open(p)) for p in [\"outputs/screening_audit/funnel.json\", \"outputs/screening_audit/integrity.json\"]]'",
        "```",
        "",
        "## Gate definitions and availability",
        "",
        "| Gate | Definition | Status |",
        "|---|---|---|",
        f"| Strict AD (applied) | `in_domain == True` from `Part_17/results/coconut_with_domain.csv` | {gate_availability['strict_AD_in_domain_column']} — BUT it is a descriptor box, not the maxT AD score |",
        f"| Nearest-active support | `maxT_343act >= {NEAREST_ACTIVE_CUT}` | {gate_availability['nearest_active_support_maxT_343act']} |",
        f"| Active-inactive margin | `maxT(active) - maxT(inactive) >= {MARGIN_CUT}` | {gate_availability['active_inactive_margin']} |",
        f"| Core-model agreement | range <= {AGREEMENT_RANGE_CUT} and std <= {AGREEMENT_STD_CUT} on pIC50 | {gate_availability['core_model_agreement']} |",
        f"| Documented strict AD (diagnostic) | `maxT_464 >= {ad_cut_strict}` | {ad_info.get('maxT_464_rows_ge_cut', 'n/a')} rows scored as passing; counted only, not chained |",
        "",
        "The `in_domain` column was **not** computed from `maxT_464`. The generating notebook "
        "(`Part_17/Part_17_Pretrain_Finetune_v1_regression.ipynb`, cell 21) defines it as MW, logP, "
        "HBA and HBD each inside mean +/- 2 SD of the 645 training compounds. The documented strict "
        "AD cut in `docs/IMPROVEMENTS.md` P1 #8, `docs/ERRORS_AND_FIXES.md` and `cuts.json` is "
        f"`maxT_464 >= {ad_cut_strict}`; those scores exist only in "
        "`03_screening_pipeline/part20_fixed/screening_full_regression.csv` and are reported "
        "separately as a diagnostic. Do not conflate the two.",
        "",
        "## Funnel (counts and percentages)",
        "",
        "| Stage | Gate | Status | Rows in | Raw rows | % raw | Unique parents | % parents | Unique scaffolds | % scaffolds |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for x in rows:
        lines.append(
            f"| {x['stage']} | {x['gate']} | {x['status']} | {r(x['rows_entering_gate'])} | "
            f"{r(x['raw_rows'])} | {r(x['pct_raw_of_input'], 3)} | {r(x['unique_parents'])} | "
            f"{r(x['pct_parents_of_input'], 3)} | {r(x['unique_scaffolds'])} | {r(x['pct_scaffolds_of_input'], 3)} |"
        )
    lines += [
        "",
        "Percentages use stage-00 denominators: "
        f"{den_raw:,} raw rows, {den_parent:,} unique parents, {den_scaffold:,} unique Murcko scaffolds. "
        "Stages with status `unavailable: required columns missing` have blank counts: the gate was "
        "NOT applied and no value was invented. `diagnostic` rows were counted but did not filter "
        "the chained funnel.",
        "",
        "## Integrity findings",
        "",
        f"- Primary screen rows: **{integrity['screen_integrity']['data_rows']:,}**; "
        f"columns: `{', '.join(integrity['screen_integrity']['header'])}`.",
        f"- Missing identifiers: **{integrity['screen_integrity']['missing_identifiers']}**; "
        f"duplicate identifier rows: **{integrity['screen_integrity']['duplicate_identifier_rows']}**; "
        f"duplicate parent rows (version stripped): **{integrity['screen_integrity']['duplicate_parent_rows']}**.",
        f"- Valid SMILES: **{integrity['screen_integrity']['valid_smiles_rows']:,}/"
        f"{integrity['screen_integrity']['data_rows']:,}** "
        f"({100 * integrity['screen_integrity']['valid_smiles_fraction']:.2f}%).",
        f"- Exact training canonical-SMILES overlap: **{overlap.get('screening_rows_matching_training_smiles', 'n/a')} "
        f"screening rows / {overlap.get('unique_screening_smiles_matching_training', 'n/a')} unique SMILES** "
        f"against {overlap.get('training_rows', 'n/a')} training rows "
        f"({overlap.get('rule', 'n/a')}). Shared Murcko scaffolds: "
        f"**{overlap.get('shared_murcko_scaffolds', 'n/a')}** (analogue proximity only, not a label-leak proof).",
        f"- Part_17 `in_domain == True`: **{ad_info.get('in_domain_true_rows', 'n/a')}** rows "
        f"({100 * (ad_info.get('in_domain_fraction') or 0):.2f}%); "
        f"`maxT_464 >= {ad_cut_strict}`: **{ad_info.get('maxT_464_rows_ge_cut', 'n/a')}** rows.",
        "- Cross-file identifier/schema comparisons are in `integrity.json` -> `cross_file_checks`.",
        "",
        "## Why chemotype collapsing matters",
        "",
        "Identifier versions like `.1`/`.2` mark stereoisomers/analogs of one parent. The GCN graph "
        "featurisation is stereo-blind, so stereoisomer rows can carry identical predictions and "
        "inflate any raw hit count. Every funnel stage therefore reports raw rows, unique parents "
        "(identifier with trailing `.N` stripped) and unique BDK Murcko scaffolds "
        "(`includeChirality=False`). Acyclic molecules share the empty Murcko scaffold string; "
        "that is counted as one scaffold bucket, not as a missing value.",
        "",
        "## Limitations",
        "",
        "- No prospective, experimental or assay validation was done; outputs are computational hypotheses only.",
        "- The applied AD gate is the Part_17 descriptor-box `in_domain` column, not the documented "
        "maxT Tanimoto AD; the latter is reported only as a diagnostic count.",
        "- The active-inactive margin gate was NOT applied: no `active_margin`/`maxT_inactive` column "
        "exists in the available CSVs, and no proxy was substituted.",
        "- Core-model agreement uses only the saved pIC50 model columns "
        f"({', '.join(core_pred_cols) if core_pred_cols else 'none'}); the documented three-model "
        "GCN/GIN/AttentiveFP agreement is not reconstructible from these artifacts.",
        "- No retraining, recalibration or re-scoring was performed; all numbers are inherited from "
        "existing artifacts whose provenance is recorded in `integrity.json`.",
        "- Exact-string training overlap is a conservative leak check: zero matches do not exclude "
        "scaffold/analogue similarity to labelled compounds.",
        "- All downstream gates inherit any limitations of the upstream saved columns, including "
        "differences in row counts between the 154,647-row screen and 154,587-row Part_20 artifact.",
        "",
        "## Warnings emitted during this run",
        "",
    ]
    if WARNINGS:
        lines += [f"- {w}" for w in WARNINGS]
    else:
        lines.append("- None.")
    lines += [
        "",
        "## Output files",
        "",
        "- `outputs/screening_audit/funnel.csv` — stage table with counts/percentages.",
        "- `outputs/screening_audit/funnel.json` — same funnel plus denominators and gate availability.",
        "- `outputs/screening_audit/integrity.json` — schemas, row counts, overlap and AD definition evidence.",
        "- `outputs/screening_audit/README.md` — this file.",
        "",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines))

    elapsed = time.time() - started
    print(f"[DONE] {elapsed:.1f}s | inputs={len(inputs)} | funnel stages={len(rows)} | warnings={len(WARNINGS)}")
    for x in rows:
        print(f"  {x['stage']:<28} {str(x['status']):<55} raw={str(x['raw_rows']):>10} parents={str(x['unique_parents']):>10} scaffolds={str(x['unique_scaffolds']):>10}")
    print(f"[OUTPUT] {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
