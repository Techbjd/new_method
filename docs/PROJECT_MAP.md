# MDM2 Project Map

This document is the navigation layer for the repository. It is intentionally
non-destructive: every original folder is treated as an archive, and all
reorganized copies live under `organized/`.

## Hard rules

1. **Never delete or move an original file.** Copy first, verify, then leave the
   original in place.
2. Put provenance labels on every organized item:
   - `[ARJUN]` — code or notebooks authored in Arjun sir's repository.
   - `[USER]` — code or notebooks authored by the user.
   - `[ADAPTED]` — user copy/adaptation of an Arjun sir notebook or script.
   - `[GENERATED]` — outputs, caches, plots, model artifacts, or downloads.
3. Generated artifacts and caches are not source. Keep them out of the source
   tree; copy only what is needed for reproducibility.
4. The current `Part_*` folders are the working archive. Do not rename them in
   place.
5. The current Part_20 pipeline is the authoritative screening implementation
   until a newer version is explicitly promoted.

## Current top level

| Path | Role | Provenance |
|---|---|---|
| `Natural_MDM2_Inhibitor_Discovery_using_ML/` | Original 12-part Arjun sir repository; separate git repo/submodule | `[ARJUN]` |
| `Part_2/` | Training table and early data preparation | `[ARJUN]` / `[USER]` |
| `Part_8/` | Screening library and traditional-ML screening outputs | `[ARJUN]` / `[USER]` |
| `Part_9/` | GCN work, including `Part_9_GCN_HFooladi.ipynb` and `train_gcn_hfooladi.py` | `[ADAPTED]` / `[USER]` |
| `Part_10/` | GIN work, including `Part_10_GIN_HFooladi.ipynb` and `train_gin_hfooladi.py` | `[ADAPTED]` / `[USER]` |
| `Part_11/` | GNN screening | `[USER]` |
| `Part_12/` | AttentiveFP | `[USER]` |
| `Part_13/` | Hybrid GCN+GAT | `[USER]` |
| `Part_14/` | Ensemble screening | `[USER]` |
| `Part_15/` | Small-data deep learning | `[USER]` |
| `Part_17/` | Pretrain/finetune experiments | `[USER]` |
| `Part_18_GNN_Triage/` | Fixed GCN/GIN training, triage, `common.py` | `[USER]` |
| `part19/` | Earlier experiment notebook | `[USER]` |
| `Part_20_Fixed/` | Current production-style regression/screening pipeline | `[USER]` |
| `testing/` | Ad-hoc experiments, including chemophore fixes | `[USER]` / `[GENERATED]` |
| `xo_cache/` | Cache/experimental artifacts | `[GENERATED]` |
| `diary.md`, `REFERENCES.md` | Project history and citations | `[USER]` |
| root `*.ipynb`, `*.py` | Earlier/duplicate entry points | `[USER]` / `[ADAPTED]` |

## Target `organized/` tree

```text
organized/
  00_data/
    README.md
    ro5_properties_filtered.csv          # copy of Part_2 training table
    screening_results.csv                # copy of Part_8 library
    cuts.json                            # current thresholds
  01_traditional_ml/
    arjun_parts_1_8/                     # original Arjun notebooks/scripts
    user_screening_baseline/             # user copies used for the paper baseline
  02_gnn_models/
    part09_gcn/
    part10_gin/
    part12_attentivefp/
    part13_hybrid_gcn_gat/
    part17_pretrain_finetune/
    part18_fixed_triage/
    part19_experiments/
  03_screening_pipeline/
    part11_gnn_screening/
    part14_ensemble/
    part20_fixed/                        # current authoritative pipeline
  04_docs/
    diary.md
    references.md
    project_map.md
    provenance.md
    journey.md
    errors_and_fixes.md
  05_archive/
    testing_fix_of_chemphore/
    duplicate_root_notebooks/
    legacy_outputs/
```

The tree is a target, not a claim that every file has already been copied.
`organized/README.md` records copy status and unresolved provenance questions.

## Current authoritative entry points

- Notebook: `Part_20_Fixed/MDM2_Fixed_Pipeline_Colab.ipynb`
- Flat Kaggle script: `Part_20_Fixed/MDM2_Fixed_Pipeline_KAGGLE_ALL_CODE.py`
- Training data: `Part_2/ro5_properties_filtered.csv` (645 rows)
- Screening library: `Part_8/screening_results.csv` (154,647 rows)
- Thresholds: `cuts.json`
- Current shortlist output (after a run): `Part_20_Fixed/shortlist_low_confidence.csv`

## Safe workflow

1. Read `PROVENANCE.md`, `JOURNEY.md`, and `ERRORS_AND_FIXES.md`.
2. Identify the source folder and its provenance label.
3. Copy into the matching `organized/` folder; do not rename the source.
4. Add a short `README.md` beside the copy with the source path, date, and
   whether it is runnable as-is.
5. Validate syntax (`python -m py_compile` for scripts, JSON parse for notebooks).
6. Update this map and `organized/README.md`.

## Open questions to resolve from the full audit

- Which root Part_9/10/11 files are exact copies versus substantive adaptations?
- Which root notebooks are superseded by `Part_20_Fixed`?
- Which `testing/` and `xo_cache/` items are reproducible experiments versus
  disposable artifacts?
- Which Arjun sir Parts 1-8 notebooks were actually used for the paper numbers?
