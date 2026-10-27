# MDM2 Natural-Product Inhibitor Discovery

This repository was reorganized on **2026-10-25** into a clean, provenance-aware
layout. The original file history is preserved in Git. A safety branch,
`backup/pre-root-restructure`, points at the pre-restructure state.

## New structure

```text
00_data/                    canonical training table, screening library, config
01_traditional_ml/          placeholder README only; Arjun's clone is local/untracked
02_gnn_models/              GCN/GIN/AttentiveFP/hybrid/pretrain/triage code
03_screening_pipeline/      Part 11, Part 14, and the current Part 20 pipeline
04_docs/                    diary, references, upstream README/diary
05_archive/                 testing code, caches, stale notebooks, unclassified material
Natural_MDM2_Inhibitor_Discovery_using_ML/   upstream git submodule (untouched)
docs/                       local working notes (ignored by Git)
.opencode/                  local project skill (ignored by Git)
```

## Canonical files

| Purpose | Path |
|---|---|
| 645-compound training table | `00_data/training/ro5_properties_filtered.csv` |
| 154,647-row screening library | `00_data/screening/screening_results.csv` |
| Thresholds | `00_data/config/cuts.json` |
| Kaggle single-source runner (recommended) | `03_screening_pipeline/part20_fixed/MDM2_Kaggle_SingleSource.py` |
| Development notebook (legacy) | `03_screening_pipeline/part20_fixed/MDM2_Fixed_Pipeline_Colab.ipynb` |
| Development flat script | `03_screening_pipeline/part20_fixed/MDM2_Fixed_Pipeline_KAGGLE_ALL_CODE.py` |
| Deliverable shortlist | `03_screening_pipeline/part20_fixed/shortlist_low_confidence.csv` (after a run) |
| Backward-compatible path | root symlinks such as `Part_20_Fixed/` |

## Path compatibility

The old `Part_*` names are now **relative symlinks** to the new folders. This
keeps old notebooks and hard-coded paths working without duplicating data:

- `Part_2` -> `00_data/training`
- `Part_8` -> `00_data/screening`
- `Part_9` … `Part_20_Fixed` -> the matching new folders
- `cuts.json`, `diary.md`, `REFERENCES.md`, and the root data CSV are symlinks

Do not recreate full copies at the old paths.

## Provenance

- `Natural_MDM2_Inhibitor_Discovery_using_ML/` is Arjun sir's separate local clone,
  kept on disk but intentionally not tracked in this repository.
- `01_traditional_ml/` is reserved for traditional ML; Arjun's Parts 1-8 are not
  tracked here.
- `02_gnn_models/` contains the user's GNN, AttentiveFP, hybrid, pretraining, and
  triage work.
- `03_screening_pipeline/` contains the user's screening and Part 20 regression
  pipeline.
- See `04_docs/diary.md` and `04_docs/REFERENCES.md` for the project history and
  citations.

## Important scientific notes

- The Part 20 shortlist is a set of **low-confidence hypotheses**, not validated
  hits. An empty shortlist is a valid result.
- Do not tune on the test split, reuse test-fitted Part 18 calibrators, or mix
  78-dim and 124-dim checkpoints.
- Most of the 154,647-row COCONUT library is outside the training applicability
  domain; strict AD, active-neighbour support, model agreement, and parent/scaffold
  deduplication are required.
- Large caches, generated CSVs, `docs/`, `.opencode/`, and `organized/` are
  ignored by Git.
