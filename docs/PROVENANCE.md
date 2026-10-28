# Provenance Map

This file separates the original Arjun sir/traditional-ML work from the user's GNN, screening, pretraining, and regression extensions. Nothing here authorizes deleting or moving an original file.

Legend:

- `[ARJUN]` — original or near-original code from `Natural_MDM2_Inhibitor_Discovery_using_ML`.
- `[ARJUN COPY]` — a copy of Arjun sir's notebook kept inside a later part folder.
- `[ADAPTED]` — user-modified or re-run version of Arjun sir's approach.
- `[USER]` — user-designed implementation.
- `[GENERATED]` — model weights, caches, plots, CSV outputs, or logs.

## Original Arjun sir repository

Source: `https://github.com/arjunpahi/Natural_MDM2_Inhibitor_Discovery_using_ML.git`, cloned 2026-09-07.

| Path | Label | Role |
|---|---|---|
| `Natural_MDM2_Inhibitor_Discovery_using_ML/Part_1` | `[ARJUN]` | ChEMBL data extraction/cleaning |
| `Natural_MDM2_Inhibitor_Discovery_using_ML/Part_2` | `[ARJUN]` | Lipinski/RO5 filtering to 645 compounds |
| `Natural_MDM2_Inhibitor_Discovery_using_ML/Part_3` | `[ARJUN]` | EDA |
| `Natural_MDM2_Inhibitor_Discovery_using_ML/Part_4` | `[ARJUN]` | Morgan/MACCS fingerprint generation |
| `Natural_MDM2_Inhibitor_Discovery_using_ML/Part_5` | `[ARJUN]` | PCA/t-SNE visualization |
| `Natural_MDM2_Inhibitor_Discovery_using_ML/Part_6` | `[ARJUN]` | 20 traditional classifiers; RF baseline |
| `Natural_MDM2_Inhibitor_Discovery_using_ML/Part_7` | `[ARJUN]` | COCONUT library preparation |
| `Natural_MDM2_Inhibitor_Discovery_using_ML/Part_8` | `[ARJUN]` | RF screening (116 predicted inhibitors) |
| `Natural_MDM2_Inhibitor_Discovery_using_ML/Part_9` | `[ARJUN]` | DeepChem GCN from scratch |
| `Natural_MDM2_Inhibitor_Discovery_using_ML/Part_10` | `[ARJUN]` | Pretrained GIN (MolCLR architecture) |
| `Natural_MDM2_Inhibitor_Discovery_using_ML/Part_11` | `[ARJUN]` | GNN screening |
| `Natural_MDM2_Inhibitor_Discovery_using_ML/Part_12` | `[ARJUN]` | later/auxiliary notebook |

## User's root-level extension folders

These directories are not part of the original Arjun sir repository unless explicitly marked.

| Path | Label | Role |
|---|---|---|
| `Part_9/Part_9_GNN_DeepChem.ipynb` | `[ARJUN COPY]` | copied baseline GCN notebook |
| `Part_9/Part_9_GCN_HFooladi.ipynb` | `[USER]` | user GCN/HFooladi experiment |
| `Part_9/train_gcn_hfooladi.py` | `[USER]` | standalone GCN training script |
| `Part_10/Part_10_Pretrained_GNN.ipynb` | `[ARJUN COPY]` | copied pretrained-GIN notebook |
| `Part_10/Part_10_GIN_HFooladi.ipynb` | `[USER]` | user GIN/HFooladi experiment |
| `Part_10/train_gin_hfooladi.py` | `[USER]` | standalone GIN training script |
| `Part_11/Part_11_GNN_Screening.ipynb` | `[ADAPTED]` | screening extended from Arjun sir's Part 11 |
| `Part_12` | `[USER]` | AttentiveFP regression/screening |
| `Part_13` | `[USER]` | hybrid GCN+GAT experiment |
| `Part_14` | `[USER]` | ensemble screening |
| `Part_15` | `[USER]` | small-data deep-learning experiment |
| `Part_17` | `[USER]` | pretrain/finetune and regression variants |
| `Part_18_GNN_Triage` | `[USER]` | fixed GCN/GIN training and triage scripts |
| `part19` | `[USER]` | exploratory notebook |
| `Part_20_Fixed` | `[USER]` | final leak-controlled regression/screening pipeline |
| `testing/fix_of_chemphore` | `[USER]` | pharmacophore/shape experimentation |
| `xo_cache` | `[GENERATED]` | cache artifacts |
| root `pretrain_stage1a_npass.ipynb` | `[USER]` | pretraining experiment |
| root `MDM2_Fixed_Pipeline_Colab.ipynb` | `[USER]` | flattened/earlier Part_20 copy |

## Data and shared assets

| Path | Label | Role |
|---|---|---|
| `Part_2/ro5_properties_filtered.csv` | `[ARJUN-derived]` | 645-compound training table used by later models |
| `Part_8/screening_results.csv` | `[ARJUN-derived]` | 154,587-row COCONUT screening library |
| `cuts.json` | `[USER]` | AD/hit cutoffs; Part_20 adds `ad_cut_strict` |
| `ro5_properties_filtered.csv` | `[ARJUN-derived copy]` | root copy of the 645 table |
| `REFERENCES.md` | `[USER]` | paper/model citations |
| `diary.md` | `[USER]` | project journey log |

## Provenance rules for future edits

1. Do not overwrite an `[ARJUN]` file with a user version. Put the user version in the appropriate `organized/` subtree.
2. Keep `[ARJUN COPY]` and `[ADAPTED]` distinct so results are not attributed incorrectly.
3. Never treat generated CSVs/model weights as source code; document their producing script.
4. If provenance is uncertain, mark it `[UNVERIFIED]` and explain why. Do not guess.
