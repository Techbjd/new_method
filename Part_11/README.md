# Part 11: GNN Virtual Screening of COCONUT Database

Screens ~154k natural products from the COCONUT database with the trained GNN
models (Part 9 GCN + Part 10 GIN) and reports high-confidence consensus hits.

Pipeline: COCONUT SMILES → Ro5 pre-filter → domain-of-applicability filter →
GCN + GIN scoring → strict thresholds → consensus hits.

## Contents

| File | Description |
|------|-------------|
| `Part_11_GNN_Screening.ipynb` | Screening notebook (copy of the root notebook) |
| `ro5_properties_filtered.csv` | Training-domain reference for the domain-of-applicability check |
| `requirements.txt` | Python dependencies |

## Inputs (full-pipeline layout, not duplicated here)

- `Part_8/screening_results.csv` — 154,648 COCONUT compounds (too large to duplicate)
- `gcn_scratch_model.pth` — trained GCN weights, in `Part_9/`
- `pretrained_gin_mdm2.pth` — trained GIN weights, in `Part_10/`

## Outputs (written to working directory when run)

- `gnn_screening_results.csv` — all screenable compounds with GCN/GIN scores
- `gnn_consensus_hits.csv` — high-confidence consensus hits
- `gnn_screening_analysis.png` — score distribution plots
- `gnn_paperstyle_hits.csv` — deep-learning screening mode (paper §3.5 logic:
  mean GCN/GIN prob > 0.6 + PAINS/Brenk/Ro5, for docking)
- `gnn_hits_for_docking.sdf` — 3D structures of DL hits (needs local COCONUT SDF)

## Paper-style deep-learning mode

The last notebook section (§11) mirrors the paper pipeline: downloads
`coconut_csv-03-2025.csv`, applies PAINS → Brenk → Lipinski filters, and screens
with GCN + GIN at the paper's `prob > 0.6` cutoff instead of RF.

## How to run

```bash
pip install -r Part_11/requirements.txt
jupyter notebook Part_11/Part_11_GNN_Screening.ipynb
```

Note: skip the `git clone` cell if you already have the repository locally.
