# Part 14: All-Models Ensemble Screening of COCONUT

Screens COCONUT with **every trained model at once** — RF (Part 6/8) + GCN
(Part 9) + GIN (Part 10) + AttentiveFP (Part 12) + Hybrid GCN+GAT (Part 13) —
and reports majority consensus + soft-vote ensemble hits for docking.

Design: **read-only**. This notebook only loads weights/metrics from Parts
6/8/9/10/12/13 and writes its own outputs here. It never modifies those
folders. Models whose weights are absent (e.g. Parts 12/13 before you train
them on Colab) are skipped automatically and the vote runs on the rest.

## Contents

| File | Description |
|------|-------------|
| `Part_14_Ensemble_Screening.ipynb` | Screening notebook (local + Colab; clone cells included) |
| `requirements.txt` | Python dependencies |

## Inputs (read from repo layout, not duplicated here)

- `Part_8/screening_results.csv` — 154,647 COCONUT compounds (default input),
  or raw `coconut_csv-03-2025.csv` via the built-in Option B filter cell
- `Part_9/gcn_scratch_model.pth`, `Part_10/pretrained_gin_mdm2.pth`,
  `Part_12/attentivefp_mdm2.pth`, `Part_13/hybrid_gcn_gat_mdm2.pth`
  (missing ones auto-downloaded if on GitHub, else skipped with a message)

## Outputs (written to working directory when run)

- `ensemble_screening_results.csv` — all compounds with every model score
- `ensemble_consensus_hits.csv` — strict consensus + medchem hits
- `ensemble_screening_analysis.png` — probability distributions
- `ensemble_hits_for_docking.sdf` — 3D structures (needs local COCONUT SDF)

## How to run — Colab GPU (recommended for 154k screening)

Open in Colab → T4 GPU runtime → Run all. Download the CSV/SDF outputs
from the Files panel.

## How to run — local

```bash
pip install -r Part_14/requirements.txt
jupyter notebook Part_14/Part_14_Ensemble_Screening.ipynb
```

Note: skip the `git clone` cell if you already have the repository locally.
Screening 154k molecules needs RAM — close other apps, or screen in chunks.
