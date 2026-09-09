# Part 13: Hybrid GCN+GAT for MDM2 Classification

Trains a **hybrid graph network** from scratch on the 645 MDM2 compounds:
GCN layers extract local structure first, then a multi-head GAT layer refines
with learned neighbor attention, then mean+max pooling + classifier.

Reference (hybrid design): HDTI-IC50 — hybrid GCN+GAT for p53-inhibitor IC50
prediction, *J. Comput.-Aided Mol. Des.* 2025 (MAE 0.1, RMSE 0.19, R² 0.8,
beating single GCN/GAT). GCN: Kipf & Welling, ICLR 2017. GAT: Velickovic
et al., ICLR 2018. Same 78-dim atom features and 645-compound protocol as
Parts 9/10/12, so results compare directly (RF 0.960 / GCN 0.952 / GIN 0.947).

## Contents

| File | Description |
|------|-------------|
| `Part_13_HybridGCNGAT_Local.ipynb` | Training notebook for local runs (clone + CPU/GPU-auto) |
| `Part_13_HybridGCNGAT_Colab.ipynb` | Same core, fully self-contained for Colab (clone + pip + data cells) |
| `ro5_properties_filtered.csv` | Input data: 645 compounds with SMILES + activity class |
| `hybrid_gcn_gat_mdm2.pth` | Trained weights — saved by YOUR run (never committed untrained) |
| `requirements.txt` | Python dependencies |

## Architecture

`GCNConv(78→128) → GCNConv(128→128) → GATConv(128→32×4 heads) → mean+max
pool(256) → Linear(256→64) → ReLU → Dropout → Linear(64→2)`. ~60k params —
deliberately small for n=645.

Training: 100 epochs, Adam (lr=1e-3), batch_size=64, 5-fold stratified CV,
10× Y-randomization (expect ~0.5 if the model is honest, like Parts 9–12).

## How to run — local PC (CPU now, GPU later)

```bash
pip install -r Part_13/requirements.txt
jupyter notebook Part_13/Part_13_HybridGCNGAT_Local.ipynb
```

## How to run — Colab GPU

Open `Part_13_HybridGCNGAT_Colab.ipynb` in Colab → T4 GPU runtime → Run all.
Saves `hybrid_gcn_gat_mdm2.pth`; download it and place in `Part_13/`.
