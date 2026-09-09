# Part 12: AttentiveFP — Attention-Based GNN for MDM2 Classification

Trains an AttentiveFP network (atom-level attention with bond features, GRU
memory, attentive graph readout) from scratch on the 645 MDM2 compounds.

Reference: Xiong et al., "Pushing the Boundaries of Molecular Representation
for Drug Discovery with the Graph Attention Mechanism," *J. Med. Chem.* 2020,
63(16):8749–8760. Original code: `OpenDrugAI/AttentiveFP` (PyTorch), also
wrapped in DeepChem / DGL-LifeSci. This is a PyTorch-Geometric port of that
logic: hidden=128, K=2 attentive layers, T=2 readout steps (paper defaults
200-dim; reduced for our n=645 dataset).

## Contents

| File | Description |
|------|-------------|
| `Part_12_AttentiveFP_Local.ipynb` | Training notebook for local runs (clone + CPU/GPU-auto) |
| `Part_12_AttentiveFP_Colab.ipynb` | Same core, fully self-contained for Colab (clone + pip + data cells) |
| `ro5_properties_filtered.csv` | Input data: 645 compounds with SMILES + activity class |
| `attentivefp_mdm2.pth` | Trained weights — saved by YOUR run (never committed untrained) |
| `requirements.txt` | Python dependencies |

## Architecture

`Linear(78→128)` atoms + `Linear(6→128)` bonds → 2× AttentiveLayer
(neighbor attention + GRU update) → AttentiveReadout (T=2, GRU memory) →
Linear(128→64) → ReLU → Dropout → Linear(64→2). ~350k params.

Training: 100 epochs, Adam (lr=1e-3), batch_size=64, 5-fold stratified CV,
10× Y-randomization (expect ~0.5 if the model is honest, like Parts 9–10).

## How to run — local PC (CPU now, GPU later)

```bash
pip install -r Part_12/requirements.txt
jupyter notebook Part_12/Part_12_AttentiveFP_Local.ipynb
```

## How to run — Colab GPU

Open `Part_12_AttentiveFP_Colab.ipynb` in Colab → T4 GPU runtime → Run all.
Saves `attentivefp_mdm2.pth`; download it and place in `Part_12/`.
