# Part 10: Pretrained GNN Fine-Tuning (MolCLR GIN)

Fine-tunes a Graph Isomorphism Network (MolCLR architecture) on the 645 MDM2
compounds. The notebook tries to load MolCLR pretrained weights; on
architecture mismatch it falls back to training the same 5-layer GIN
architecture from scratch.

Reference: Wang et al., 2020 (MolCLR, NeurIPS 2020).

## Contents

| File | Description |
|------|-------------|
| `Part_10_Pretrained_GNN.ipynb` | Training notebook (copy of the root notebook) |
| `ro5_properties_filtered.csv` | Input data: 645 compounds with SMILES + activity class |
| `pretrained_gin_mdm2.pth` | Trained model weights |
| `requirements.txt` | Python dependencies |

## Architecture

5-layer GIN (hidden=300) → mean+max pool(600) → Linear(600→128) → ReLU →
Dropout(0.3) → Linear(128→2). Total: 465,091 params.

Training: 100 epochs, Adam with differential LR (encoder 1e-5, classifier
1e-3), batch_size=64.

## Results (Colab run)

Test set (80/20 split): Accuracy 0.899, ROC-AUC 0.947, F1 0.913, MCC 0.807,
Sensitivity 0.986 (highest of all models — best at catching actives).
5-fold CV mean: ROC-AUC 0.937 ± 0.017.

## How to run

```bash
pip install -r Part_10/requirements.txt
jupyter notebook Part_10/Part_10_Pretrained_GNN.ipynb
```

Notes:
- The notebook reads `Part_2/ro5_properties_filtered.csv` using the
  full-pipeline layout. A copy of the input file is included in this folder;
  point the notebook at it if you run standalone.
- Internet access is needed on first run to fetch the MolCLR repo for the
  pretrained-weights attempt (optional — falls back to training from scratch
  offline).
