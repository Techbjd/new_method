# Part 9: GNN From Scratch (GCN for MDM2 Classification)

Trains a 3-layer Graph Convolutional Network directly on molecular graphs
(atoms = nodes, bonds = edges). No pretrained weights — learns from scratch
on the 645 MDM2 compounds.

Reference: Yasir et al., 2025 (GraphConvMol pipeline).

## Contents

| File | Description |
|------|-------------|
| `Part_9_GNN_DeepChem.ipynb` | Training notebook (copy of the root notebook) |
| `ro5_properties_filtered.csv` | Input data: 645 compounds with SMILES + activity class |
| `gcn_scratch_model.pth` | Trained model weights |
| `requirements.txt` | Python dependencies |

## Architecture

`GCNConv(78→128) → GCNConv(128→128) → GCNConv(128→128) → mean+max pool(256) → Linear(256→64) → ReLU → Dropout → Linear(64→2)`

Training: 100 epochs, Adam (lr=1e-3), batch_size=64, 5-fold stratified CV.

## Results (Colab run)

Test set (80/20 split): Accuracy 0.915, ROC-AUC 0.952, F1 0.923, MCC 0.830.
5-fold CV mean: ROC-AUC 0.945 ± 0.019. Y-randomization AUC ~0.28–0.59 (random),
confirming real signal.

## How to run

```bash
pip install -r Part_9/requirements.txt
jupyter notebook Part_9/Part_9_GNN_DeepChem.ipynb
```

Note: the notebook reads `Part_2/ro5_properties_filtered.csv` and
`Part_6/performance_morgan_tuned.csv` using the full-pipeline layout (see the
`Natural_MDM2_Inhibitor_Discovery_using_ML` project). A copy of the main input
file is included in this folder; point the notebook at it if you run
standalone.
