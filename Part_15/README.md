# Part 15: Small-Data Deep Learning for MDM2 Classification

Trains three **small-data-safe** deep-learning models on the 645 MDM2
compounds (n=645 is too small for big GNNs — Part 12's 350k-param AttentiveFP
scored AUC 0.924 while the 60k-param Hybrid of Part 13 reached 0.956, and
Random Forest still leads at 0.960). Every model here is kept small
(<50k trainable params), regularized, and validated with the same protocol
as Parts 9/10/12/13: 80/20 stratified split (seed=57), 100 epochs,
10x Y-randomization, 5-fold CV.

## Models (all paper-backed, GitHub-standard implementations)

| # | Model | Reference | Original code | This port |
|---|-------|-----------|---------------|-----------|
| 1 | **D-MPNN-small** — directed edge message passing, hidden=64, depth=2 (~23k params) | Yang et al., "Analyzing Learned Molecular Representations for Property Prediction," *J. Chem. Inf. Model.* 2019 (Chemprop) | `chemprop/chemprop` | PyTorch-Geometric port reusing the 78-dim atom + 6-dim bond featurization of Parts 12/13 |
| 2 | **GINE-small** — GIN with edge features, 3 layers, hidden=64 (~40k params) | Hu et al., "Strategies for Pre-training Graph Neural Networks," *ICLR* 2020 | `snap-stanford/pretrain-gnns` | Hand-rolled GINE layer (project + edge-MLP + eps-MLP), same style as the from-scratch ports in Parts 9/12/13 |
| 3 | **Frozen ChemBERTa + MLP** — `seyonec/ChemBERTa-zinc-base-v1` frozen (mean-pooled, 768-dim), trainable head 768->64->2 (~49k params) | Chithrananda et al., "ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction," *arXiv:2010.09885* 2020 | `huggingface/transformers` + HF hub `seyonec/ChemBERTa-zinc-base-v1` | Embeddings cached to `chemberta_embeddings.pt`; only the MLP head trains |

Small-data tactics used: small hidden dims, dropout 0.2-0.3, **SMILES-enumeration
augmentation (train-only, x4)** — same molecule written 4 ways via
`MolToSmiles(doRandom=True)` — and a frozen language-model featurizer so most
"learning" comes from millions of pretraining molecules, not our 645.

## Contents

| File | Description |
|------|-------------|
| `Part_15_SmallData_DL_Local.ipynb` | Training notebook for local runs (clone + CPU/GPU-auto) |
| `Part_15_SmallData_DL_Colab.ipynb` | Same core, fully self-contained for Colab (clone + pip + data cells) |
| `ro5_properties_filtered.csv` | Input data: 645 compounds with SMILES + activity class (MD5 `eae88fc7...`, identical to Parts 9-13) |
| `dmpnn_mdm2.pth` | Trained D-MPNN weights — saved by YOUR run (never committed untrained) |
| `gine_mdm2.pth` | Trained GINE weights — saved by YOUR run (never committed untrained) |
| `chemberta_mlp_mdm2.pth` | Trained MLP-head weights — saved by YOUR run (never committed untrained) |
| `chemberta_embeddings.pt` | Cached frozen ChemBERTa embeddings (train/aug/test/all) — built by YOUR run |
| `requirements.txt` | Python dependencies |

## Architecture

- D-MPNN: `Linear(84->64)` directed-edge init -> 2x directed message steps
  (reverse-edge exclusion, skip connection) -> atom update `Linear(142->64)`
  -> add-pool -> `Linear(64->64)->ReLU->Dropout->Linear(64->2)`. ~23k params.
- GINE: 3x `GINELayer` (atom project + bond-MLP messages + eps self-update)
  -> add-pool -> `Linear(64->32)->ReLU->Dropout->Linear(32->2)`. ~40k params.
- ChemBERTa-MLP: frozen 768-dim mean-pooled embedding ->
  `Linear(768->64)->ReLU->Dropout->Linear(64->2)`. ~49k trainable params.

Training: 100 epochs, Adam (lr=1e-3), batch_size=64 (graphs) / 64 (embeddings),
5-fold stratified CV, 10x Y-randomization (expect ~0.5 if the model is honest,
like Parts 9-10/12-13). Main training uses the SMILES-augmented (x4) train
pool; Y-randomization and CV use the standard non-augmented protocol so the
honesty checks stay comparable.

## How to run — local PC (CPU now, GPU later)

```bash
pip install -r Part_15/requirements.txt
jupyter notebook Part_15/Part_15_SmallData_DL_Local.ipynb
```

NOTE: the ChemBERTa section downloads `seyonec/ChemBERTa-zinc-base-v1`
(~300 MB) from the HuggingFace hub on first run (needs internet once);
after that the cached `chemberta_embeddings.pt` is reused. If the download
fails, that section is skipped gracefully — D-MPNN and GINE still run.

## How to run — Colab GPU

Open `Part_15_SmallData_DL_Colab.ipynb` in Colab → T4 GPU runtime → Run all.
Saves `dmpnn_mdm2.pth`, `gine_mdm2.pth`, `chemberta_mlp_mdm2.pth`; download
them and place in `Part_15/` — Part 14 auto-loads them as voters 6-8.
