# Deep Learning Model Research & Provenance

This document records the deep learning models used in `06_deep_learning_models/`, their source papers, official/community GitHub implementations, and how they are implemented in this project. It is a provenance/research note, not a claim that external pretrained weights were reused.

## Guardrails

- **No label leakage**: all models train on the leak-free Murcko scaffold split (464/52/129, seed 57); validation-only early stopping/selection; test scored once.
- **No external pretrained weights on MDM2 labels**: DL05 self-supervised pretraining uses unlabeled open-source chemistry SMILES (default ZINC250k, downloaded/cached) and never sees MDM2 labels.
- **Applicability domain**: 154k COCONUT screening predictions are low-confidence hypotheses; report Tanimoto AD.
- **Implementation style**: PyTorch/PyTorch Geometric keyword-only factories; molecular graphs use 78-dim atom features and 6-dim bond features, undirected edges.

## Interpretation limits (paper-facing)

> **Fast AUCs are plumbing/sanity results on a 5-scaffold, 129-compound holdout — not prospective performance.** Use repeated/nested scaffold CV, Y-randomization, AD coverage and external validation before claiming generalization.

- The fast profile (8 epochs, 2 CV folds, 2 Y-randomization shuffles) validates that every architecture in this note trains, tunes, CVs and scores once; it is **not** the full protocol and not a model-comparison result.
- With only five Murcko scaffolds in the test set, single-split architecture rankings are unstable. Prefer validation-only CV selection, then report enrichment (precision@10, EF1%, BEDROC) and bootstrap CIs.
- Graphormer, 3-D/equivariant models and large pretrained chemical LMs are **excluded** below: they need data volumes, conformers or pretraining corpora that 645 2-D molecules cannot support. They are future work, not evidence for this paper.

## Newly Added Graph Models (2026-11-22)

| Model | Paper | Venue/Year | GitHub / Code | PyG Class | Uses edge_attr | Added to |
|---|---|---:|---|---|---:|---|
| GINE | Hu et al., *Strategies for Pre-training Graph Neural Networks* | ICLR 2020 | https://github.com/snap-stanford/pretrain-gnns ; PyG: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html | `GINEConv` | Yes | DL01 |
| GraphSAGE | Hamilton, Ying, Leskovec, *Inductive Representation Learning on Large Graphs* | NeurIPS 2017 | https://github.com/williamleif/GraphSAGE ; PyG: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SAGEConv.html | `SAGEConv` | No | DL01 |
| GatedGCN / ResGatedGraphConv | Bresson & Laurent, *Residual Gated Graph ConvNets* | arXiv 2017 / ICLR workshop lineage | https://github.com/xbresson/spatial_graph_convnets ; PyG: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.ResGatedGraphConv.html | `ResGatedGraphConv` | Yes | DL01 |

### 1) GINE — Graph Isomorphism Network with Edge features

- **Paper**: Weihua Hu, Bowen Liu, Joseph Gomes, Marinka Zitnik, Percy Liang, Vijay Pande, Jure Leskovec. “Strategies for Pre-training Graph Neural Networks.” *ICLR 2020*. arXiv:1905.12265. https://arxiv.org/abs/1905.12265
- **Official GitHub**: https://github.com/snap-stanford/pretrain-gnns (MIT/Apache-style research repo; includes pretraining strategies and GIN/GINE backbones).
- **Key idea**: Extends GIN by incorporating edge features into the message before aggregation: `h_v^(k) = MLP((1+eps) * h_v^(k-1) + sum_{u in N(v)} ReLU(h_u^(k-1) + e_{u,v}))`. This makes bond information explicit, which plain GIN ignores.
- **Why relevant to MDM2/chemistry**: Bond type, aromaticity, conjugation, ring membership, and stereo are encoded in our 6-dim edge features; GINE can use them directly.
- **Project implementation**: `GINENet` in DL01 (`build_notebooks.py` `MODELS_NB1_SRC`). 3 GINEConv layers, hidden 128, `train_eps=True`, `edge_dim=6`, BN+ReLU, mean+max graph pooling, MLP head. Uses `d.edge_attr.float()`.
- **Input**: PyG graph `x=[N,78], edge_index=[2,E], edge_attr=[E,6], batch`.

### 2) GraphSAGE — Inductive Representation Learning

- **Paper**: William L. Hamilton, Rex Ying, Jure Leskovec. “Inductive Representation Learning on Large Graphs.” *NeurIPS 2017*. arXiv:1706.02216. https://arxiv.org/abs/1706.02216
- **Official GitHub**: https://github.com/williamleif/GraphSAGE
- **Key idea**: Learns aggregator functions (mean/LSTM/pool) to generate node embeddings by sampling and aggregating neighborhood features; supports inductive generalization to unseen nodes/graphs. In small molecular graphs we use full-neighborhood mean aggregation.
- **Why relevant**: Strong, simple neighborhood-aggregation baseline; useful to compare against attention/edge-aware models on small scaffold-split datasets.
- **Project implementation**: `GraphSAGENet` in DL01. 3 SAGEConv layers, hidden 128, BN+ReLU+dropout, mean+max pooling, MLP head. Does not consume `edge_attr` (SAGEConv has no edge feature argument).
- **Input**: `x, edge_index, batch`.

### 3) GatedGCN / Residual Gated Graph ConvNet

- **Paper**: Xavier Bresson, Thomas Laurent. “Residual Gated Graph ConvNets.” arXiv:1711.07553, 2017. https://arxiv.org/abs/1711.07553
- **Reference GitHub**: https://github.com/xbresson/spatial_graph_convnets (original spatial graph convnet code); PyG implementation: `ResGatedGraphConv`.
- **Key idea**: Edge gates control message passing: `h_i = W1 h_i + sum_{j in N(i)} eta_{i,j} ⊙ W2 h_j`, with gate `eta_{i,j}=sigmoid(W3 h_i + W4 h_j + W5 e_{i,j})`. Residual connections and edge features make it expressive for molecular graphs.
- **Why relevant**: Explicit edge gating uses bond features and can learn to suppress/weight chemical interactions.
- **Project implementation**: `GatedGCNNet` in DL01. 3 ResGatedGraphConv layers, `edge_dim=6`, hidden 128, BN+ReLU+dropout, mean+max pooling, MLP head.
- **Input**: `x, edge_index, edge_attr, batch`.

## Existing Models (already in notebooks)

| Model | Paper | GitHub / Reference | Edge attr | Notebook |
|---|---|---|---:|---|
| DescriptorMLP | Classic QSAR descriptors + MLP | Project lineage | N/A | DL01 |
| SmilesCNN | Char-CNN style; Hirohara et al./Kimber-inspired SMILES CNNs; project Part_22 | Project lineage | N/A | DL01 |
| GCN | Kipf & Welling, ICLR 2017 | https://github.com/tkipf/gcn | No | DL01 |
| GIN | Xu et al., ICLR 2019 | https://github.com/weihua916/powerful-gnns | No | DL01, DL05 |
| GAT | Veličković et al., ICLR 2018 | https://github.com/PetarV-/GAT | No | DL01 |
| GATv2 | Brody et al., ICLR 2022 | https://github.com/tech-srl/how_attentive_are_gats | No | DL01 |
| HybridGCNGAT | Project hybrid | This repo | No | DL01 |
| AttentiveFP | Xiong et al., J. Med. Chem. 2020 | https://github.com/OpenDrugAI/AttentiveFP | Yes | DL01 |
| D-MPNN | Yang et al., JCIM 2019 (Chemprop) | https://github.com/chemprop/chemprop | Yes | DL01 |
| DeepChemGCN | Ramsundar/DeepChem GraphConv; Altae-Tran et al. lineage | https://github.com/deepchem/deepchem | No (this impl) | DL01 |
| PNA | Corso et al., NeurIPS 2020 | https://github.com/lukecavabarrett/pna | Yes | DL04 |
| GraphTransformer | Shi et al., IJCAI 2021 (TransformerConv) | PyG TransformerConv | Yes | DL04 |
| GPS | Rampášek et al., NeurIPS 2022 | https://github.com/rampasek/GraphGPS | No (mini impl) | DL04 |
| SmilesTransformer | Vaswani et al., NeurIPS 2017 adapted to SMILES | https://github.com/huggingface/transformers (architecture origin) | N/A | DL04 |
| FT-Transformer | Gorishniy et al., NeurIPS 2021 | https://github.com/yandex-research/rtdl-revisiting-models | N/A | DL04 |
| GIN-MolCLR | Wang et al., Nat. Mach. Intell. 2022 | https://github.com/yuyangw/MolCLR | Pretrain then GIN | DL05 |

## Candidate Models Reviewed (not all added yet)

These are research models with GitHub code that are relevant; they were reviewed for fit with the 645-compound, 2D-graph, no-extra-setup constraints.

| Candidate | Paper | GitHub | Why not added now / note |
|---|---|---|---|
| Chemprop v2 (full package) | Heid et al., JCIM 2024; Yang et al., JCIM 2019 | https://github.com/chemprop/chemprop | Full package adds heavy deps/CLI; DL01 already has a D-MPNN-style implementation. Can be run externally. |
| GatedGCN/LSPE/GraphGPS full | Rampášek et al., NeurIPS 2022 | https://github.com/rampasek/GraphGPS | Full GPS needs Laplacian PE and config stack; DL04 has a mini GPS. |
| GROVER | Rong et al., NeurIPS 2020 | https://github.com/tencent-ailab/grover | Large pretrained model; using released weights risks label leakage/provenance issues for this 645-label project. |
| GraphMAE | Hou et al., KDD 2022 | https://github.com/THUDM/GraphMAE | Good self-supervised candidate; can be added as DL05 variant with careful no-leak pretraining. |
| Uni-Mol | Zhou et al., ICLR 2023 | https://github.com/dptech-corp/Uni-Mol | 3D/pretrained; needs conformers and careful leakage audit. |
| SchNet/DimeNet++ | Schütt et al. 2017; Gasteiger et al. 2020 | PyG/DeepChem | Require 3D coordinates/conformers; not available in current 2D SMILES pipeline. |
| MolRep benchmark suite | Rao et al., Patterns 2022 | https://github.com/biomed-AI/MolRep | Useful benchmark reference; not a single model to drop in. |
| KPGT | Li et al., KDD 2022 | https://github.com/lihan97/KPGT | Knowledge-guided pretraining; heavyweight and external deps. |

## Advanced / data-hungry models — reviewed, not used

These are stronger architectures, but each one needs something this project
does not have (large labeled sets, external pretraining with clean provenance,
or 3-D conformers). Listing them as *future work* strengthens the paper;
claiming them as results would not.

| Advanced model | Paper / code | What it needs | Why it is excluded on 645 compounds | When to revisit |
|---|---|---|---|---|
| Graphormer | Ying et al., NeurIPS 2021, https://github.com/microsoft/Graphormer | Large graph corpus + full attention stack | Data-hungry; degree/edge encodings and dense attention overfit this scaffold split | After an external MDM2 set (≥10k labeled) and validation-only pretraining |
| GROVER | Rong et al., NeurIPS 2020, https://github.com/tencent-ailab/grover | Huge unlabeled corpus + released 100M/300M checkpoints | Label-provenance risk; weights may have seen related bioactivity data | Only with a documented label-leakage audit and an external test set |
| Uni-Mol | Zhou et al., ICLR 2023, https://github.com/dptech-corp/Uni-Mol | 3-D conformers + 3-D pretraining | No conformers in the 2-D SMILES pipeline | If conformers are generated and frozen before splitting, with AD reporting |
| SchNet / DimeNet++ / PaiNN / GemNet | Schütt 2017; Gasteiger 2020; Schütt 2021; Gasteiger 2022 (PyG/DeepChem) | 3-D coordinates, distances/angles | Not available from canonical SMILES alone; 645 molecules is too few | Only after a conformer pipeline plus external validation |
| Chemical LMs (ChemBERTa, MoLFormer-XL) | HF `DeepChem/ChemBERTa-77M-MLM`, `ibm/MoLFormer-XL` | External pretrained weights + GPU | Pretraining corpus provenance must stay label-free; use as sensitivity only | NB1 cell 10c already supports them behind `DL_USE_HF=1` for sensitivity analyses |
| GraphMAE / KPGT / full MolCLR scale | Hou et al. KDD 2022; Li et al. KDD 2022; Wang et al. Nat. Mach. Intell. 2022 | Large unlabeled corpora + long pretraining | DL05 runs a reduced MolCLR comparison on ZINC250k; scaling up is a compute decision, not extra evidence by itself | Under the same validation-only protocol, report scratch-vs-pretrained deltas with CIs |
| Full GraphGPS (Laplacian PE, config stack) | Rampášek et al., NeurIPS 2022, https://github.com/rampasek/GraphGPS | Spectral PE + heavy config | DL04 already has a compact GPS; full stack adds complexity without more data | Only if validation CV shows a stable gain over mini-GPS |
| Chemprop v2 (full package) | Heid et al., JCIM 2024, https://github.com/chemprop/chemprop | Extra dependencies + CLI | DL01 already implements the D-MPNN message passing | Useful as an external baseline on an external test set |

**Priority order for making the paper stronger:** (1) external MDM2 test set;
(2) repeated/nested scaffold CV with enrichment metrics and bootstrap CIs;
(3) Y-randomization + AD coverage reported beside every ranking; (4) a
calibrated classifier → regressor two-stage model; (5) conformal/ensemble
uncertainty; (6) larger self-supervised pretraining; 3-D and Graphormer last.
See `../docs/IMPROVEMENTS.md` P0/P1/P3.

## Models still left (not in the 23 used, not in the 14 advanced)

Deduplicated: each entry is a genuinely different design; variants already in
the suite (GIN, D-MPNN, GPS, MolCLR, ChemBERTa) are not repeated here.

| Family | Models left | Feasible on 645 compounds? |
|---|---|---|
| Classic/local GNNs | ChebNet (Defferrard et al. 2016), MoNet/GMMConv (Monti et al. CVPR 2017), GGNN (Li et al. ICLR 2016), GCNII (Chen et al. ICML 2020), JKNet (Xu et al. ICML 2018), APPNP (Gasteiger et al. ICLR 2019), SGC (Wu et al. ICML 2019) | **Yes** for GCNII, JKNet, GGNN, ChebNet, MoNet, APPNP — small-graph baselines; SGC needs larger graphs |
| Message-passing frameworks / readouts | MPNN + Set2Set readout (Gilmer et al. ICML 2017; Vinyals et al. ICLR 2016), GraphNets (Battaglia et al. 2018), virtual-node GIN, SAGPool (Lee et al. ICML 2019), TopKPool (Cangea et al. 2018), DiffPool (Ying et al. NeurIPS 2018) | **Partly** — MPNN/Set2Set and virtual node are feasible; DiffPool/data-driven pooling is data-hungry |
| Self-supervised 2-D (no conformers) | GraphCL (You et al. NeurIPS 2020), JOAO (You et al. ICML 2021), GCC (Qiu et al. KDD 2020), Hu et al. attr/context pretraining (ICLR 2020, same lineage as GINE) | **Yes** — natural extensions of DL05 on ZINC250k under a validation-only protocol |
| SMILES / sequence LMs | SMILES-BERT (Wang et al. 2019), MolBERT (Fabian et al. 2020), SMILES-Transformer (Honda et al. 2019), MolBART (Austin et al. 2021), MolT5 (Edwards et al. 2022) | **Sensitivity only** — external weights; provenance/label-leak audit required |
| Tabular / fingerprint DL | TabNet (Arik & Pfister AAAI 2020), NODE (Popov et al. ICLR 2020), TabTransformer (Huang et al. 2020) | **Yes** — small-capacity tabular models fit 645 rows and are cheap baselines |
| 3-D / equivariant | NequIP (Batzner et al. 2022), MACE (Batatia et al. 2022), SphereNet (Liu et al. 2022), PhysNet (Unke & Meuwly 2019) — all use e3nn-style equivariant operations | **No** — need conformers + external validation |
| Uncertainty / training schemes (not new architectures) | Deep Ensembles (Lakshminarayanan et al. 2017), Bayesian GNN, evidential regression, conformal prediction, multi-task cls+reg head | **Yes** — recommended before any new architecture |

**Best next additions for this project:** GCNII or GGNN (strong small-graph
baselines), GraphCL/JOAO (extends the ZINC250k pretraining comparison), TabNet
or NODE (fingerprint/tabular baseline), and deep ensembles + conformal
prediction for uncertainty. Each must use the existing scaffold CV,
Y-randomization, AD reporting and test-once guardrails.

## Implementation status of the 34 left (2026-11-23) [USER]

All 34 candidates are now encoded **inside the existing five notebooks**, generated
from `build_notebooks.py` (cells S07c / S07d / S16b). Default runs are unchanged:
the original 23-model registry still produces the reported results. Set
`DL_EXTRA_MODELS=1` to register the extra roster; this is a plumbing/capability
addition, **not** a validated performance claim.

- Registered with `DL_EXTRA_MODELS=1`: ChebNet, MoNet/GMMConv, GGNN, GCNII
  (hand-written initial-residual layer — PyG 2.8 has no `GCNIIConv`), JKNet,
  APPNP, SGC, MPNN+Set2Set, GraphNets, virtual-node GIN, SAGPool, TopKPool,
  DiffPool-style, TabTransformer, NODE, TabNet, and five from-scratch SMILES
  Transformer language-model analogs (SMILES-BERT, MolBERT, SMILES-Transformer,
  MolBART, MolT5 — **no external weights**, sensitivity-only).
- DL05 with `DL_SSL_METHOD=graphcl|joao|gcc|hu`: alternative label-free
  augmentation policies reusing the ZINC250k/MolCLR corpus and encoder; default
  `molclr` path untouched. JOAO/Hu are simplified proxies, not the published
  bi-level/attribute objectives.
- `DL_UNCERTAINTY=1`: deep ensembles, MC-dropout Bayesian proxy, conformal
  intervals, evidential proxy, dual-head multi-task diagnostics — all computed
  from validation rows only; the test split is never re-scored.
- 3-D (NequIP, MACE, SphereNet, PhysNet): classes are defined but registration
  requires both conformer coordinates (`d.pos`) and `DL_USE_3D=1`; e3nn is not
  installed here, so the guards raise a clear `NotImplementedError` rather than
  fabricate results. They become runnable when 3-D data arrives.

## How to validate added models

```bash
cd 06_deep_learning_models
python3 build_notebooks.py
python3 run_smoke.py DL_01   # quick sanity
python3 run_fast.py DL_01    # CPU-fast real-data check
```

Check `smoke_logs/DL_01_Graph_Models_No_Fingerprints.log` and `outputs/dl01/*/leaderboard_test.csv` for `GINE`, `GraphSAGE`, `GatedGCN`.
