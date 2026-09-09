# References — Models, Code & Data Behind Parts 9–14

Copy-pasteable citation pack for the manuscript. Each entry: the paper to cite,
where the code came from, what went into the model in this repo, and the
one-line conclusion to reference.

## Models used in this repo

### [1] Random Forest baseline — Parts 6/8 (your main paper)
- **Cite:** Budha et al., "Machine Learning-Guided Discovery of Natural MDM2
  Inhibitors: A Multistage In Silico Pipeline from Screening to ADMET Profiling,"
  *Adv. Theory Simul.* 2025. DOI: 10.1002/adts.202501502
- **Code:** `Natural_MDM2_Inhibitor_Discovery_using_ML/Part_6` (scikit-learn)
- **Inputs:** Morgan fingerprints (radius 3, 512-bit), 645 ChEMBL compounds
  (343 active / 302 inactive, Lipinski-filtered)
- **Conclusion:** ROC-AUC **0.960**, F1 0.943, MCC 0.876; `prob > 0.6` → 116
  COCONUT hits → 2 leads (docking −10.0 / −9.6 kcal/mol). The bar every GNN
  below is measured against.

### [2] GCN from scratch — Part 9
- **Cite:** Kipf & Welling, "Semi-Supervised Classification with Graph
  Convolutional Networks," *ICLR* 2017 — pipeline following Yasir et al.,
  *Future Pharmacol.* 2025 (GraphConvMol on ChEMBL + docking + MD).
- **Code:** from-scratch PyTorch Geometric in `Part_9/` (this repo);
  method ref: `github.com/deepchem/deepchem`
- **Inputs:** 78-dim atom features, molecular graphs, same 645 compounds
- **Conclusion:** ROC-AUC **0.952**, F1 0.923, MCC 0.830; 5-fold CV
  0.945 ± 0.019; Y-scrambling ~0.5 (no chance correlation)

### [3] GIN (MolCLR architecture) — Part 10
- **Cite:** Xu et al., "How Powerful are Graph Neural Networks?," *ICLR* 2019
  (GIN) — pretraining architecture: Wang et al., "MolCLR: Molecular
  Contrastive Learning of Representations via Graph Neural Networks,"
  *NeurIPS* 2020.
- **Code:** from-scratch encoder in `Part_10/` (this repo); weights attempted
  from `github.com/yuyangw/MolCLR` (28/63 layers matched → trained from
  scratch with the same architecture, documented in-notebook)
- **Inputs:** Same graphs; 5-layer GIN, hidden 300, 465,091 params
- **Conclusion:** ROC-AUC **0.947**, F1 0.913, **sensitivity 0.986** (best
  recall of all models); 5-fold CV 0.937 ± 0.017

### [4] AttentiveFP — Part 12
- **Cite:** Xiong et al., "Pushing the Boundaries of Molecular Representation
  for Drug Discovery with the Graph Attention Mechanism," *J. Med. Chem.*
  2020, 63(16):8749–8760.
- **Code:** PyG port in `Part_12/` of `github.com/OpenDrugAI/AttentiveFP`
  (original PyTorch; also wrapped in DeepChem / DGL-LifeSci);
  hidden 128, K=2 layers, T=2 readout steps (paper default 200-dim reduced
  for n=645)
- **Inputs:** 78-dim atoms + 6-dim bonds (type, conjugation, ring);
  same 645 compounds
- **Conclusion:** Paper reports SOTA + interpretability (learns nonlocal
  intramolecular interactions); ours: fill in after Colab run
  (target ≥ 0.94 to stand beside Parts 9/10)

### [5] Hybrid GCN+GAT — Part 13
- **Cite:** HDTI-IC50 hybrid GCN+GAT for p53-inhibitor IC50 prediction,
  *J. Comput.-Aided Mol. Des.* 2025 (MAE 0.1, RMSE 0.19, R² 0.8, beating
  single GCN/GAT) — GCN: Kipf & Welling, ICLR 2017 — GAT: Veličković et al.,
  "Graph Attention Networks," ICLR 2018.
- **Code:** from-scratch PyG in `Part_13/` (this repo), ~60k params
  (deliberately small for n=645)
- **Inputs:** Same 78-dim graphs, same 645 compounds
- **Conclusion:** HDTI paper proves the hybrid beats either part alone on
  p53 data; ours: fill in after Colab run

### [6] D-MPNN / Chemprop (backup, not built)
- **Cite:** Yang et al., *J. Chem. Inf. Model.* 2019, 59:3370–3388 (theory);
  Heid et al., *J. Chem. Inf. Model.* 2023/2024 (software v1).
- **Code:** `github.com/chemprop/chemprop` (MIT; docs: chemprop.readthedocs.io)
- **Inputs:** SMILES → RDKit graph; default atom/bond features
- **Conclusion:** SOTA incl. MoleculeNet/SAMPL; held as fallback (strong
  generally, but no MDM2-region paper and heavier dependency)

### [7] All-models ensemble — Part 14 (novel integration)
- **Cite:** ensemble of [1]–[5] above (majority consensus + soft vote);
  methodological precedent: multi-algorithm consensus in MDM2 SBVS
  (*Frontiers Drug Discov.* 2025/26, PLEC-RF/SVM voting).
- **Code:** `Part_14/` (this repo, read-only over Parts 6/8/9/10/12/13)
- **Inputs:** Every model probability + paper RF-116 list
- **Conclusion:** Overlap-vs-116 + new-candidates count (fill in after runs)

## Same-region supporting work (discussion section)

- PLEC/Grid + RF/SVM/XGB/ANN/DNN at the MDM2 p53 site; PLEC-RF/SVM beat
  Smina, CNN-Score, SCORCH — *Frontiers Drug Discov.* 2025/26.
- Selective-cleaning ML → pIC50 R² 0.87, 24k-compound repurposing screen,
  docking + 100 ns MD — Akmal & Wong, *Molecules* 2025.
- mol2vec + 10 regressors, KNN R² 0.74, MDM2pred web app —
  Ghafoor & Yıldız, 2023 (preprint).
- Glide HTVS→SP→XP + 100 ns MD + DFT, hits −9.1 to −8.6 kcal/mol —
  *Sci. Rep.* 2025 (ASINEX p53-MDM2).
- Deliberately excluded: standalone GAT (strict subset of AttentiveFP),
  Hot2Mol generative PPI design (bioRxiv 2024, wrong task), 3D-CNN scoring
  (needs 154k docked poses first — listed as future work).

## Data & tools

- ChEMBL bioactivity data (EBI, accessed April 2024) — training labels
- COCONUT natural products, `coconut_csv-03-2025.csv` (March 2025 snapshot) —
  Sorokina et al., *J. Cheminform.* 2021
- RDKit (`rdkit.org`), PyTorch Geometric (`pyg.org`), scikit-learn
