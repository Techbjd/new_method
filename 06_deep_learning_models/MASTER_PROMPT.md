# MASTER PROMPT — `06_deep_learning_models/` (Deep-Learning Notebook Suite)

> **Provenance:** [USER] request (Nov 2026) — "Create a new folder deep learning models with 3 separate
> notebooks (Arjun style) training MDM2 data with ALL deep learning models, fine-tuning stepwise, full
> GPU/RAM/CPU utilization, preload chemistry primer, and Y-randomization in all code. NB1 no Morgan/MACCS,
> NB2 with Morgan, NB3 with MACCS. Write a proper master prompt first; record everything in the diary."
> This file IS that master prompt. The notebooks in this folder were generated from it.
>
> **2026-11-22 update:** the suite now has **five** notebooks (DL 01–05); DL 04 (Tier-1
> architectures) and DL 05 (MolCLR pretraining) were added on 2026-11-13. The
> portability/runtime audit is recorded in §10 and in `README.md`.

---

## 1. ROLE & GOAL

You are building **five self-contained, Colab/Kaggle-ready Jupyter notebooks** that train
**all deep-learning model families** on the **MDM2 p53 inhibitor dataset (645 compounds)**
following the rigorous protocol already validated in this repository (Parts 9→22).
Each notebook is written in **Arjun sir's teaching style**: a titled markdown header with
`**Reference:**` and `**Pipeline:**` lines, then strictly **numbered sections** (`## 1. …`,
`## 2. …`) each followed by its code cell.

DL 01–03 differ **only** in featurization; DL 04 adds Tier-1 architectures and DL 05 adds
MolCLR pretraining:

| # | Notebook | Featurization | Models trained |
|---|----------|---------------|----------------|
| 1 | `DL_01_Graph_Models_No_Fingerprints.ipynb` | RDKit molecular **graphs** (78-dim atoms, 6-dim bonds) + 10 descriptors + raw SMILES char codes — **no fingerprints** | DescriptorMLP, SmilesCNN, GCN, GIN, GINE, GAT, GATv2, GraphSAGE, GatedGCN, HybridGCNGAT, AttentiveFP, D-MPNN, DeepChemGCN |
| 2 | `DL_02_Morgan_Fingerprint_DL.ipynb` | **Morgan/ECFP4** r=2, 2048 bits (+ graphs for the hybrid) | FPMlpShallow, FPMlpDeep, ResidualMlp, FingerPrintCNN, HybridGCNFp(Morgan) |
| 3 | `DL_03_MACCS_Keys_DL.ipynb` | **MACCS keys** 167 bits (+ graphs for the hybrid) | FPMlpShallow, FPMlpDeep, ResidualMlp, FingerPrintCNN, HybridGCNFp(MACCS) |
| 4 | `DL_04_Tier1_Advanced_Architectures.ipynb` | graphs + descriptors/chars as needed | PNA, GraphTransformer, GPS, SmilesTransformer, FTTransformer |
| 5 | `DL_05_MolCLR_Pretrain_Finetune.ipynb` | graphs; MolCLR pretraining on unlabeled open-source chemistry SMILES (default ZINC250k, not screening COCONUT) | GIN (scratch), GIN-MolCLR |

## 2. INPUTS (never move/delete originals)

- `00_data/training/ro5_properties_filtered.csv` — 645 compounds
  (`molecule_chembl_id, bioactivity_class ∈ {Active,Inactive}, pIC50, canonical_smiles, …`; 343 Active).
  Path fallbacks: repo-relative → GitHub raw `Techbjd/new_method/main/00_data/training/…`.
- `00_data/screening/screening_results.csv` — 154,647 COCONUT rows (optional screening cell only).
- `00_data/config/cuts.json` — `ad_cut=0.40`, `ad_cut_strict=0.52`, `gnn_hit_cut=7.0` (optional screening cell only).

## 3. TASK FLAG

Every notebook supports both tasks with one flag, `TASK = "cls"` (default) or `"reg"`:

- `cls`: label = `bioactivity_class=="Active"`; head `out_dim=2`; loss = cross-entropy on logits;
  score = softmax(logits)[:,1]; primary metric **ROC-AUC** (also AP, balanced accuracy).
- `reg`: label = standardized pIC50 (mean/std from **train only**); head `out_dim=1`; loss = MSE;
  predictions mapped back to pIC50; primary metric **RMSE** (also MAE, R²).

## 4. NOTEBOOK CELL MAP (identical skeleton in all five)

0. **Title markdown** — Arjun style: `# DL 0X: …`, `**Reference:**` papers per model,
   `**Pipeline:** SMILES → featurize → split → train → tune → CV → Y-randomization → test (once)`,
   plus dataset/guardrail notes.
1. **Chemistry primer ("preload chemistry")** — markdown: atoms/bonds/molecules, SMILES,
   molecular graphs, fingerprints (ECFP/Morgan and/or MACCS), descriptors, pIC50 & bioactivity
   classes, ROC-AUC/AP, scaffold split & leakage, Y-randomization, applicability domain.
2. **Setup + installs** — idempotent (skip pip if imports already work); env-var thread limits set
   BEFORE torch import; works on Colab/Kaggle/local.
3. **Reproducibility + imports** — `RANDOM_SEED=57` for random/numpy/torch(+cuda), `SMOKE` flag
   read from env `DL_SMOKE` (0/1), `TASK`, timings helper.
4. **Full GPU/RAM/CPU utilization** — device pick (cuda:0/cuda:1…), `cudnn.benchmark=True`,
   TF32 + `float32_matmul_precision("high")`, AMP autocast/GradScaler (CUDA only), thread scaling,
   `gpu_report()` (mem/util per GPU via pynvml if present), `free_mem()`, DataLoader factory
   (pin_memory, num_workers, persistent_workers, prefetch), batch auto-scaler by free VRAM,
   `run_parallel()` dual-GPU ThreadPoolExecutor pattern from Part_20, `autocast` context helper.
5. **Load data + EDA** — load 645-row table w/ fallbacks, sanity asserts, class balance /
   pIC50 / MW / logP plots (saved to `outputs/`).
6. **Featurization + leak-free scaffold split** — per-notebook featurizers; Murcko scaffold
   (**non-isomeric**, stereoisomers stay together), exact Part_20 algorithm: whole scaffold
   groups sorted largest-first greedily fill the 129-compound test set (`round(0.20 × 645)`);
   the remaining 516 are split **stratified by class** into train 464 / validation 52 with
   `random_state=57`; assert zero index and canonical-SMILES overlap between train and test;
   5-fold `GroupKFold` fold sizes printed; scaler (if any) fit on train only. The test set is
   scaffold-disjoint and scored once; validation is a stratified random 10% of the non-test
   remainder (the canonical Part_20 protocol), not a second scaffold holdout.
7. **Model definitions** — classes copied/adapted from validated repo code with a
   `DL_PAPERS` lineage dict printed (see §6). All architectures take `in_dim, out_dim` **by keyword**
   (the positional-arg bug that shrank GCN to 411 params must not repeat). `DL_MODELS` factory dict.
8. **Training engine** — generic `fit_model()` for graph/array inputs: Adam, val-only early
   stopping (patience 10; 3 in smoke), best-state restore, AMP autocast + grad-clip 5.0,
   ReduceLROnPlateau optional, history recorded; `evaluate()` on val/test loaders.
9. **Baseline training (all models)** — default config, 150 epochs (SMOKE: 3; FAST: 8);
   val-selection only; **test scored once** after early stopping; results table printed+saved.
10. **Stepwise fine-tuning** — stage ladder, all selection on **validation only**:
    - **Stage A** lr ∈ {1e-3, 5e-4, 3e-4, 1e-4} (SMOKE: 1 value)
    - **Stage B** dropout {0.1, 0.3} × weight_decay {0, 1e-4} (SMOKE: 1 combo)
    - **Stage C** capacity hidden {64, 128, 256} (SMOKE: 1)
    - **Stage D** batch {64, 256} on GPU / {32, 64} CPU (SMOKE: 1)
    Each stage = short runs (`FT_EPOCHS=60`, patience 8; SMOKE 3/2) keeping the best val config;
    final refit with full epochs. Optional **pretrained warm-start** extras:
    D-MPNN body from `stage1a.pt` (searched; skipped with message if absent), and
    `DL_USE_HF=1` ChemBERTa/MoLFormer ladder (freeze head-only → unfreeze last 2 →
    full unfreeze at low lr) guarded by try/except install of `transformers`.
11. **5-fold GroupKFold CV** — folds over scaffold groups of train+val, fresh model per fold with
    the tuned config, mean±std of the primary metric; dual-GPU parallel across folds when >1 GPU;
    folds ≥ 5 (SMOKE: 2, epochs 3).
12. **Y-randomization** — protocol of Part_9 §8: 10 shuffles (`sk_shuffle(random_state=si)`),
    fresh model per shuffle, `YRAND_EPOCHS=100` (SMOKE: 2 shuffles × 3 epochs on 1 model),
    per-shuffle AUC + mean±std table saved; compare vs real val AUC. Present **in all five notebooks**.
13. **Test — scored ONCE + bootstrap CI** — final best (CV-selected) model evaluated on the untouched
    test split; 2000-resample bootstrap 95% CI of the primary metric (SMOKE: 200); ROC/PR or
    pred-vs-true plot saved.
14. **Summary + saves** — baseline/fine-tune/CV/Y-rand/test tables as CSV + one JSON bundle +
    leaderboard plot; everything under `outputs/<nb_id>/{smoke,full}/`.
15. **Optional screening** — `DL_RUN_SCREENING=1` flag: chunked (5,000) prediction over the 154k
    COCONUT library with the best model + **Tanimoto applicability domain** (Morgan r=3/512 vs
    train set, cut from `cuts.json`), `AD`/`AD_strict` columns, CSV saved. Smoke/fast skip screening.
16. **Final diary-style summary** — prints timestamp, device, task, split sizes, per-model table,
    files written, and a run-record block meant to be pasted into `docs/JOURNEY.md`.

## 5. MODEL ARCHITECTURES (mirroring validated repo code)

- **78-dim atom feats:** atomic_num(53) | degree(6) | formal_charge(6) | numHs(5) | hybridization(5)
  | aromatic(1) | in_ring(1) | chiral(1); **6-dim bonds:** single/double/triple/aromatic/conj/ring;
  edges emitted as adjacent i→j, j→i pairs (`rev = arange ^ 1`).
- **GCNNet** Kipf&Welling 2017 — 3×GCNConv+BN, mean⊕max pool, head 256→64→out.
- **GINNet** Xu 2019 / MolCLR — GINConv(Linear+BN+ReLU)×layers, hidden 300, head 600→128→out.
- **GATNet / GATv2** Veličković 2018 / Brody 2022 — 2 layers, heads=4 concat, BN, head 1024→64→out.
- **HybridGCNGAT** — parallel GCN & GAT branches, pooled concat → head.
- **AttentiveFPNet** Xiong 2020 — PyG AttentiveFP(in 78, hidden 128, edge 6, layers 3, steps 2).
- **DMPNN** Yang 2019 (Chemprop) — directed messages `m = relu(W_i shortcut + W_h(in_sum[src] − m[rev]))`,
  depth 3, W_o atom readout, mean⊕max pool.
- **SmilesCNN** — char vocab + PAD/UNK/BOS/EOS, emb 96, Conv1d k=3/5/7, max-pool concat, head 384→64→out.
- **DescriptorMLP / FPMlpShallow / FPMlpDeep / ResidualMlp / FingerPrintCNN** — plain DL on
  descriptor/fingerprint vectors (NB1 descriptors 10; NB2 Morgan 2048; NB3 MACCS 167), incl.
  a 1-D CNN over the bit vector and a residual MLP.
- **HybridGCNFp** — graph encoder (GCN trunk) late-fused with the fingerprint branch of that notebook.
- `HFForSequence` (optional, NB1) — HF `AutoModel` + CLS pooling + head, 3-stage unfreeze fine-tune.

## 6. PROTOCOL GUARDRAILS (non-negotiable)

1. `RANDOM_SEED=57` everywhere; the split is deterministic and leak-free (assert in code).
2. **Validation-only** early stopping, model selection, and fine-tune decisions.
3. **Test split is scored exactly once**, at the very end (§13), never during tuning.
4. CV folds are scaffold-grouped; no fingerprint scaler leakage (fit on train only).
5. Y-randomization is a sanity check, not a model-selection tool.
6. No pretrained weights that saw the 645 labels (stage1a.pt / ChemBERTa are label-free corpora).
7. Report applicability domain coverage in the screening cell; a high pIC50 prediction is a
   hypothesis, not a hit.
8. Every notebook must run **end-to-end in SMOKE mode on CPU** before delivery.

## 7. SMOKE MODE (local validation contract)

`SMOKE = os.environ.get("DL_SMOKE","0")=="1"`: subset train/val/test (60/32/24 rows), 3 epochs,
patience 2, 1 option per FT stage, 2 CV folds (first 3 models), 2 Y-rand shuffles (first model),
bootstrap 200, screening skipped.
Smoke executes each notebook via `nbclient` with `DL_SMOKE=1`, requires **zero errors**,
then outputs are cleared and the clean notebook saved.

## 8. DELIVERABLES & VALIDATION CHECKLIST

- [ ] `06_deep_learning_models/MASTER_PROMPT.md` (this file)
- [ ] `06_deep_learning_models/README.md` — folder map + provenance labels ([ARJUN]/[USER]/[ADAPTED]/[GENERATED])
- [ ] 5 notebooks, `nbformat.validate` clean, every code cell `ast.parse` clean
- [ ] All five pass SMOKE end-to-end (nbclient, CPU, `DL_SMOKE=1`)
- [ ] Outputs cleared after smoke; smoke artifacts kept under `outputs/<nb_id>/smoke/`
- [ ] Full record appended to `diary.md` and `docs/JOURNEY.md` ("fully, not single thing")

---

## 9. Addendum (2026-11-13, later same day): DL 04 + DL 05 extension

**User request:** "add it tier 1 and tier 2 — make two a separate notebook with all the
information where model come from."

### DL 04 — `DL_04_Tier1_Advanced_Architectures.ipynb` (Tier 1, 5 models)

| Model | Source paper | Kind |
|---|---|---|
| PNA | Corso et al., NeurIPS 2020 — Principal Neighbourhood Aggregation | graph |
| GraphTransformer | Shi et al. 2021 — TransformerConv (edge-aware multi-head attention) | graph |
| GPS | Rampášek et al., Nature Machine Intelligence 2022 — local MPNN + global attention | graph |
| SmilesTransformer | Vaswani et al., NeurIPS 2017 — char-level transformer encoder | array/chars |
| FT-Transformer | Gorishniy et al., NeurIPS 2021 — Revisiting Deep Learning for Tabular Data | array/desc |

Same 17-cell flow, same leak-free 464/52/129 scaffold split, same engine/FT ladder/CV/
Y-randomization/test-once; PNA degree histogram from TRAIN split only; explicitly NO
Graphormer/3-D models (too few molecules).

### DL 05 — `DL_05_MolCLR_Pretrain_Finetune.ipynb` (Tier 2, 2 models)

- **Cell 7b — MolCLR self-supervised pretraining** (Wang et al., *Nature Machine
  Intelligence* 2022): two augmented views per unlabeled molecule (atom masking 15%,
  bond deletion 15%, subgraph removal 25%), NT-Xent loss (SimCLR, Chen et al. ICML 2020,
  T=0.2) on a GIN encoder (1500 molecules × 2 epochs in smoke; 30,000 × 30 full).
  Default corpus is ZINC250k (standard open-source, downloaded/cached), **not** the
  COCONUT screening library; `DL_PRETRAIN_SOURCE` can select `zinc250k`, `chembl_small`,
  `coconut` (explicit opt-in), or a custom path/URL. No MDM2 labels enter pretraining;
  the pretrained trunk
  is saved as `molclr_gin_pretrain.pt` and warm-starts "GIN-MolCLR" via the generalized
  `WARM_START_NAME/WARM_START_SD` hook in the engine (D-MPNN stage1a hook unchanged).
- **Models:** `GIN (scratch)` vs `GIN-MolCLR` — same class, same split, same engine;
  only the initial weights differ, so val/CV/test deltas isolate the value of
  pretraining.
- Same guardrails as all other notebooks (scaffold split, val-only selection,
  Y-randomization, test scored once, softmax prob columns, test_predictions.csv).

---

## 10. Addendum (2026-11-22): portability and CPU-runtime audit

All five notebooks were executed end-to-end on a CPU-only Python 3.13 machine
and passed smoke mode (`python3 run_smoke.py`, exit 0). The following fixes
make the suite work on a fresh machine without manual edits:

- **Installs are idempotent and interpreter-correct**: `python -m pip`; missing
  `torch`, `torch_geometric`, `rdkit`, `scikit-learn`, `matplotlib`, `pandas`
  are installed only when absent. `rdkit-pypi` (no Python 3.13 wheel) was
  replaced by `rdkit`.
- **Torch index is detected**: CPU-only machines use
  `https://download.pytorch.org/whl/cpu`; GPU machines/macOS use default PyPI.
- **DeepChem is opt-in** (`DL_INSTALL_DEEPCHEM=1`) and re-checks the import
  inside its cell. The old unconditional `pip install deepchem` / `dgl` calls
  were removed; the pure-torch `DeepChemGCN` replica always runs.
- **Hugging Face is opt-in** (`DL_USE_HF=1`); `transformers` is installed
  before the import, so enabling it on a fresh machine works.
- **DL05 pretraining streams batches** instead of prebuilding 60,000 PyG
  graphs, avoiding OOM on 7 GB RAM. Full profile: 30,000 molecules × 30
  epochs; fast: 4,000 × 3; smoke: 1,500 × 2.
- **Data fallbacks** include repo-relative paths, `/content/...`, and both
  GitHub raw URLs for training and screening CSVs.
- **Screening is opt-in** (`DL_RUN_SCREENING=1`), chunked, and robust to
  invalid molecules/missing fingerprints; it is skipped in smoke/fast.
- **Bootstrap CIs** guard the single-class-resample case.
- **Zero-bond molecules** produce correctly shaped `(2,0)` edge indices and
  `(0,6)` edge features.
- **CPU cap**: thread count is capped at 8 to avoid oversubscription on
  many-core machines.
- **MolCLR warm-start is shape-safe**: when fine-tuning explores a different
  hidden size, incompatible pretrained tensors are skipped (logged) instead of
  crashing with a size mismatch.
- **Split warning**: the split cell reports test Murcko scaffold count and
  warns when the test set has <=10 scaffolds (the canonical 129-test holdout
  has 5 scaffolds; high AUC can reflect scaffold/series separation).
- **Runner outputs**: `run_smoke.py`/`run_fast.py` keep notebook outputs by
  default; set `DL_CLEAR_OUTPUTS=1` for clean deliverables.

New profiles and runners:

| Command | Profile | Scope |
|---|---|---|
| `python3 run_smoke.py` | `DL_SMOKE=1` | 24 test rows, 3 epochs, 2 CV folds, 2 Y-rand, 200 bootstraps |
| `python3 run_fast.py` | `DL_FAST=1` | all 645 rows, every model, 8 epochs, 5-epoch FT, 2 CV folds, 2 Y-rand, 300 bootstraps |
| unset both | full | 150 epochs, 60-epoch FT, 5 CV folds, 10 Y-rand, 2,000 bootstraps |

Validation (2026-11-22, CPU, Python 3.13.9, torch 2.13.0+cpu):
- Smoke: DL01 0.6 min, DL02 0.3, DL03 0.2, DL04 0.7, DL05 0.3 — all pass, split 464/52/129, no tracebacks.
- Fast (all 645 rows, every model): DL01 PASS 9.0 min AUC 0.9989 [0.9955,1.0000]; DL02 PASS 2.3 min AUC 0.9933 [0.9753,1.0000]; DL03 PASS 1.0 min AUC 0.9899 [0.9701,1.0000]; DL04 PASS 8.3 min AUC 0.9871 [0.9639,1.0000]; DL05 PASS 1.3 min AUC 0.9981 [0.9941,1.0000]. Fast runs kept outputs/logs in `fast_logs/` and `outputs/dl0X/fast/`.
- 2026-11-23 [USER]: the 34 remaining reviewed models are now defined inside the same five notebooks via `build_notebooks.py` cells S07c/S07d/S16b, all opt-in (`DL_EXTRA_MODELS=1`, `DL_SSL_METHOD`, `DL_UNCERTAINTY=1`, `DL_USE_3D=1`). The numbers above are the **default 23-model** runs and are unchanged; no AUC above refers to an extra model.
- Caveat: the canonical 129-test holdout has only 5 Murcko scaffolds, so high fast-profile AUCs are scaffold-holdout sanity/plumbing signals, not prospective performance. Full-GPU protocol, real DeepChem/TensorFlow, HF, and 154k screening remain unexercised.
