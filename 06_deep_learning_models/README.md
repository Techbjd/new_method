# 06_deep_learning_models — Deep Learning on MDM2 (five notebooks)

`[USER]` folder. Built 2026-11-13, portability and runtime audit 2026-11-22.
One folder with **five separate Arjun-style notebooks** training the
645-compound MDM2 table with the **full deep-learning model roster**, including
stepwise fine-tuning, full GPU/RAM/CPU utilization, a chemistry primer
("preload chemistry") and **Y-randomization in every notebook**.

Everything here is NEW code (provenance labels below); the originals in
`02_gnn_models/`, `01_traditional_ml/`, `Part_9/`, `Part_20_Fixed/` were not
touched.

## Read this before quoting any number (paper-facing caveat)

> **Fast AUCs are plumbing/sanity results on a 5-scaffold, 129-compound holdout
> — not prospective performance.** Use repeated/nested scaffold CV,
> Y-randomization, applicability-domain (AD) coverage, enrichment metrics and
> external validation before making any potency or generalization claim.

Why this belongs in the paper:

- The canonical test split is scaffold-disjoint but has **only 5 Murcko
  scaffolds** (40 all-inactive, 34 all-inactive, 31 mostly active, 21 mostly
  active, 3 mixed). A high single-split AUC can be scaffold/series separation,
  not molecule-level generalization.
- The CPU-fast profile uses **8 epochs, 2 CV folds and 2 Y-randomization
  shuffles**. It proves the training → fine-tuning → CV → Y-randomization →
  test-once → AD plumbing works end to end; it is not the full protocol
  (150 epochs, 5 folds, 10 shuffles, 2,000 bootstraps).
- **Y-randomization is a sanity check, not model selection.** Passing it only
  says the labels carry real signal.
- **AD is a screening filter, not a performance metric.** About 99% of the
  154k COCONUT library lies outside the strict AD (median Tanimoto 0.217 vs
  cuts 0.40/0.52); report AD columns with every ranking.
- **No external test set was run here.** In the paper, report precision@10,
  EF1%, BEDROC and bootstrap CIs next to R²/RMSE/MAE, and describe these DL
  results as computational sanity/sensitivity analyses until an external MDM2
  set or a prospective assay supports them.

Project-level versions: `docs/IMPROVEMENTS.md` (P0 items 2–5, P1 item 8,
P3 item 20) and `docs/JOURNEY.md` §12–13.

## Original repository

Arjun sir's original project is in this repo, untouched:
`Natural_MDM2_Inhibitor_Discovery_using_ML/` — cloned 2026-09-07 from
`https://github.com/arjunpahi/Natural_MDM2_Inhibitor_Discovery_using_ML`.
`Part_9_GNN_DeepChem.ipynb` there is the origin of the DeepChem GraphConv GCN;
`02_gnn_models/models/` holds the `[ADAPTED]` copies of its model scripts.
Every notebook's title markdown also states this location and the data
lineage, so the notebooks are self-documenting when shared.

## Folder map

| File | Provenance | What it is |
|---|---|---|
| `MASTER_PROMPT.md` | `[USER]` | Master build spec: role/goal, five-notebook roster, cell map, architectures, guardrails, smoke contract, checklist |
| `DL_01_Graph_Models_No_Fingerprints.ipynb` | `[USER]` | Notebook 1 — NO fingerprints. Graph + descriptor + sequence models (13 models incl. GINE, GraphSAGE, GatedGCN, DeepChemGCN) |
| `DL_02_Morgan_Fingerprint_DL.ipynb` | `[USER]` | Notebook 2 — WITH Morgan (ECFP4 r=2, 2048 bits). 5 models |
| `DL_03_MACCS_Keys_DL.ipynb` | `[USER]` | Notebook 3 — WITH MACCS keys (167 bits). Same 5 models |
| `DL_04_Tier1_Advanced_Architectures.ipynb` | `[USER]` | Notebook 4 — Tier 1 extension: PNA, Graph Transformer, GPS, SMILES Transformer, FT-Transformer (paper lineage in title + primer + `DL_PAPERS`) |
| `DL_05_MolCLR_Pretrain_Finetune.ipynb` | `[USER]` | Notebook 5 — Tier 2: MolCLR self-supervised pretraining on a standard open-source unlabeled chemistry corpus (default ZINC250k, downloaded/cached; not screening COCONUT) → fine-tune, GIN scratch vs GIN-MolCLR |
| `build_notebooks.py` | `[GENERATED]` | Builder that regenerates all five notebooks from cell templates (`python3 build_notebooks.py`); validates nbformat + `ast.parse` per cell |
| `run_smoke.py` | `[GENERATED]` | Local smoke runner (`DL_SMOKE=1`, nbclient, timeout 2400 s); validates each notebook end-to-end and clears outputs on success |
| `run_fast.py` | `[GENERATED]` | CPU-fast real-data runner (`DL_SMOKE=0`, `DL_FAST=1`); all 645 compounds, every model, reduced epochs/CV; writes `fast_logs/` |
| `validation_package.py` | `[USER]` + `[GENERATED]` | Repeated/nested scaffold CV on train+val only, plus descriptive enrichment/bootstrap CIs from existing fast test predictions; writes `outputs/validation_package/` |
| `screening_audit.py` | `[USER]` + `[GENERATED]` | Read-only integrity + gate-by-gate AD funnel with parent/scaffold dedup; writes `outputs/screening_audit/` |
| `smoke_logs/` | `[GENERATED]` | Full stdout logs of the five smoke runs |
| `fast_logs/` | `[GENERATED]` | Full stdout logs of the CPU-fast real-data runs |
| `outputs/` | `[GENERATED]` | Per-profile artifacts: baseline/finetune/CV/Y-randomization CSVs, `eda.png`, `leaderboard.png`, `test_roc.png`, `dl0X_results.json`, `dl0X_diary_block.txt` |

Originals that the notebook code adapts (all kept untouched):

- `[ADAPTED]` graph featurization + GCN/GIN/GAT/Hybrid/AttentiveFP/D-MPNN/CharCNN
  architectures from `02_gnn_models/models/` (78-dim atom features, 6-dim bond
  features, Chemprop-style directed-message D-MPNN, MolCLR-style GIN).
- `[ADAPTED]` descriptor/fingerprint helpers and the leak-free scaffold split
  concept from `01_traditional_ml/models/featurize.py`.
- `[ADAPTED]` Y-randomization protocol (10 shuffles × 100 epochs) from
  `Part_9_GNN_DeepChem.ipynb`.
- `[ADAPTED]` exact 464/52/129 greedy largest-scaffold-first test split,
  dual-GPU `ThreadPoolExecutor` pattern, Tanimoto applicability-domain logic
  (Morgan r3/512, cut 0.40/0.52 from `cuts.json`) from `Part_20_Fixed/`.
- `[ADAPTED]` pretrained-warm-start hook idea (`stage1a.pt` → D-MPNN) from
  `pretrain_stage1a_npass.ipynb` (prints "not found → skipped" when absent).

## The five notebooks

| Notebook | Featurization | Models |
|---|---|---|
| `DL_01` | 78-dim graph feats + 10 RDKit descriptors (train-fit scaler) + char-sequence (max_len 96, vocab 80) | GCN, GIN, GINE, GAT, GATv2, GraphSAGE, GatedGCN, HybridGCN+GAT, AttentiveFP, D-MPNN, SmilesCNN, DescriptorMLP, DeepChemGCN |
| `DL_02` | Morgan ECFP4 r=2 / 2048 bits (+ graphs for the hybrid) | FPMlpShallow(512), FPMlpDeep, ResidualMLP, FingerPrintCNN(Conv1d), HybridGCNFp |
| `DL_03` | MACCS keys 167 bits (+ graphs for the hybrid) | same 5 families on MACCS |
| `DL_04` | graphs + descriptors/chars as needed | PNA, GraphTransformer, GPS, SmilesTransformer, FTTransformer |
| `DL_05` | 78-dim graphs; MolCLR pretraining on unlabeled open-source chemistry SMILES (default ZINC250k) | GIN (scratch), GIN-MolCLR |

**Model count (deduplicated): 23 unique models.** A model trained in more than
one notebook/phase is counted **once**: DL02 and DL03 train the same five
fingerprint models (Morgan vs MACCS encoding), and DL05's `GIN (scratch)` and
`GIN-MolCLR` are the same GIN already counted in DL01 — only the initial weights
differ. Breakdown: 13 (DL01) + 5 (fingerprint families) + 5 (DL04) = **23**.
The notebooks execute 30 training slots in total because of those deliberate
re-runs. DL 04 adds five Tier-1 architectures (PNA — Corso et al. NeurIPS 2020;
GraphTransformer — TransformerConv, Shi et al. 2021; GPS — Rampášek et al.
*Nature Machine Intelligence* 2022; SmilesTransformer — Vaswani et al. 2017
char-level; FT-Transformer — Gorishniy et al. NeurIPS 2021). DL 05 adds
MolCLR self-supervised pretraining (Wang et al. *Nature Machine Intelligence*
2022, NT-Xent/SimCLR) then fine-tunes GIN scratch vs GIN-MolCLR head-to-head.
Graphormer and 3-D models are deliberately excluded (645 molecules is too few).
Full paper lineage is in every notebook's title markdown, chemistry primer and
`DL_PAPERS` dict. Advanced, data-hungry architectures that were reviewed but
deliberately **not** used on 645 compounds (Graphormer, GROVER, Uni-Mol,
3-D/equivariant models, chemical language models, full GraphGPS) are listed
with reasons in `MODEL_RESEARCH.md` → *Advanced / data-hungry models —
reviewed, not used*.

NB1 additionally offers two optional models: cell 9b runs the REAL deepchem
`dc.models.GCNModel` (guarded — skipped unless deepchem + tensorflow import;
the pure-torch DeepChemGCN replica in the roster always runs), and cell 10c
fine-tunes a pretrained chemical LM — ChemBERTa by default, MoLFormer-XL with
`DL_HF_MODEL=molformer` (`DL_USE_HF=1` to enable; pretrained weights stay
sensitivity-only and are never selected against the test set).

**Probability outputs:** classification predictions are **softmax
probabilities by default** — `prob_class_1 = P(Active)`, `prob_class_0 =
1 − prob_class_1`, the same `prediction / prob_class_0 / prob_class_1`
schema as `00_data/screening/screening_results.csv`. Cell 13 saves
`test_predictions.csv` (identifier, canonical_smiles, y_true, prediction,
prob_class_1, prob_class_0) in every notebook; regression mode saves
pIC50-scale predictions instead.

Every notebook follows the same 17-cell flow (Arjun sir's numbered-section
style, `# DL 0X: title` + `**Reference:**` + `**Pipeline:**` header):

0. Title markdown (reference, pipeline, data) — 1. **Chemistry primer**
   ("preload chemistry": graphs/bonds/descriptors, ECFP/MACCS bits) —
2. Setup + idempotent installs — 3. Reproducibility (`RANDOM_SEED=57`,
   `DL_SMOKE`/`DL_FAST`/`DL_TASK` env flags) — 4. **Full utilization** (GPU
   report, cudnn.benchmark, TF32, AMP, pin_memory, num_workers, prefetch,
   auto-batch, dual-GPU `run_parallel`) — 5. Load 645 compounds + EDA —
6. Featurize + leak-free split (464/52/129, canonical-SMILES overlap
   asserted 0) — 7. Model definitions with paper lineage — 8. Training engine
   (val-only early stopping, best-state restore, AMP, grad clip 5,
   ReduceLROnPlateau) — 9. Baseline training of all models (NB1 cell 9b: the
   REAL deepchem `GCNModel`, guarded) — 10. **Stepwise fine-tuning ladder**
   (A: lr → B: dropout×weight-decay → C: hidden 64/256 → D: batch;
   validation-only selection) — 10b. Optional pretrain warm-start —
11. 5-fold GroupKFold CV on scaffold groups — 12. **Y-randomization**
   (train labels permuted only; val/test untouched) — 13. **Test scored ONCE**
   + bootstrap 95% CI — 14. Summary JSON + leaderboard — 15. Screening
   (on by default; `DL_RUN_SCREENING=0` to skip; 154,647 COCONUT, chunked,
   Tanimoto AD; CSV loaded from Arjun sir's repo when not local) —
16. Diary block printer.

Task switch: `DL_TASK=cls` (default; Active=1, ROC-AUC primary) or `DL_TASK=reg`
(pIC50, RMSE primary) via the same model heads (`out_dim` 2 or 1).

## How to run

```bash
cd 06_deep_learning_models

# 1) Local smoke check (CPU, minutes): 24 test rows, 3 epochs, reduced grids
python3 run_smoke.py              # all five
python3 run_smoke.py DL_02        # one notebook

# 2) CPU-fast real-data check (all 645 rows, every model, reduced epochs)
python3 run_fast.py               # all five; writes fast_logs/ + outputs/*/fast/

# 3) Full protocol — upload the folder to Colab/Kaggle (GPU), Run All:
#    DL_SMOKE unset / DL_FAST unset -> 150 epochs, 10 Y-rand shuffles, 5-fold CV
```

### Environment flags

| Flag | Default | Effect |
|---|---|---|
| `DL_SMOKE=1` | `0` | Tiny CPU validation profile (24 test rows, 3 epochs, 2 CV folds, 2 Y-rand, 200 bootstraps) |
| `DL_FAST=1` | `0` | CPU real-data profile (all 645 rows, 8 epochs, 5-epoch FT, 2 CV folds, 2 Y-rand, 300 bootstraps) |
| `DL_TASK=reg` | `cls` | Regression on pIC50 instead of Active/Inactive classification |
| `DL_RUN_SCREENING=0` | `1` (on) | **Skip** the 154,647-molecule COCONUT screen + applicability domain (screening still auto-skips in smoke/fast) |
| `DL_INSTALL_DEEPCHEM=1` | `0` | Opt-in `pip install deepchem` for the real DeepChem GCN cell (needs TensorFlow; the pure-torch replica always runs) |
| `DL_USE_HF=1` | `0` | Opt-in Hugging Face ChemBERTa/MoLFormer fine-tuning (GPU + model download) |
| `DL_HF_MODEL=molformer` | `chemberta` | Choose the HF chemical LM |
| `DL_PRETRAIN_SOURCE=zinc250k` | `zinc250k` | DL05 unlabeled chemistry corpus: `zinc250k` (default, open-source download/cached), `chembl_small`, `coconut` (explicit opt-in only), or a local path/URL |
| `DL_PRETRAIN_MAX=30000` | profile default | Max unlabeled molecules for DL05 pretraining (smoke 1500, fast 4000, full 30000) |
| `DL_PRETRAIN_CACHE=...` | `outputs/dl05/<profile>/pretrain_cache` | Cache directory for downloaded pretraining corpora |
| `DL_CLEAR_OUTPUTS=1` | `0` | Clear notebook cell outputs after a successful `run_smoke.py`/`run_fast.py` run |
| `DL_EXTRA_MODELS=1` | `0` | Enable the extended 34-model roster in the existing notebooks (default 23-model results unchanged) |
| `DL_SSL_METHOD=graphcl\|joao\|gcc\|hu` | `molclr` | DL05 alternative label-free pretraining policy (only with `DL_EXTRA_MODELS=1`) |
| `DL_UNCERTAINTY=1` | `0` | Validation-only uncertainty diagnostics: deep ensembles, MC-dropout, conformal, evidential proxy, multi-task (only with `DL_EXTRA_MODELS=1`) |
| `DL_USE_3D=1` | `0` | Register 3-D models **only** when conformer coordinates (`d.pos`) exist and e3nn is installed; otherwise they are defined but skipped |
| `DL_EXTRA_FT=1` | `0` | Also run the 4-stage fine-tuning ladder for the extra roster (default: extras get baseline training only) |

### Portability (no manual setup)

- The setup cell installs missing packages with the **current interpreter**
  (`python -m pip`): `torch`, `torch_geometric`, `rdkit`, `scikit-learn`,
  `matplotlib`, `pandas`.
- Torch is installed from the **CPU wheel index on machines without a working
  `nvidia-smi`**, and from the default PyPI index on GPU machines/macOS; an
  existing torch install is left untouched.
- `rdkit-pypi` was replaced by the maintained `rdkit` package (Python 3.13
  wheels exist).
- Data paths fall back through `00_data/`, `Part_2/`, `Part_8/` (screening),
  `/content/...`, and then the public GitHub raw URLs: the 645-compound table
  from `Techbjd/new_method`, the **COCONUT screening library directly from
  Arjun sir's original repo**
  (`raw.githubusercontent.com/arjunpahi/Natural_MDM2_Inhibitor_Discovery_using_ML/main/Part_8/screening_results.csv`),
  so the notebooks also work on Colab/Kaggle without the repo.
- Outputs land in `outputs/dl0X/<profile>/` next to the notebook.
- Optional extras are **opt-in only**; a missing DeepChem/TensorFlow or
  Transformers install never stops the notebook.
- The notebooks run on CPU (slow for full mode) or CUDA; no code changes are
  needed when moving between machines.

## Split protocol (exact)

This matches `Part_20_Fixed/MDM2_Fixed_Pipeline_KAGGLE_ALL_CODE.py`:

1. Compute one Murcko scaffold per compound (non-isomeric).
2. Sort whole scaffold groups by size (largest first), then scaffold string.
3. Greedily add whole groups to the test set while the total is ≤ 129
   (`round(0.20 × 645)`).
4. The remaining 516 compounds are split **stratified by class** into
   train 464 / validation 52 with `random_state=57`.
5. Assert zero index overlap and zero canonical-SMILES overlap between train
   and test.

So the **test set is scaffold-disjoint and scored once**; validation is a
stratified random 10% of the non-test remainder (the canonical Part_20
protocol). It is not a second scaffold holdout. Every notebook prints
`464/52/129`, Active `262/29/52`, and scaffold overlap train↔test `0`.

## Validation status

### Smoke — 2026-11-22, CPU (outputs kept by default)

All five PASSED (`python3 run_smoke.py`, exit 0) after adding GINE/GraphSAGE/GatedGCN and ZINC250k default pretraining:

| Notebook | Time | Final test AUC (smoke) | Notes |
|---|---:|---:|---|
| `DL_01` | 0.7 min | 0.8667 | 13 models incl. GINE/GraphSAGE/GatedGCN; DeepChem cell skipped (no TensorFlow); HF disabled |
| `DL_02` | 0.3 min | 0.9704 | 5 fingerprint models |
| `DL_03` | 0.2 min | 0.7852 | 5 MACCS models |
| `DL_04` | 0.7 min | 0.8667 | 5/5 Tier-1 models |
| `DL_05` | 0.3 min | 0.9037 | MolCLR 1,500 mol × 2 epochs from ZINC250k (downloaded/cached, not screening COCONUT); warm-start 24/24 |

Smoke logs show `torch 2.13.0+cpu | CPU | 8 cpu threads`, `profile=smoke`,
split `464/52/129`, no tracebacks. Balanced accuracy can be 0.5 in smoke
because the tiny 3-epoch models put every test probability on one side of the
0.5 threshold; this is a smoke-scale calibration artifact, not a crash.

These are smoke-scale numbers on reduced epochs/rows — **not project results**.
Full-GPU numbers must come from a Colab/Kaggle run and be recorded in
`diary.md` + `docs/JOURNEY.md`.

### Extended 34-model roster — opt-in smoke (2026-11-23)

The opt-in paths were executed end-to-end on CPU via `run_smoke.py` (all exit 0):

| Notebook | Flags | Time | Registry added |
|---|---|---:|---|
| `DL_01` | `DL_EXTRA_MODELS=1 DL_UNCERTAINTY=1` | 9.2 min | +21 (13 local GNNs/readouts + 3 tabular + 5 SMILES LMs) |
| `DL_02` | `DL_EXTRA_MODELS=1` | 0.9 min | +16 (no `chars` features → SMILES LMs skipped) |
| `DL_03` | `DL_EXTRA_MODELS=1` | 0.6 min | +16 |
| `DL_04` | `DL_EXTRA_MODELS=1` | 1.3 min | +21 |
| `DL_05` | `DL_EXTRA_MODELS=1 DL_SSL_METHOD=graphcl DL_UNCERTAINTY=1` | 8.8 min | +13 graph-only (no `FEATS` matrix); GraphCL warm-start overrode MolCLR as intended |

The four 3-D classes are defined but **not registered** (no `d.pos`, no e3nn). All
34 named additions are present across the cells: 21 registry models, 4 SSL
policies (S07d), 4 uncertainty schemes plus multi-task (S16b), 4 guarded 3-D
classes. No full or fast run was performed with the extended roster (hours of
CPU), and no performance claim is made for any added model. In smoke,
`uncertainty.json` can show `AUC: NaN` because the 52-row validation fold is
single-class at this scale; that is a smoke artifact, not a result.

### Repeated/nested scaffold CV + enrichment (2026-11-23)

`python3 validation_package.py` (61.8 s clean run; 146 s when re-run under concurrent load, exit 0) regenerated the exact 464/52/129
scaffold split and ran 3 seeds × 5 outer `GroupKFold` folds with 3-fold inner
selection **on the 516 train+val compounds only**. The official test set was
never used for selection.

| Model family | Outer AUC | AP | BalAcc | MCC |
|---|---:|---:|---:|---:|
| descriptor logistic | 0.8525 ± 0.0533 | 0.8535 ± 0.1018 | 0.7721 ± 0.0577 | 0.5404 ± 0.1077 |
| ECFP4 logistic | 0.9016 ± 0.0360 | 0.8870 ± 0.0635 | 0.8288 ± 0.0643 | 0.6558 ± 0.1275 |
| ECFP4 random forest | 0.9158 ± 0.0338 | 0.9056 ± 0.0479 | 0.8420 ± 0.0480 | 0.6809 ± 0.1082 |

Descriptive re-scoring (once, no retraining) of the five existing fast test
prediction files gave AUC 0.9871–0.9989 (bootstrap 95% CI lower bounds
0.9652–0.9959), precision@10 = 1.00 in every notebook, EF1% = 2.48 (top 1% is
only 2 compounds, so unstable), and BEDROC(α=20) 0.9924–1.00. These are single
split, 52-actives/129 numbers — **not prospective performance** and not a basis
for choosing a model. The source predictions used are preserved in
`outputs/validation_package/source_fast_predictions/`.

### Screening + AD audit (2026-11-23)

`python3 screening_audit.py` (46.2 s, exit 0) audited the existing 154,647-row
COCONUT screen. Integrity: 0 missing/duplicate identifiers, 100% valid SMILES,
0 exact training-SMILES overlaps, 38,388 duplicate parent rows, 9 shared Murcko
scaffolds with training. Funnel (raw / unique parents / unique scaffolds):

| Stage | Raw | Parents | Scaffolds |
|---|---:|---:|---:|
| Input | 154,647 | 116,259 | 38,138 |
| `in_domain` | 38,761 (25.06%) | 28,781 | 12,594 |
| Nearest-active ≥ 0.40 | 81 | 45 | 23 |
| Margin ≥ 0.02 | unavailable | — | — |
| Agreement proxy (2/3 models) | 52 | 27 | 13 |
| Strict `maxT_464 ≥ 0.52` diagnostic | 14 | 7 | 1 |

Important correction: `Part_17/results/coconut_with_domain.csv::in_domain` is a
descriptor-box flag (MW/logP/HBA/HBD within mean ±2 SD), **not** the strict
Tanimoto AD cut. The margin gate cannot be recomputed because `maxT_inactive`
was not saved. All surviving compounds remain low-confidence computational
hypotheses, not hits.

An extended-roster fast run (`DL_EXTRA_MODELS=1 python3 run_fast.py`) was started
on 2026-11-23 but stopped by the user after 20.6 min while DL01 was still running.
**No extended fast results exist**; the partially rewritten DL01 tables matched
the saved deterministic 22:58 log, and the preserved prediction snapshots were
unchanged. Only the smoke checks above support the added models.

### CPU-fast real-data (2026-11-22, this PC, outputs kept)

`python3 run_fast.py` runs all 645 compounds with every model (8 epochs
baseline, 5-epoch fine-tune grid, 2 CV folds, 2 Y-randomization shuffles,
300 bootstraps, DL05 4,000 pretrain molecules × 3 epochs). It is a
**runtime/plumbing validation**, not a substitute for the full GPU protocol.

| NB | Status | Time | Final test (once, 129) | 95% CI AUC | Notes |
|----|--------|-----:|------------------------|------------|-------|
| `DL_01` | PASS | 10.5 min | AUC 0.9989, AP 0.9982, BalAcc 0.9610 | [0.9955, 1.0000] | 13 models incl. GINE/GraphSAGE/GatedGCN; best fast test GatedGCN; DeepChem real package skipped (no TensorFlow) |
| `DL_02` | PASS | 2.2 min | AUC 0.9933, AP 0.9858, BalAcc 0.9805 | [0.9753, 1.0000] | 5 Morgan-FP models |
| `DL_03` | PASS | 0.9 min | AUC 0.9899, AP 0.9786, BalAcc 0.9740 | [0.9701, 1.0000] | 5 MACCS models |
| `DL_04` | PASS | 9.5 min | AUC 0.9871, AP 0.9752, BalAcc 0.9353 | [0.9639, 1.0000] | 5 Tier-1 models |
| `DL_05` | PASS | 1.5 min | AUC 0.9981, AP 0.9972, BalAcc 0.8766 | [0.9941, 1.0000] | ZINC250k pretraining (4,000 mol × 3 epochs, not screening COCONUT); warm-start 24/24 tensors; shape-mismatch warm-starts skipped safely |

**Caveat (important):** the canonical 129-compound test set contains only 5
Murcko scaffolds (40 all-inactive, 34 all-inactive, 31 mostly active, 21 mostly
active, 3 mixed). The very high fast-profile AUCs likely reflect
scaffold/series separation on this small holdout, not prospective
molecule-level generalization. Treat these as plumbing/sanity results; use CV,
Y-randomization, AD, and external validation before making potency claims.

### Not exercised here

- Real DeepChem GCN (needs `tensorflow`; `DL_INSTALL_DEEPCHEM=1`).
- HF ChemBERTa/MoLFormer (needs GPU + model download; `DL_USE_HF=1`).
- Full 154k screening (`DL_RUN_SCREENING=1`; code is chunked but this CPU box
  has only ~2.3 GiB free RAM).
- Full 150-epoch GPU protocol.

## Scientific guardrails (same as the rest of the project)

- Split computed before any scaler/feature fitting; scaler fit on train only.
- Validation-only model selection, calibration, early stopping; the 129-mol
  test set is evaluated once per notebook (assert-guarded).
- Y-randomization never touches val/test labels.
- Screening hits are low-confidence hypotheses; ~99% of COCONUT lies outside
  the applicability domain (median screen↔train Tanimoto 0.217, cut 0.40) —
  always report AD columns with any ranking.
- A high predicted pIC50 is not a validated inhibitor.

## Regenerating the notebooks

`build_notebooks.py` is the source of truth. Edit the cell templates there,
then:

```bash
python3 build_notebooks.py      # rewrites all five .ipynb files
python3 run_smoke.py            # verify end-to-end on CPU
```

The builder validates nbformat, `ast.parse` for every code cell, and checks
that no `%%template%%` placeholders remain.
