# Part_21 — Fresh Multitask Pipeline (Kaggle-ready)

Standalone successor experiment to Part_20. One self-contained script,
`mdm2_pipeline_v1.py`, runs scaffold splitting through multitask training,
top-50 output, and self-verification. It does **not** reuse any Part_20
predictions or Part_20 cache artifacts; the screening library is read as
structures only (identifier + canonical_smiles; old prediction /
prob_class_0 / prob_class_1 columns are dropped on load).

## What it does

1. Loads real MDM2 training data (645 compounds, pIC50 + Active/Inactive).
2. Type-safety pass: coerces pIC50 to float, validates label vocabulary,
   counts NaN / invalid SMILES / duplicates (reported, not silent).
3. Bemis-Murcko **scaffold split** 80/10/10 — scaffolds never span splits
   (verified live: 0 shared scaffolds during the smoke run).
4. Features only from free chemistry: RDKit Morgan FP (2048 bits, r=2)
   + 10 molecular descriptors. No labels, no identifiers in the matrix.
5. Separate regression (pIC50) and classification (Active vs Inactive)
   tracks, each with its own printed metrics block (RMSE/MAE/R2 for
   regression; AUC-ROC, AUC-PR, balanced accuracy, per-class probability
   means for classification).
6. Three model families:
   - XGBoost regressor + XGBoost classifier,
   - PyTorch multitask network (shared trunk, 2 heads),
   - optional **ChemBERTa fine-tuning** (two separate fine-tunes) —
     wrapped in `nn.DataParallel`, so a Kaggle session with 2× T4 trains
     on **both GPUs** (fallback: 1 GPU → CPU).
7. Screens the 154,647-molecule library, adds an applicability-domain gate
   (max Tanimoto similarity to training; `ad_cut_strict = 0.52` from
   cuts.json), and writes the TOP-50 tables:
   - `top50_regression.csv` (ranked by predicted pIC50)
   - `top50_classification.csv` (ranked by probability of being active)
   Both are also printed to stdout.
8. Writes `verification_report.json` + `run_manifest.json` (SHA256 of every
   artifact) into the output folder.

## Run on Kaggle (2 GPUs)

1. Create (or reuse) a Kaggle dataset with **one** of these callings:
   dataset named `mdm2-part21` holding the three inputs — or attach the
   repo-published files; the script falls back to the GitHub branch
   (`MDM2_PART21_GH_BASE`, default `main`) if the dataset is missing.
2. Paste the file into a notebook cell or run it as a script:
   ```
   !python mdm2_pipeline_v1.py
   ```
3. Outputs land in `/kaggle/working/mdm2_part21_outputs/`.

## Local verification

```
python3 verify_pipeline.py        # 13 checks, verbose PASS/FAIL, exit 0 on green
```

## Flags

| Flag | Meaning |
|---|---|
| `MDM2_PART21_SMOKE=1` | tiny dry run (150 train / 300 screening rows), no ChemBERTa |
| `MDM2_PART21_USE_CHEMBERTA=1` | force ChemBERTa even in smoke mode |
| `MDM2_PART21_SKIP_CHEMBERTA=1` | force ChemBERTa off |
| `MDM2_PART21_INPUT_DIR` / `MDM2_PART21_OUTPUT_DIR` | path overrides |
| `MDM2_PART21_EPOCHS` | override multitask-net epochs (default 150) |

## Guarantees checked by the built-in verifier

- train/val/test scaffold leakage = 0 (hard check, FAIL otherwise)
- old screening predictions cannot re-enter features (columns dropped)
- feature matrix = SMILES chemistry only (no label/identifier leakage)
- pIC50/label dtype consistency
- outputs land only in this folder; Part_20 stays byte-identical
  (verify_pipeline.py snapshots Part_20 mtimes before and after runs)
