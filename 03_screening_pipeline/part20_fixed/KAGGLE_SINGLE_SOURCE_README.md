# Kaggle single-source run

Use **`MDM2_Kaggle_SingleSource.py`**. It is the only Kaggle entry point you
need to copy. It reads one input dataset and writes every output to one folder.

## 1. Create one input dataset

Create a Kaggle dataset named `mdm2-single-source` with exactly these four
files at its root:

| File | Canonical source |
|---|---|
| `ro5_properties_filtered.csv` | `00_data/training/ro5_properties_filtered.csv` |
| `screening_results.csv` | `00_data/screening/screening_results.csv` |
| `cuts.json` | `00_data/config/cuts.json` |
| `common.py` | `02_gnn_models/part18_gnn_triage/common.py` |

The script searches the dataset root and a few common subfolders, but the
simplest layout is the table above. If a file is missing, the setup stops with
a clear message instead of silently downloading a different copy.

## 2. Add the script

Upload `MDM2_Kaggle_SingleSource.py` as a Kaggle script/notebook, or paste the
whole file into one notebook cell, and attach the `mdm2-single-source` dataset.
Do not add a second data source. The file is self-contained: it runs in the
same kernel (no kernel replacement), changes into a private work tree, and
collects the results at the end.

The defaults are:

```python
INPUT_DIR  = /kaggle/input/mdm2-single-source
OUTPUT_DIR = /kaggle/working/mdm2_outputs
```

Override them with `MDM2_INPUT_DIR` and `MDM2_OUTPUT_DIR` if your dataset slug
or folder names differ.

## 3. Run

Run the script normally. The runner:

1. validates and copies the four inputs into a private work tree;
2. runs the full Part_20 pipeline there;
3. copies every CSV/JSON/PNG/TXT/MD/PDF/HTML output into
   `/kaggle/working/mdm2_outputs/`;
4. writes `KAGGLE_OUTPUT_MANIFEST.json` with SHA256 hashes of the inputs and
   outputs.

The output folder is cleared at the start of each run so stale files from an
earlier run cannot masquerade as current results.

### Setup check without training

```bash
MDM2_DRY_RUN=1 python MDM2_Kaggle_SingleSource.py
```

This validates the input dataset and builds the work tree, then exits before
the pipeline starts.

## What this version fixes

- Active-neighbour evidence uses the **464 training compounds only**, not all
  645 compounds.
- The shortlist gate uses strict AD (`maxT_464 >= 0.52`), train-active support,
  positive active–inactive margin, and core-model agreement.
- Ranking uses `pred_ensemble_pic50 - 1.0 * model_std_core`, so uncertain
  predictions are penalized instead of promoted.
- The ensemble is a fixed simple mean of the pre-specified core models; the
  old NNLS diagnostic is not presented as fitted evidence.
- Tier labels and schema version are aligned (`schema_version = 4`).
- All shortlist/tier/diagnostic files keep the columns needed downstream.

## Reading the result

`screening_full_regression.csv` contains the full library with AD and ranking
columns. `shortlist_low_confidence.csv` is the only deliverable shortlist.
Old `hits_T*.csv` and `top100_*` files are diagnostics only.

**An empty or very small shortlist is a valid scientific result.** Only
96 of 154,647 library compounds are inside the strict 0.52 AD, and the
top-100 raw predictions are outside AD. Do not fill the shortlist with
extrapolation just to make it large.
