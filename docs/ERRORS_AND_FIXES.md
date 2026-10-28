# Errors, failure modes, and fixes

This file records the main problems found while moving from the original
classification pipeline to the Part_20 regression/screening pipeline. It is a
practical history, not a claim that every run was reproduced locally.

## 1. Leakage and test-set misuse

| Problem | Why it is wrong | Fix used |
|---|---|---|
| Reusing pretrained Part_18/HFooladi checkpoints that had already seen the 645 labels | The test compounds were effectively in training | Retrain from scratch on the 464/52/129 scaffold split; do not reuse those weights |
| Choosing the primary model because it had the best test R² | That is test-set tuning | Select on validation only; test is scored once |
| Comparing models after seeing test results and then re-running selection | Same problem in a different order | Keep the validation-selected primary; report the better test model only as sensitivity |
| Calibrating on test or on all 645 compounds | Leaks test information into the reported predictions | Fit calibration on the 52 validation compounds only |

## 2. Ensemble instability

| Problem | Why it is wrong | Fix used |
|---|---|---|
| NNLS weights fitted on 52 validation compounds | The weights are unstable and overfit a tiny sample | Report a simple mean of calibrated GCN + GIN + AttentiveFP; keep NNLS only as a labelled legacy diagnostic |
| Treating the ensemble as a new tuned model | It can hide the primary model's weaknesses | Report the simple mean and the primary model separately |

## 3. Applicability-domain failure

| Problem | Evidence | Fix used |
|---|---|---|
| AD cutoff 0.40 was too permissive | Only 886/154,587 library rows passed; raw top-100 had 0/100 in AD | Strict AD cutoff `AD_CUT_STRICT=0.52` for the shortlist |
| Similarity to any training compound is not enough | A compound can be close to an inactive neighbour | Require nearest active similarity `maxT_343act >= 0.40` |
| No active-vs-inactive check | Some high predictions were closest to inactive compounds | Require `active_margin = maxT(active) - maxT(inactive) >= 0.02` |
| Max Tanimoto alone is a weak novelty measure | It ignores local density and neighbour labels | Add active/inactive IDs, active margin, AD score, and novelty score |

## 4. Duplicate chemotypes and stereo-blind models

| Problem | Evidence | Fix used |
|---|---|---|
| GCN predictions are identical across stereoisomers | Graph features ignore stereochemistry | Collapse stereoisomers/analogs before counting candidates |
| Top-100 lists were inflated by analogs | AD top-100 had 100 rows but only 56 unique parents; raw top-100 had 66 | Deduplicate by parent ID and Murcko scaffold |
| One scaffold could dominate the list | Repeated analogs look like independent hits | Keep one compound per parent and per scaffold cluster; cap at 12 |

## 5. Regression bias and over-confidence

| Problem | Evidence | Fix used |
|---|---|---|
| Test over-prediction | Mean signed bias −0.577 pIC50; low-potency bin −1.137 | Do not present absolute pIC50 as reliable; report ranks and confidence bands |
| Calibration slope away from 1 | Test calibration slope 0.734 | Fit calibration on validation only; report the slope |
| Low-potency test enrichment | 64/129 test compounds were in (0, 6] | Report per-bin bias, not only overall R² |
| Confidence interval missing | Point predictions looked more certain than they were | Add an 80% validation-residual interval and a lower-confidence column |

## 6. Screening and output errors

| Problem | Why it happened | Fix used |
|---|---|---|
| `screening_recompute_match` failed by ~0.01–0.03 pIC50 | Saved values were calibrated; recomputation compared raw values | Apply calibration before comparison; allow a small device/graph-rebuild tolerance (0.05) |
| `ChemBERTa` was skipped with `NameError: y_train_scaled is not defined` | Scaled target variable was not defined on that code path | Define the scaled targets before the optional branch; skip ChemBERTa gracefully if unavailable |
| Old `hits_T1.csv` looked like a hit list | 47 rows were only a threshold artifact | Rename to `threshold_T1_not_hits.csv`; add a clear not-a-hit label |
| Old top-100 tables looked like recommendations | They were diagnostic rankings, mostly extrapolation | Rename to `diagnostic_top100_*`; the deliverable is `shortlist_low_confidence.csv` |
| Local outputs were stale after code changes | Notebook/script were edited but not rerun | Treat local CSVs as stale until the notebook is rerun on Kaggle/Colab |

## 7. Evaluation gaps found during cleanup

| Gap | Consequence | Current status |
|---|---|---|
| Only one 129-compound test split | Test metrics have wide uncertainty | Bootstrap CIs added; full repeated GNN CV still not run |
| No early-enrichment metrics in the first report | R² alone does not show whether actives rank at the top | precision@10, EF1%, and BEDROC added |
| Model selection stability not reported | A single validation split can pick a lucky model | 2,000-sample validation-only bootstrap win rates added |
| HFooladi-GIN looked better on test | Could tempt test-based model switching | Kept as sensitivity only; primary remains validation-selected |
| Repeated CV covers only Ridge baselines | GNN fold-to-fold stability is still unknown | Documented limitation; full GNN repeated CV is the next improvement |

## 8. Engineering / reproducibility issues

| Problem | Fix / rule |
|---|---|
| Notebook and flat Kaggle script drift | Regenerate the flat script from notebook code cells after every source edit |
| Large caches and model binaries in the repo | Exclude `cache/`, `__pycache__`, model weights, and generated CSVs from the organized source tree |
| Silent optional-model failures | Print a clear `[skipped]` message and record the reason in the manifest/audit |
| No schema checks for new output columns | Audit checks now validate strict AD, active support, margin, model agreement, shortlist size, and confidence columns |
| Random seeds not centralized | Use `RANDOM_SEED=57`; pass seeds to splits, bootstrap, and training |

## 9. Reporting rules that must not be broken

- Do not call the shortlist “hits”; it is a set of low-confidence hypotheses.
- Do not recommend 100 compounds for assay; the shortlist is capped at 12.
- Do not use test performance to switch the primary model.
- Do not use pseudo-labelling or self-training on the screening library.
- Do not add tiers or calibration layers as a substitute for external evidence.
- An empty shortlist is a valid result; it means the strict filters did not find defensible candidates.
