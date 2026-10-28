# Improvement Roadmap

Status: 2026-10-25. This is an audit of the existing code and diary, not new experimental evidence. Originals are untouched; see `docs/PROJECT_MAP.md` and `organized/MANIFEST.csv`.

## P0 — Scientific validity (do before making new claims)

1. **Freeze one label definition.** The diary and notebooks mix active = IC50 <= 100 nM, <= 1,000 nM, and pIC50 >= 6. Decide one primary definition, record it in config, and regenerate every dataset and metric from that definition. Treat pIC50 6–7 as its own middle class or exclude it explicitly; do not call it inactive in one place and active in another.
2. **Run repeated/nested scaffold CV for every model that is compared.** The current Part_20 evaluation repeats 5-fold-style scaffold CV only for the two Ridge baselines. GCN, GIN, AttentiveFP, HFooladi-GIN, and ChemBERTa are still evaluated on one 129-compound test split. Add retraining across repeated scaffold folds or clearly label the GNN result as a single-split estimate.
3. **Keep the official test split untouched.** Use validation only for early stopping, calibration, ensemble choice, and shortlist thresholds. The test set is for a final descriptive report once, not for choosing a model or a cutoff.
4. **Report enrichment, not only regression error.** For screening claims, always report precision@10, EF1%, BEDROC, and the number of actives in the test set. R2/RMSE/MAE alone do not show whether a ranking is useful.
5. **Separate prediction from evidence.** The shortlist is a set of low-confidence hypotheses. Do not call T1/T2/T3, top-100, or 116 RF predictions “hits” without assay validation. Keep `shortlist_low_confidence.csv` as the only recommended file.
6. **Re-run the Part_20 notebook end to end.** Cells 19–23 of the current notebook are unexecuted, and local `run_manifest.json`, `hits_T1.csv`, and screening outputs predate the refactor. Regenerate all outputs and confirm the audit passes before using any number.
7. **Fix or remove stale claims.** The diary’s 3,568-compound count is not the 645 modelling set; “MolCLR fine-tuning” was scratch training; Part_18 calibration used the test set; Part_17 regression failed; FULL_TRAIN “test R2” was a training monitor. Keep `docs/DISCOVERY_NOTES.md` as the correction record.

## P1 — Screening and applicability domain

8. **Keep strict AD gates.** Retain strict AD (`maxT_464 >= 0.52`), nearest-active support (`>= 0.40`), active–inactive margin (`>= 0.02`), and core-model agreement (range <= 0.70, std <= 0.40). Report how many compounds survive each gate so the funnel is auditable.
9. **Collapse chemotypes before counting.** Deduplicate by parent identifier and Murcko scaffold; report both row counts and unique-parent counts. GCN is stereo-blind, so stereoisomer rows must never inflate a hit count.
10. **Add a second novelty axis.** Max Tanimoto is not a full novelty measure. Add nearest-active distance, local training density, and a train-vs-library classifier confidence; report them beside AD, not as a replacement for it.
11. **Run orthogonal checks only after the computational shortlist.** Docking, pharmacophore/shape matching, and public bioactivity/literature review are optional filters. They must not be presented as experimental validation.
12. **Audit assay types and units.** ChEMBL records can mix IC50, Ki, Kd, and cell-based values. Add an assay-type column audit before treating the 645 labels as one homogeneous regression target.

## P2 — Reproducibility and engineering

13. **One config source.** Move paths, cutoffs, seeds, model flags, and output names into a single config file. Avoid duplicated constants across notebook cells and the flat script.
14. **One canonical entry point.** Keep the notebook for exploration and the flat Kaggle script for batch runs, but generate one from the other and test both. Do not hand-edit both.
15. **Pin the environment.** Record Python, RDKit, PyTorch, PyG, scikit-learn, pandas, and NumPy versions. The diary shows Colab/Kaggle failures from RDKit, CUDA, missing files, and import drift.
16. **Add smoke tests.** At minimum: import test, scaffold-split size test, prediction-schema test, shortlist-dedup test, and output-file existence test. The current synthetic shortlist and evaluation tests are a good start.
17. **Keep generated artifacts out of source.** Exclude caches, model binaries, and large prediction CSVs from the organized source tree; keep them in a separate `artifacts/` or external storage with a manifest and hash.
18. **Record run provenance.** Each run should save git commit, config hash, data hash, package versions, random seeds, and whether the test set was touched. `run_manifest.json` should be generated at the end, not carried over stale.
19. **Add schema checks at every handoff.** Assert required columns, finite predictions, legal ranges, unique IDs, and no overlap between train/val/test before any metric is computed.

## P3 — Research opportunities

20. **External test set.** The strongest next scientific step is a truly external set of MDM2 inhibitors with consistent assay types, not more architecture tuning on 645 compounds.
21. **Two-stage model.** A calibrated active/inactive classifier followed by a potency regressor may separate “likely active” from “high predicted pIC50” better than one regression head. Evaluate it with the same scaffold CV and enrichment metrics.
22. **Uncertainty-aware ranking.** Compare the current validation-residual interval with conformal prediction, quantile regression, or ensemble variance. Report coverage on held-out scaffolds, not just interval width.
23. **Prospective assay panel.** If resources allow, test a small diverse panel (10–20 compounds) that spans the strict-AD boundary, not only the top predicted compounds. Include inactive/negative controls from the same scaffolds.

## Do not repeat

- Do not tune on the test split or switch the primary model because its test score is higher.
- Do not use pseudo-labelling or self-training to create new “actives.”
- Do not present NNLS weights fitted on 52 validation compounds as stable evidence.
- Do not call top-100, T1, or 116 RF predictions validated hits.
- Do not reuse Part_18 calibrated probabilities; their temperature was fitted on the test set.
- Do not mix 78-dim Part_18 checkpoints with 124-dim DL features.
- Do not claim MolCLR fine-tuning, docking, MD, or MM-GBSA unless those runs are actually reproduced.
- Do not hide an empty shortlist. An empty list is a valid scientific result.
