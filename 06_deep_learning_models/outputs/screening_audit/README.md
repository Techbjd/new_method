# Screening / applicability-domain audit (2026-11-23)

**Provenance:** [USER] request / [GENERATED] implementation. **Generated:** 2026-11-23.

> **CAVEAT — computational hypotheses only, not assay validation.** No compound in any input or output is a validated hit; no prospective or experimental validation was performed.

## What was run

A read-only audit of existing CSV artifacts: schema/integrity checks, exact canonical-string training-overlap check, an AD/support/margin/agreement funnel, and parent + BDK-Murcko chemotype collapsing. No network, no model training, no source files touched.

Exact commands (from the repository root):

```bash
cd "/home/bijay/Desktop/already read paper/new_method/06_deep_learning_models"
python3 screening_audit.py
python3 -c 'import json; [json.load(open(p)) for p in ["outputs/screening_audit/funnel.json", "outputs/screening_audit/integrity.json"]]'
```

## Gate definitions and availability

| Gate | Definition | Status |
|---|---|---|
| Strict AD (applied) | `in_domain == True` from `Part_17/results/coconut_with_domain.csv` | available — BUT it is a descriptor box, not the maxT AD score |
| Nearest-active support | `maxT_343act >= 0.4` | available |
| Active-inactive margin | `maxT(active) - maxT(inactive) >= 0.02` | unavailable: required columns missing |
| Core-model agreement | range <= 0.7 and std <= 0.4 on pIC50 | partial: 2 of 3 core pIC50 model columns in saved CSVs |
| Documented strict AD (diagnostic) | `maxT_464 >= 0.52` | 96 rows scored as passing; counted only, not chained |

The `in_domain` column was **not** computed from `maxT_464`. The generating notebook (`Part_17/Part_17_Pretrain_Finetune_v1_regression.ipynb`, cell 21) defines it as MW, logP, HBA and HBD each inside mean +/- 2 SD of the 645 training compounds. The documented strict AD cut in `docs/IMPROVEMENTS.md` P1 #8, `docs/ERRORS_AND_FIXES.md` and `cuts.json` is `maxT_464 >= 0.52`; those scores exist only in `03_screening_pipeline/part20_fixed/screening_full_regression.csv` and are reported separately as a diagnostic. Do not conflate the two.

## Funnel (counts and percentages)

| Stage | Gate | Status | Rows in | Raw rows | % raw | Unique parents | % parents | Unique scaffolds | % scaffolds |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 00_input | All screening rows (denominator) | computed | 154,647 | 154,647 | 100.000 | 116,259 | 100.000 | 38,138 | 100.000 |
| 01_valid_smiles | RDKit-parseable canonical SMILES | computed | 154,647 | 154,647 | 100.000 | 116,259 | 100.000 | 38,138 | 100.000 |
| 02_strict_ad_in_domain | Strict AD gate (in_domain column) | computed: descriptor-box AD, NOT maxT_464 | 154,647 | 38,761 | 25.064 | 28,781 | 24.756 | 12,594 | 33.022 |
| 03_nearest_active_support | Nearest-active support >= 0.4 | computed | 38,761 | 81 | 0.052 | 45 | 0.039 | 23 | 0.060 |
| 04_active_inactive_margin | Active-inactive margin >= 0.02 | unavailable: required columns missing | 81 | n/a | n/a | n/a | n/a | n/a | n/a |
| 05_core_model_agreement | Agreement range <= 0.7 and std <= 0.4 | computed: PARTIAL proxy, 2/3 core pIC50 models available | 81 | 52 | 0.034 | 27 | 0.023 | 13 | 0.034 |
| 06_AD_maxT464_diagnostic | Documented strict AD maxT_464 >= 0.52 (diagnostic only) | diagnostic: counts only, not carried forward | 52 | 14 | 0.009 | 7 | 0.006 | 1 | 0.003 |
| 07_survivors_collapsed | Survivors after applied gates, chemotypes collapsed | computed (post-processing) | 52 | 52 | 0.034 | 27 | 0.023 | 13 | 0.034 |

Percentages use stage-00 denominators: 154,647 raw rows, 116,259 unique parents, 38,138 unique Murcko scaffolds. Stages with status `unavailable: required columns missing` have blank counts: the gate was NOT applied and no value was invented. `diagnostic` rows were counted but did not filter the chained funnel.

## Integrity findings

- Primary screen rows: **154,647**; columns: `identifier, canonical_smiles, prediction, prob_class_0, prob_class_1`.
- Missing identifiers: **0**; duplicate identifier rows: **0**; duplicate parent rows (version stripped): **38388**.
- Valid SMILES: **154,647/154,647** (100.00%).
- Exact training canonical-SMILES overlap: **0 screening rows / 0 unique SMILES** against 645 training rows (exact string match after str.strip() only; no canonicalisation, no tautomer/stereo normalisation). Shared Murcko scaffolds: **9** (analogue proximity only, not a label-leak proof).
- Part_17 `in_domain == True`: **38761** rows (25.06%); `maxT_464 >= 0.52`: **96** rows.
- Cross-file identifier/schema comparisons are in `integrity.json` -> `cross_file_checks`.

## Why chemotype collapsing matters

Identifier versions like `.1`/`.2` mark stereoisomers/analogs of one parent. The GCN graph featurisation is stereo-blind, so stereoisomer rows can carry identical predictions and inflate any raw hit count. Every funnel stage therefore reports raw rows, unique parents (identifier with trailing `.N` stripped) and unique BDK Murcko scaffolds (`includeChirality=False`). Acyclic molecules share the empty Murcko scaffold string; that is counted as one scaffold bucket, not as a missing value.

## Limitations

- No prospective, experimental or assay validation was done; outputs are computational hypotheses only.
- The applied AD gate is the Part_17 descriptor-box `in_domain` column, not the documented maxT Tanimoto AD; the latter is reported only as a diagnostic count.
- The active-inactive margin gate was NOT applied: no `active_margin`/`maxT_inactive` column exists in the available CSVs, and no proxy was substituted.
- Core-model agreement uses only the saved pIC50 model columns (pred_gcn_pic50, pred_gin_pic50); the documented three-model GCN/GIN/AttentiveFP agreement is not reconstructible from these artifacts.
- No retraining, recalibration or re-scoring was performed; all numbers are inherited from existing artifacts whose provenance is recorded in `integrity.json`.
- Exact-string training overlap is a conservative leak check: zero matches do not exclude scaffold/analogue similarity to labelled compounds.
- All downstream gates inherit any limitations of the upstream saved columns, including differences in row counts between the 154,647-row screen and 154,587-row Part_20 artifact.

## Warnings emitted during this run

- part20_screening_full: requested columns absent (['pred_afp_pic50', 'pred_hfgin_pic50', 'pred_chemberta_pic50', 'active_margin', 'margin', 'maxT_inactive', 'maxT_inact']); these gates may be unavailable

## Output files

- `outputs/screening_audit/funnel.csv` — stage table with counts/percentages.
- `outputs/screening_audit/funnel.json` — same funnel plus denominators and gate availability.
- `outputs/screening_audit/integrity.json` — schemas, row counts, overlap and AD definition evidence.
- `outputs/screening_audit/README.md` — this file.
