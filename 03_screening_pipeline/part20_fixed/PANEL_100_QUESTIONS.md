# 100-Question Panel: What the MDM2 Screening Tables Mean and What to Do Next

This document records 100 questions asked by ten personas about the MDM2 Part_20
screening output. Each question has a short answer. The final section selects the
best answer path.

The two tables under discussion:

| Table | Filter | Nearest-training Tanimoto | Meaning |
|---|---|---:|---|
| Top 100 raw | none | 0.152–0.239 | The model's most potent calls are its least supported; extrapolation. |
| Top 100 "AD-aware" (legacy) | `in_AD >= 0.40` | 0.400–0.567 | A selected borderline subset, not strict AD. |
| Strict diagnostic | `ad_pass` | `>= 0.52` + active support + margin | 46 compounds; final diverse shortlist is 0. |

Key numbers: 645 training compounds; 464/52/129 scaffold split; GCN test
R²=0.686, RMSE=1.030; ensemble R²=0.611; 154,587 unique screening compounds;
median nearest-training Tanimoto 0.217; 886 pass legacy `in_AD` 0.40; 96 pass
strict AD; 46 pass `ad_pass`; 0 pass the final diverse shortlist.

Important correction: the full table has **10 exact screening rows** with
`maxT_464 = 1.0`, but they come from only **2 unique training compounds**:
nine copies of `CHEMBL803` (active, pIC50 7.699) and one `CHEMBL461462`
(inactive, pIC50 4.232). The earlier "2 exact matches" counted unique training
compounds, not screening rows. All 10 are flagged by the updated code and are
excluded from shortlist/diagnostic tables.

---

## 1. Mathematician

**Q1. Is the AD-aware table a new model?**
No. It is the same model predictions on a subset selected by `in_AD >= 0.40`.

**Q2. Why is its Tanimoto range higher?**
Because the filter is `maxT_464 >= 0.40`; 0.400–0.567 is guaranteed by construction.

**Q3. Can the AD-aware pIC50 be higher than the raw pIC50?**
For the same score and same candidate pool, no: the maximum of a subset cannot
exceed the maximum of the full set. A higher-looking table means a different
score, column, or run.

**Q4. Is this mathematical bias?**
It is selection/conditioning bias in the summary, not necessarily a changed
prediction. The table estimates a conditional quantity, not the library average.

**Q5. Are the 0.40 and 0.52 cutoffs mathematically special?**
No. They are policy thresholds. They must be justified by held-out error and
coverage, not by convention.

**Q6. What does 0/100 in AD prove?**
Not much by itself: 886/154,587 = 0.57%, so the expected overlap is 0.57 and
P(0) is about 0.56. The low Tanimoto values are the real red flag.

**Q7. What is the multiple-testing problem?**
Selecting the top 100 from 154,587 noisy predictions is an extreme-value
selection. The maximum is high even under a null model; report a permutation null.

**Q8. Is `rank_score = prediction − penalty × std` a probability?**
No. It is a ranking heuristic. The standard deviation is model disagreement, not
calibrated predictive uncertainty.

**Q9. What is the correct interpretation of the empty shortlist?**
It is an honest intersection of several strict gates. It is a valid negative
result, not evidence that no MDM2 inhibitor exists.

**Q10. What should be computed next?**
Bootstrap confidence intervals, error-versus-similarity calibration, and
threshold-sensitivity curves.

## 2. Physicist

**Q1. What is being measured?**
An assay-dependent pIC50, not a molecule-only constant. A 1.03 RMSE is about a
tenfold error in IC50.

**Q2. How large is the measurement noise?**
Biochemical assays often vary 2–3-fold; cell assays can vary more. A 0.5 pIC50
difference is about threefold and may be noise.

**Q3. Why is the raw top 100 suspicious?**
Its nearest-training similarity is only 0.152–0.239. The model is most confident
exactly where it has the least support.

**Q4. What is the winner's curse here?**
The maximum of 154,587 noisy predictions is biased upward; on re-measurement it
will tend to regress toward the mean.

**Q5. Does an AD filter fix extrapolation?**
No. It removes far compounds from the table; it does not make the remaining
predictions experimentally validated.

**Q6. What does "near training" mean physically?**
The fingerprint is similar to a known compound; it does not prove the same
binding mode, assay, or mechanism.

**Q7. What is the right control?**
Measure the known active and known inactive training compounds in the same assay
as the candidates.

**Q8. What is the right validation experiment?**
A dose-response MDM2–p53 biochemical assay with replicates, then orthogonal
target engagement and cell-based mechanism checks.

**Q9. Why is a zero shortlist acceptable?**
Because the evidence bar is a conjunction of strict filters. A null result is
more useful than a fabricated hit list.

**Q10. What is the next measurement?**
A small prospective set spanning the AD boundary, not the top 100 far compounds.

## 3. Polymath ("knows all")

**Q1. What is the central confusion?**
Mixing "similar to training" with "predicted to be potent." They are different
axes.

**Q2. What is the second confusion?**
Mixing a model prediction with a filtered/ranked table. The table is a view, not
a new model.

**Q3. What is the third confusion?**
Mixing a computational hypothesis with an experimental hit. Only an assay can
create a hit.

**Q4. What is the decision-theoretic view?**
With a high cost of false leads, an empty shortlist can be optimal. Do not lower
the gate just to produce names.

**Q5. What is the epistemic view?**
The model's uncertainty is largest where the chemistry is far; the output should
say "unknown," not assign a confident high value.

**Q6. What is the engineering view?**
The pipeline is doing its job: it computes AD, flags exact training compounds,
and refuses to call extrapolation a hit.

**Q7. What is the scientific view?**
The negative/audit result is publishable if framed as AD coverage, calibration,
and reproducibility—not as a discovery of inhibitors.

**Q8. What is the practical view?**
Pick a few diverse, synthesizable compounds near the AD boundary and test them.

**Q9. What is the ethical view?**
Do not present known training compounds or far extrapolations as novel hits.

**Q10. What is the one-sentence summary?**
The raw table is extrapolation; the legacy AD table is a selected borderline
subset; the strict table is nearly empty; the honest action is validation, not
ranking harder.

## 4. Medicinal Chemist

**Q1. Is a high predicted pIC50 a hit?**
No. A hit requires confirmed biochemical activity; a prediction is only a
hypothesis.

**Q2. What does pIC50 7 mean?**
About 100 nM in the assay used for that label. It is not a universal potency.

**Q3. Are the AD-aware compounds novel?**
Mostly no. Tanimoto 0.40–0.57 means close analogs, often substituent or ring
variants of training compounds.

**Q4. What about the exact training compounds?**
They are known molecules, not discoveries. Use them as assay controls and exclude
them from novelty claims.

**Q5. Why are there 10 exact rows but only 2 compounds?**
Nine screening entries are copies/analogs of the same active parent. Deduplicate
by parent and Murcko scaffold before counting.

**Q6. What chemistry risks matter?**
Lipophilicity, solubility, stereochemistry, reactive metabolites, and whether the
series is already patented or known.

**Q7. What is the first assay?**
Biochemical MDM2–p53 FP/TR-FRET or HTRF IC50, with positive and negative controls.

**Q8. What is the second assay?**
Orthogonal target engagement (SPR/ITC/thermal shift) and p53-WT versus p53-null
cell activity.

**Q9. What selectivity matters?**
MDMX/MDM4 selectivity and normal-cell cytotoxicity, especially p53-dependent
myelosuppression risk.

**Q10. What should be synthesized?**
A small diverse set near the AD boundary, not 100 far compounds.

## 5. Pharmacologist / ADME–Tox

**Q1. Is "in AD" the same as "likely active"?**
No. It means "similar to training chemistry"; confidence is not potency.

**Q2. Why does the AD table look better?**
It is conditioned on similarity and then ranked; that is selection, not improved
biology.

**Q3. What ADMET tests are needed?**
Solubility, permeability, plasma protein binding, microsomal stability, CYP
inhibition, hERG, and cytotoxicity.

**Q4. What is the MDM2-specific toxicity?**
p53-dependent effects in normal tissues, especially bone marrow suppression.

**Q5. What about selectivity?**
Check MDMX/MDM4 and related proteins; a potent MDM2 binder can be nonselective.

**Q6. Can QSAR predict toxicity here?**
Not reliably for a novel chemotype. Treat ADMET predictions as triage only.

**Q7. What is the right dose-response design?**
At least three replicates, multiple concentrations, and the same assay format as
the training labels where possible.

**Q8. What if the inactive training compound scores high?**
Audit the label, assay, and calibration; it is a warning sign.

**Q9. What if the active control scores low?**
The model or assay scale is misaligned; do not trust the candidate ranking.

**Q10. What is the go/no-go path?**
Confirmed biochemical activity → target engagement → cell mechanism →
selectivity → ADME/tox → in vivo, in that order.

## 6. AI/ML Engineer

**Q1. Is this regression or classification?**
Regression: continuous pIC50. There is no softmax probability.

**Q2. Why is test R² 0.686 not enough?**
It is a global fit metric on an analog-rich test set, not a hit-finding metric for
a far library.

**Q3. Is the ensemble better?**
No on this test: 0.611 versus 0.686 for GCN. Averaging can hurt; validate before
using it as the primary ranker.

**Q4. What is the applicability domain?**
A similarity-based guardrail. It flags when a query is far from training; it is
not a guarantee of accuracy.

**Q5. Is `std` a calibrated uncertainty?**
No. GCN/GIN/AFP share data and errors; their spread underestimates true error.

**Q6. Is the 80% interval reliable?**
Not under domain shift. It is based on 52 validation residuals and assumes
exchangeability.

**Q7. What is leakage here?**
The 10 exact training rows are memorization if presented as novel. The split
itself is scaffold-held-out and leak-free.

**Q8. What metric should drive screening?**
Precision@k, enrichment factor, recall of known actives, and calibration by
similarity bin—not R² alone.

**Q9. What is the right baseline?**
Morgan/MACCS RF regression on the identical split, plus ablations of AD and the
uncertainty penalty.

**Q10. What is the next model step?**
Freeze the test, calibrate AD empirically, and validate prospectively; do not
tune on the library.

## 7. Biostatistician

**Q1. How uncertain is R²=0.686 with n=129?**
Roughly ±0.10 by bootstrap; 0.686 versus 0.611 is not a meaningful difference.

**Q2. Is 52 validation compounds enough for calibration?**
It is thin; the interval quantiles are unstable.

**Q3. What is the base rate of AD?**
886/154,587 = 0.57% for legacy AD; strict AD is 96/154,587 = 0.06%.

**Q4. Why is 0/100 not proof of anything?**
The expected overlap is 0.57; zero is the most likely outcome under random
selection.

**Q5. How should thresholds be chosen?**
By held-out error/coverage, with a sensitivity table around each cut.

**Q6. Is the empty shortlist a statistical failure?**
No. It is a valid result of a pre-specified conjunction of gates.

**Q7. What about multiple comparisons?**
Ten exact matches, several thresholds, and multiple models create researcher
degrees of freedom; report all configurations or pre-register.

**Q8. How should exact matches be handled?**
Exclude them from novelty claims and from the shortlist; report them separately.

**Q9. What is the right confidence statement?**
"Low-confidence hypothesis" or "no candidate passed"; never "validated hit."

**Q10. What is the next statistical step?**
Repeated scaffold CV, bootstrap CIs, and a prospective assay with pre-registered
analysis.

## 8. PhD Methodologist

**Q1. What is the defensible contribution?**
The audit: AD coverage, leakage checks, calibration, and a reproducible negative
screen.

**Q2. What is not defensible?**
A 645-compound model with no shortlist presented as a discovery paper.

**Q3. What baselines are missing?**
Same-split Morgan/MACCS RF regression and ablations of the AD gate and ensemble.

**Q4. How many configurations were tried?**
Several AD cuts, rankers, and architectures; this is a researcher degree of
freedom and must be reported.

**Q5. Is the test set reused?**
It was scored once, but if library results or thresholds fed back into model
selection, it is effectively reused.

**Q6. What is the reproducibility gap?**
The Kaggle artifacts must be saved with hashes and a manifest; a pasted log is
not a reproducible output.

**Q7. What is the right publication frame?**
A methods/cautionary benchmark or a negative result, not a hit discovery.

**Q8. What would strengthen it?**
A small prospective validation set spanning the AD boundary.

**Q9. What is the main claim?**
"Most of the library is outside the model's validated domain; strict filtering
yields no diverse shortlist."

**Q10. What is the next paper step?**
Freeze code, rerun end-to-end, archive outputs, and write the limitations first.

## 9. Reviewer 2 (skeptic)

**Q1. Where is the novelty?**
Not in the empty shortlist; in the honest AD/leakage/calibration audit.

**Q2. Are the top 100 "hits"?**
No. They are predictions; the raw top 100 is outside AD.

**Q3. Is the AD-aware table cherry-picked?**
Yes, if presented as the main result without saying it is `in_AD >= 0.40`.

**Q4. Are the exact training compounds hidden?**
They are in the full table; the updated code flags all 10 rows and excludes them
from shortlist/diagnostic outputs.

**Q5. Is the inactive control a problem?**
If it ranks high, audit the label and calibration; an inactive should not look
like a potent hit.

**Q6. Are the claims calibrated?**
No; the 80% interval is not valid under domain shift, and `std` is disagreement.

**Q7. Is the code reproducible?**
Only after the artifacts and manifest are saved and the updated code is pushed.

**Q8. Is zero shortlist a failure?**
No. It is a negative result; lowering gates post hoc would be worse.

**Q9. What would make you accept a hit?**
Prospective dose-response, orthogonal target engagement, selectivity, and
replication—not a higher predicted pIC50.

**Q10. What should be removed from the paper?**
Any implication that the legacy AD-aware top 100 is a validated or novel hit
list.

## 10. Beginner and Science Teacher

**Q1. What is the model predicting?**
A potency score, pIC50. Higher means "works at a lower dose" in the assay it
learned from; it is not a promise.

**Q2. What is Tanimoto?**
A 0–1 similarity score. 1.0 means identical fingerprint; 0.2 means quite
different.

**Q3. Why does the AD table have higher similarity?**
Because it keeps only compounds with similarity at least 0.40. That is like
showing only the tall students and saying the class is tall.

**Q4. Is the AD table a better model?**
No. It is the same model, but a selected group of rows.

**Q5. Why is a high prediction outside AD dangerous?**
The model has not seen similar chemistry; it may sound confident and be wrong.

**Q6. Why are exact training compounds not discoveries?**
The model already knows them; it is remembering, not discovering.

**Q7. Why are there 10 exact rows but 2 compounds?**
Several screening entries are copies of the same parent; count unique chemistry,
not rows.

**Q8. Why is an empty shortlist okay?**
"I don't know" is a valid scientific answer. Forcing a list would be like
prescribing a random pill.

**Q9. What should the user do first?**
Do not chase the top 100. Check the AD, exclude exact training compounds, and
pick a few diverse compounds near the AD boundary for a real assay.

**Q10. What is the one-sentence lesson?**
A prediction is not a hit; a filtered table is not a new model; an honest zero
is better than a biased top 100.

---

## Best Answer Path

1. **Freeze the model and the split.** Do not tune on the test set or the library.
2. **Label the tables honestly.** Raw = extrapolation; legacy AD-aware = selected
   `in_AD >= 0.40` subset; strict diagnostic = `ad_pass`; final shortlist = 0.
3. **Exclude exact training compounds.** The full table has 10 exact rows from 2
   unique training compounds; flag them and exclude them from novelty/shortlist.
4. **Validate the AD cut.** Use the 129 held-out test compounds: bin by nearest
   training similarity (<0.40, 0.40–0.52, >=0.52) and compare error/coverage.
5. **Choose one research direction:**
   - **Audit/negative paper:** AD coverage, calibration, baselines, CIs,
     reproducibility. The empty shortlist is the result.
   - **Prospective discovery:** test 5–10 diverse compounds spanning the AD
     boundary, with the known active/inactive training compounds as controls.
6. **Do not manufacture hits.** If nothing passes, report the zero.

## What the Updated Code Does

The updated code adds, for every screening row:

- `nearest_train_id`
- `nearest_train_similarity`
- `nearest_train_class`
- `nearest_train_pIC50`
- `near_training` (`unknown`, `exact_or_near_exact`, `near_strict`,
  `borderline_legacy`, `far`)
- `similarity_informative`
- `known_training_compound`
- `novelty_claim_allowed`
- `prediction_status`

It also writes `training_overlap.csv`, excludes exact training compounds from
the shortlist and diagnostic top-100 tables, and keeps them visible in
`screening_full_regression.csv` with a warning.
