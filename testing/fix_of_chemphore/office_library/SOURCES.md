# Office library — MDM2 familiarization pool (SOURCES)

## What this is
`mdm2_familiarization_pool.csv` — 2,515 unique RDKit-valid canonical SMILES of MDM2–p53
compounds for **familiarization / pretraining only (chemical-space exposure)**.
NEVER for model evaluation or testing. Disjoint from our 645 by construction.

## Primary source (verified live, accessed 2026-10-16)
- Repo: https://github.com/MarciaC18/Identification-of-MDM2-P53-Inhibitors-Using-Machine-Learning-Guided-Screening
- Archive: Zenodo v1.1, published 2025-09-17, MIT License, Universidad de los Andes.
  Cite as: DOI **10.5281/zenodo.17143785**
  (Note: the repo README badge points to 10.5281/zenodo.17058139; the live v1.1
  record resolves to 10.5281/zenodo.17143785 — cite the latter.)
- Method basis (as cited in their README): Tran-Nguyen, V. K., Junaid, M., Simeon, S.,
  & Ballester, P. J. (2023). A practical guide to machine-learning scoring functions
  for structure-based virtual screening. *Nature Protocols*.
  Protocol code: https://github.com/vktrannguyen/MLSF-protocol (verified HTTP 200).

## Pool construction (script: /tmp/opencode/build_office_library.py)
1. Downloaded `MDM2-training-OTS-data.csv` (5,562) + `MDM2-test-OTS-data.csv` (1,754)
   and docked SDFs (train ~22.9 MB, test ~7.2 MB). CSVs carry ChEMBL IDs only —
   SMILES extracted from SDF entries joined on molecule `_Name` == ChEMBL ID.
2. SDF parse losses: 100 train + 41 test rows (mostly N-valence sanitization failures
   in docked poses) — dropped.
3. RDKit canonicalization; 0 additional invalid.
4. Collapsed 4,544 duplicate-SMILES rows: distinct ChEMBL IDs in their docked SDFs
   parse to only 2,060 (train) / 813 (test) unique SMILES — stereo/protonation
   variants from docking prep. First (their-train) representative kept.
5. Removed 116 exact canonical-SMILES overlaps with our 645
   (`Part_2/ro5_properties_filtered.csv`). Pool is disjoint from all supervised data.
6. Final: **2,515** (their-train 1,972 + their-test 543; annotated Active 1,925 /
   Inactive 590 + `potency` as shipped).

## Caveats (for the paper's Methods / Data section)
- SMILES derive from **docked poses** (protonation/tautomer artifacts possible,
  e.g. quaternary `[n+]`); stereo partly unrecoverable.
- 216/2,631 SMILES groups carry **conflicting Active/Inactive labels** and 221 have
  potency spread > 1.0 — their labels are pose/context-dependent (PLEC), not
  SMILES-determined. Labels here are **annotations only**; do not supervise on them.
- License of source: MIT. Our pool file: derived annotations; cite the Zenodo DOI.

## Suggested paper citation
> Additional MDM2–p53 chemical space for model familiarization (2,515 unique
> compounds, disjoint from the training set) was derived from the open MLSF-based
> MDM2 virtual-screening dataset (MarciaC18, Zenodo DOI 10.5281/zenodo.17143785,
> MIT), following Tran-Nguyen et al., Nat. Protoc. 2023.
