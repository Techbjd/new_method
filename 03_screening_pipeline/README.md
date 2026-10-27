# Screening Pipeline

- Arjun sir's original GNN screening is kept in the separate local clone and is not tracked here.
- `part11_gnn_screening/` — user GNN screening.
- `part14_ensemble/` — ensemble screening experiment.
- `part20_fixed/` — current authoritative regression/screening pipeline.

For Kaggle, use `part20_fixed/MDM2_Kaggle_SingleSource.py`: one input dataset
(`ro5_properties_filtered.csv`, `screening_results.csv`, `cuts.json`,
`common.py`) and one output folder (`/kaggle/working/mdm2_outputs`). The
notebook and the flat all-code script are development copies; the notebook is
legacy and does not include the latest fixes. The deliverable is
`shortlist_low_confidence.csv`; it contains low-confidence hypotheses, not
validated hits. Old `hits_T*` and `top100` tables are diagnostics only.

Run from the repo root or from `part20_fixed`; paths support both. Strict AD,
active-neighbour support, active margin, model agreement, and parent/scaffold
deduplication are mandatory.
